#!/usr/bin/env python3
"""
NaVILA Agent for Multi-Model Evaluation Framework

This agent is specifically designed for NaVILA models (Navigation with Vision-Language Models),
providing vision-language navigation capabilities based on the NaVILA architecture.
"""

import os
import sys
import re
import copy
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch
from transformers import PreTrainedModel, ProcessorMixin
# NaVILA specific imports
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from .base_agent import BaseAgent, AgentConfig
from .processors import BaseProcessor


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """Sample and pad images to fixed number of frames."""
    frames = copy.deepcopy(images)
    
    if len(frames) < num_frames:
        padding_frames = num_frames - len(frames)
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    else:
        padding_frames = 0
    
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    
    return sampled_frames


class NaVILAProcessor(ProcessorMixin, BaseProcessor):
    """NaVILA processor that strictly follows BaseProcessor interface with single responsibility."""
    
    # 兼容 Transformers Processor 接口
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, tokenizer, image_processor, model_config, device=None, chat_template: Optional[str] = None):
        # 初始化 ProcessorMixin，自动挂载并校验组件
        ProcessorMixin.__init__(self, image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)
        # 轻量基类工具
        BaseProcessor.__init__(self)
        self.model_config = model_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 可选设置聊天模板
        if chat_template is not None:
            try:
                self.tokenizer.chat_template = chat_template
            except Exception:
                pass
    
    def prepare_from_inputs(
        self,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        *,
        add_generation_prompt: bool = True,
        add_system: bool = False,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare model-ready inputs from unified inputs following BaseProcessor interface.
        
        Args:
            inputs: Input dictionary or list of input dictionaries containing:
                - images: List[PIL.Image] - Image sequence (optional)
                - messages: List[Dict] - Chat messages (optional)
                - text: str - Direct text input (required if no messages)
            add_generation_prompt: Whether to add generation prompt
            add_system: Whether to add system prompt  
            padding: Whether to enable padding
            device: Target device for tensors
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary ready for model.generate() containing:
            - input_ids: Tokenized input with image tokens
            - images: Processed image tensor (if images provided)
        """
        # Handle batch inputs
        if isinstance(inputs, list):
            if len(inputs) != 1:
                raise NotImplementedError("Batch processing not yet implemented for NaVILA")
            inputs = inputs[0]

        result = {}
        # Extract text content
        text_content = self._extract_text_content(inputs, add_generation_prompt)
        if text_content:
            # Tokenize input with image tokens
            input_ids = tokenizer_image_token(
                text_content, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            )
            result['input_ids'] = input_ids
        
        # Process images if provided
        images = inputs.get('images', None)
        if images:
            images_tensor = process_images(
                images, 
                self.image_processor, 
                self.model_config
            ).to(device, dtype=torch.float16)
            result['images'] = images_tensor
        
        return result
    
    def _extract_text_content(self, inputs: Dict[str, Any], add_generation_prompt: bool) -> str:
        """Extract and format text content from various input formats."""
        
        # Priority 1: Direct text input
        if 'text' in inputs:
            return inputs['text']
        
        # Priority 2: Chat messages
        if 'messages' in inputs:
            text_content, _ = self.process_messages(
                inputs['messages'], 
                add_generation_prompt=add_generation_prompt
            )
            return text_content
        
        return None


class NaVILAAgent(BaseAgent):
    """NaVILA agent implementation."""
    @property
    def name(self) -> str:
        return "navila"

    def __init__(self, config: AgentConfig, *args, **kwargs):
        """Initialize NaVILA agent.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        
        # Extract parameters from config
        self.device = config.device
        params = config.agent_params or {}
        self.num_video_frames = params.get('num_video_frames', 8)
        self.conv_mode = params.get('conv_mode', "llama_3")
        self.max_new_tokens = params.get('max_new_tokens', 32)
        self.temperature = params.get('temperature', 0.0)
        
        # Action patterns for parsing
        self.action_patterns = {
            0: re.compile(r"\bstop\b", re.IGNORECASE),
            1: re.compile(r"\bis move forward\b", re.IGNORECASE),
            2: re.compile(r"\bis turn left\b", re.IGNORECASE),
            3: re.compile(r"\bis turn right\b", re.IGNORECASE),
        }
        
        # Environment state (single environment)
        self.past_rgbs = []  # list of RGB images
        self.action_queues = []  # list of queued actions
        
        # Load model and processor
        super().__init__(config, *args, **kwargs)
        
    def load_model_and_processor(self, config: AgentConfig, **kwargs) -> Tuple[PreTrainedModel, BaseProcessor]:
        """Load NaVILA model and processor."""
        # Get model name from path
        model_name = os.path.basename(os.path.normpath(config.model_path))
        
        # Load pretrained model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            config.model_path, model_name, device_map=self.device if self.device is not None else "auto"
        )
        
        # Create processor
        processor_device = self.device if self.device is not None else model.device
        processor = NaVILAProcessor(tokenizer, image_processor, model.config, device=processor_device)
        
        return model, processor
    
    def act(self, env_id: int, step_id: int, input_dict: Dict[str, Any]) -> int:
        """Generate navigation action based on current observation and instruction.
        
        Args:
            env_id: Environment ID (ignored, single environment only)
            step_id: Current step ID
            input_dict: Input dictionary containing:
                - rgb: Current RGB observation
                - instruction: Navigation instruction text
                
        Returns:
            action: Integer action (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)
        """
        curr_rgb = input_dict.get('rgb')
        # Convert numpy array to PIL Image if needed
        if isinstance(curr_rgb, np.ndarray):
            curr_rgb = Image.fromarray(np.uint8(curr_rgb)).convert("RGB")
        self.past_rgbs.append(curr_rgb)

        # Check if there are queued actions
        if len(self.action_queues) > 0:
            action = self.action_queues.pop(0)
            print(f"Using queued action: {action}, queue length: {len(self.action_queues)}")
            return action
        
        past_and_current_rgbs = sample_and_pad_images(
            self.past_rgbs, 
            num_frames=self.num_video_frames
        )
        
        instruction = input_dict.get('instruction', '')
        # Create navigation prompt
        prompt_text = self._create_navigation_prompt(past_and_current_rgbs, instruction)
        
        # Prepare inputs for the model
        processor_inputs = {
            'images': past_and_current_rgbs,
            'text': prompt_text,
            'conv_mode': self.conv_mode,
        }
        
        # Use the processor to prepare model inputs, moving to the correct device
        model_inputs = self.processor.prepare_from_inputs(
            processor_inputs,
            device=self.device,  # Pass device, processor will handle if it's None
        )
        
        # Add stopping criteria
        model_inputs['stopping_criteria'] = self._create_stopping_criteria(model_inputs['input_ids'])
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        
        # Decode output using BaseProcessor decode method
        outputs = self.processor.decode_trimmed(output_ids, skip_special_tokens=True)[0].strip()
        
        print(f"Model output: {outputs}")
        
        # Parse complete action sequence from output
        actions = self.parse_actions(outputs)
        if not actions:
            actions = [1]  # Default to forward if no action parsed
        
        # Get the first action to execute now
        action = actions[0]
        print(f"Current action: {action}")
        
        # Queue remaining actions for future steps
        if len(actions) > 1:
            self.action_queues.extend(actions[1:])
            print(f"Queued {len(actions) - 1} additional actions: {actions[1:]}")
        
        return action
    
    def _create_navigation_prompt(self, images: List[Image.Image], instruction: str) -> str:
        """Create navigation-specific prompt with conversation template."""
        # Create interleaved image tokens
        interleaved_images = "<image>\n" * (len(images) - 1)
        
        # Create navigation question
        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. '
            f'Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        
        # Use conversation template
        if self.conv_mode in conv_templates:
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        else:
            return question
    
    def _create_stopping_criteria(self, input_ids: torch.Tensor) -> List:
        """Create stopping criteria for text generation."""
        stopping_criteria = []
        
        # Try to create conversation-aware stopping criteria
        if self.conv_mode in conv_templates:
            conv = conv_templates[self.conv_mode].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria.append(
                KeywordsStoppingCriteria(keywords, self.processor.tokenizer, input_ids)
            )
        else:
            # Fallback to generic stopping criteria
            keywords = ["</s>", "<|im_end|>", "<|endoftext|>"]
            for keyword in keywords:
                try:
                    keyword_ids = self.processor.tokenizer.encode(keyword, add_special_tokens=False)
                    if keyword_ids:
                        stopping_criteria.append(
                            KeywordsStoppingCriteria([keyword], self.processor.tokenizer, input_ids)
                        )
                except:
                    continue
        
        return stopping_criteria
    
    def parse_actions(self, output_text: str) -> List[int]:
        """Parse complete action sequence from model output text.
        
        Analyzes the model output to extract the primary action and any additional
        actions needed for multi-step movements (e.g., multiple forward steps for
        longer distances, multiple turns for larger angles).
        
        Args:
            output_text: Generated text from model
            
        Returns:
            List of action integers representing the complete action sequence
        """
        actions = []
        
        # Find the primary action
        primary_action = None
        for action, pattern in self.action_patterns.items():
            if pattern.search(output_text):
                primary_action = action
                break
        
        if primary_action is None:
            return []  # No action found
        
        # Add primary action
        actions.append(primary_action)
        
        # Parse multi-step movements and add additional actions
        match primary_action:
            case 1:  # Forward movement
                try:
                    match_result = re.search(r"move forward (\d+) cm", output_text)
                    distance = int(match_result.group(1)) if match_result else 25
                except:
                    distance = 25
                
                # Quantize to 25cm steps
                if (distance % 25) != 0:
                    distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                
                # Add additional forward actions
                additional_steps = int(distance // 25) - 1
                actions.extend([1] * additional_steps)
                    
            case 2:  # Left turn
                try:
                    match_result = re.search(r"turn left (\d+) degree", output_text)
                    degree = int(match_result.group(1)) if match_result else 15
                except:
                    degree = 15
                
                # Quantize to 15 degree steps
                if (degree % 15) != 0:
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                
                # Add additional left turn actions
                additional_turns = int(degree // 15) - 1
                actions.extend([2] * additional_turns)
                    
            case 3:  # Right turn
                try:
                    match_result = re.search(r"turn right (\d+) degree", output_text)
                    degree = int(match_result.group(1)) if match_result else 15
                except:
                    degree = 15
                
                # Quantize to 15 degree steps
                if (degree % 15) != 0:
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                
                # Add additional right turn actions
                additional_turns = int(degree // 15) - 1
                actions.extend([3] * additional_turns)
                
            case _:  # Default case (e.g., case 0 for stop)
                pass  # No additional actions needed
        
        # Log the parsed action sequence
        if len(actions) > 1:
            print(f"Parsed multi-step action sequence: {actions} (total: {len(actions)} steps)")
        
        return actions
    
    def reset(self, env_id: int = None):
        """Reset environment state."""
        self.past_rgbs = []
        self.action_queues = []
        print("Reset navigation state")