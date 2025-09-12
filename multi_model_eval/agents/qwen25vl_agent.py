#!/usr/bin/env python3
"""
Qwen2.5-VL Agent for Multi-Model Evaluation Framework

This agent is specifically designed for Qwen2.5-VL models, providing:
- Vision-language understanding capabilities
- Efficient image and text processing
- Navigation action generation
- Memory management for VLN tasks
"""

from typing import List, Dict, Any, Tuple
import re
import math
import copy
from PIL import Image
import numpy as np
import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration, 
    PreTrainedModel
)

from .processors.vision_language_processor import Qwen25VLProcessor
from .base_agent import BaseAgent, AgentConfig
from utility import detect_best_attention_implementation


def sample_and_pad_images(images, num_frames=8, width=640, height=480):
    """Sample and pad images to fixed number of frames."""
    frames = copy.deepcopy(images)
    
    # if len(frames) < num_frames:
    #     padding_frames = num_frames - len(frames)
    #     while len(frames) < num_frames:
    #         frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    # else:
    #     padding_frames = 0
    
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    
    return sampled_frames


class Qwen25VLAgent(BaseAgent):
    """Qwen2.5-VL agent implementation for vision-language navigation tasks.
    
    Features:
    - Automatic attention implementation detection
    - Model quantization support (4-bit/8-bit)
    - Multi-frame processing with history management
    - Efficient vision-language understanding
    - Habitat-Lab integration
    """
    @property
    def name(self) -> str:
        return "qwen25vl"

    def __init__(self, config: AgentConfig, *args, **kwargs):
        """Initialize Qwen2.5-VL agent.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        
        # Extract parameters from config
        params = config.agent_params or {}
        self.num_frames = params.get('num_frames', 8)
        self.num_future_steps = params.get('num_future_steps', 4)
        self.num_history = params.get('num_history', 8)
        
        # Canonical action names for annotations-style indexing
        self.actions2idx = {'STOP': 0, 'FORWARD': 1, 'LEFT': 2, 'RIGHT': 3}
        
        # Memory management
        self.rgb_list = []
        self.time_ids = []
        self.action_queue = []
        self.output_ids = None
        self.past_key_values = None
        # Load model and processor
        super().__init__(config, *args, **kwargs)

    def load_model_and_processor(self, config: AgentConfig, **kwargs) -> Tuple[PreTrainedModel, Qwen25VLProcessor]:
        """Load Qwen2.5-VL model and processor.
        
        Args:
            config: Agent configuration object
            
        Returns:
            Tuple of (model, processor)
        """
        print(f"Loading Qwen2.5-VL model from: {config.model_path}")
        params = config.agent_params or {}
        
        # Prepare model loading arguments
        model_kwargs = {
            'pretrained_model_name_or_path': config.model_path,
            'attn_implementation': detect_best_attention_implementation(),
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'device_map': config.device if config.device is not None else 'auto',
            'trust_remote_code': True,
        }
        
        # Add quantization if enabled
        quantization_bits = params.get('quantization_bits', None)
        if quantization_bits == 4:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            model_kwargs['quantization_config'] = qconf
            print(f"Using 4-bit quantization")
        elif quantization_bits == 8:
            qconf = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs['quantization_config'] = qconf
            print(f"Using 8-bit quantization")
        else:
            print(f"\033[91mWarning: Unsupported quantization bits: {quantization_bits}, falling back to no quantization\033[0m")
        
        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**model_kwargs)

        # Load processor
        processor = Qwen25VLProcessor.from_pretrained(
            config.model_path,
            model_max_length=config.model_max_length,
            trust_remote_code=True
        )
        
        print("✅ Qwen2.5-VL model and processor loaded successfully!")
        return model, processor

    def parse_actions(self, text: str) -> List[int]:
        """Parse action sequence from generated text.
        
        Args:
            text: Generated text from the model
            
        Returns:
            List of action indices
        """
        # Atomic actions
        # 0: STOP, 1: FORWARD (25cm), 2: LEFT (15deg), 3: RIGHT (15deg)
        s = text.lower()
        s = s.replace("↑", " forward ").replace("←", " left ").replace("→", " right ")

        # Patterns (use finditer to preserve order)
        turn_pat = re.compile(
            r"(turn|rotate|spin|face|veer)\s+(left|right)\s*(?:by\s*)?(?:(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees))?",
            re.IGNORECASE,
        )
        # Also support "left 30 degrees" / "right 45 degree" without an explicit verb
        turn_short_pat = re.compile(
            r"\b(left|right)\s*(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)\b",
            re.IGNORECASE,
        )
        forward_pat = re.compile(
            r"(move|go|walk|proceed|advance|head)\s*(?:straight|forward|ahead)?\s*(?:by|for|around|approximately|about)?\s*(\d+(?:\.\d+)?)?\s*(cm|centimeter|centimeters|m|meter|meters)?",
            re.IGNORECASE,
        )
        steps_pat = re.compile(r"(?:take\s*)?(\d+)\s*steps?", re.IGNORECASE)
        stop_pat = re.compile(r"\b(stop|halt|end|finish)\b", re.IGNORECASE)
        # Simple commands that might be missed if other patterns match
        forward_simple_pat = re.compile(r"\b(forward|straight|ahead)\b", re.IGNORECASE)
        turn_simple_pat = re.compile(r"\b(left|right)\b", re.IGNORECASE)

        raw_events: List[Tuple[int, int, Tuple[int, int]]] = []  # (start, end, (code, steps))

        # Collect turns with degrees
        for m in turn_pat.finditer(s):
            direction = m.group(2).lower()
            deg = m.group(3)
            degrees = float(deg) if deg is not None else 15.0
            steps = max(1, round(degrees / 15.0))
            code = 2 if direction == "left" else 3
            raw_events.append((m.start(), m.end(), (code, steps)))

        for m in turn_short_pat.finditer(s):
            direction = m.group(1).lower()
            degrees = float(m.group(2))
            steps = max(1, round(degrees / 15.0))
            code = 2 if direction == "left" else 3
            raw_events.append((m.start(), m.end(), (code, steps)))

        # Collect forward with distances
        for m in forward_pat.finditer(s):
            num = m.group(2)
            unit = (m.group(3) or "").lower()
            if num is None:
                # a plain move/go/walk without distance
                steps = 1
            else:
                val = float(num)
                if unit in {"m", "meter", "meters"}:
                    dist_cm = val * 100.0
                else:
                    # default to cm if unit missing or is cm
                    dist_cm = val
                steps = max(1, round(dist_cm / 25.0))
            raw_events.append((m.start(), m.end(), (1, steps)))

        # Collect steps specifically (avoid duplication if already captured by forward_pat)
        for m in steps_pat.finditer(s):
            num_steps = int(m.group(1))
            if num_steps > 0:
                raw_events.append((m.start(), m.end(), (1, num_steps)))

        # Collect stop
        for m in stop_pat.finditer(s):
            raw_events.append((m.start(), m.end(), (0, 1)))
            
        # Collect simple forward
        for m in forward_simple_pat.finditer(s):
            raw_events.append((m.start(), m.end(), (1, 1)))

        # Collect simple turns
        for m in turn_simple_pat.finditer(s):
            direction = m.group(1).lower()
            code = 2 if direction == "left" else 3
            raw_events.append((m.start(), m.end(), (code, 1)))
        
        # Filter out subsumed events
        events: List[Tuple[int, int]] = []
        if raw_events:
            filtered_events: List[Tuple[int, int, Tuple[int, int]]] = []
            for i, event1 in enumerate(raw_events):
                is_subsumed = False
                for j, event2 in enumerate(raw_events):
                    if i == j:
                        continue
                    # event1 is subsumed by event2 if it's contained within event2 and not identical
                    if event2[0] <= event1[0] and event1[1] <= event2[1] and \
                       (event2[1] - event2[0] > event1[1] - event1[0]):
                        is_subsumed = True
                        break
                if not is_subsumed:
                    filtered_events.append(event1)
            
            # Use start position for sorting, and extract event data
            events = [
                (event[0], event[2])
                for event in sorted(list(set(filtered_events)), key=lambda x: x[0])
            ]


        # If nothing matched, fallback heuristics
        if not events:
            if any(w in s for w in ["forward", "ahead", "straight"]):
                return [1]
            if "left" in s:
                return [2]
            if "right" in s:
                return [3]
            if "stop" in s:
                return [0]
            return [1]

        # Sort by text order and expand counts
        actions: List[int] = []
        stop_seen = False
        for _, (code, count) in events:
            if stop_seen:
                break
            actions.extend([code] * count)
            if code == 0:
                stop_seen = True

        # In case we created an empty list (shouldn't happen), default to forward
        return actions or [1]

    def reset(self, env_id: int):
        """Reset environment-related state."""
        self.rgb_list = []
        self.action_queue = []

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
        self.rgb_list.append(curr_rgb)
        print(f'step{step_id}: ', end='', flush=True)
        # Check if there are queued actions
        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            print(f"Using queued action: {action}, queue length: {len(self.action_queue)}", flush=True)
            return action
        
        past_and_current_rgbs = sample_and_pad_images(
            self.rgb_list, 
            num_frames=self.num_history
        )
        
        instruction = input_dict.get('instruction', '')
        # Create navigation prompt
        prompt_text = self._create_navigation_prompt(past_and_current_rgbs, instruction)
        with torch.inference_mode():
            # Process inputs using BaseProcessor interface
            processor_inputs = {
                'images': past_and_current_rgbs,
                'text': prompt_text
            }
            
            # Use prepare_from_inputs - the only allowed interface
            processor_kwargs = {'add_generation_prompt': True}
            if self.device is not None:
                processor_kwargs['device'] = self.device
            
            model_inputs = self.processor.prepare_from_inputs(
                processor_inputs,
                **processor_kwargs
            )
            input_ids = model_inputs['input_ids']
            # Generate response - use model_inputs directly as it's ready for model.generate()
            output_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True,
            )
            
            # Decode output using BaseProcessor decode method
            outputs = self.processor.decode_trimmed(output_ids, input_ids, skip_special_tokens=True)[0].strip()
        
        print("--------------------------------")
        print(f"llm output: {outputs}", flush=True)
        print("--------------------------------")
        # Parse complete action sequence from output
        actions = self.parse_actions(outputs)
        if not actions:
            actions = [1]  # Default to forward if no action parsed
        
        # Get the first action to execute now
        action = actions.pop(0)
        print(f"Current action: {action}", flush=True)
        
        # Queue remaining actions for future steps
        if len(actions) > 0:
            self.action_queue.extend(actions)
            print(f"Queued {len(self.action_queue)} additional actions: {self.action_queue}", flush=True)
        
        return action

    def _create_navigation_prompt(self, images: List[Image.Image], instruction: str) -> str:
        """Create navigation-specific prompt with conversation template."""
        # Create interleaved image tokens
        interleaved_images = "<|vision_start|><|image_pad|><|vision_end|>" * (len(images) - 1)
        
        # Create navigation question
        # question = (
        #     f"You have been given a video of historical observations:{interleaved_images}, "
        #     f'and current observation:<|vision_start|><|image_pad|><|vision_end|>\n '
        #     f'Your assigned task is: "{instruction}" '
        #     f"Analyze this series of images to decide your next action, which could be turning left or right by a specific degree, "
        #     f"moving forward a certain distance(cm), or stop if the task is completed."
        # )
        # conversation=[
        # {
        #   'role': 'system',
        #   'content': f'You are a intelligent robot programmed for vision-language navigation tasks.'
        #              f'You can not only understand the instructions from user to perform navigation tasks,'
        #              f'but alse see visual observations and understand spatial relationships to avoid collisions.'
        # },
        # {
        #     'role': 'user',
        #     'content': question
        # }]
        question = (
            f'You are an autonomous navigation assistant. Your task is: {instruction}.'
            'There are your historical observations:' + interleaved_images + '\n'
            'Your current observation is: <|vision_start|><|image_pad|><|vision_end|>\n'
            'Devise an action sequence to follow the instruction using the four actions: '
            'TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP.'
        )
            
        conversation=[
        {
          'role': 'system',
          'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': question
        }]
        return conversation