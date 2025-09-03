#!/usr/bin/env python3
"""
Qwen2.5-VL Agent for Multi-Model Evaluation Framework

This agent is specifically designed for Qwen2.5-VL models, providing:
- Vision-language understanding capabilities
- Efficient image and text processing
- Navigation action generation
- Memory management for VLN tasks
"""

import sys
import os
from typing import List, Dict, Any, Sequence, Optional, Union, Tuple
import re
import copy
import random
from collections import OrderedDict
import itertools
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    PreTrainedModel
)
from transformers.image_utils import to_numpy_array

from ..processors.vision_language_processor import Qwen25VLProcessor
from ..agents.base_agent import BaseAgent
from ..utility import detect_best_attention_implementation


class Qwen25VLAgent(BaseAgent):
    """Qwen2.5-VL agent implementation for vision-language navigation tasks.
    
    Features:
    - Automatic attention implementation detection
    - Model quantization support (4-bit/8-bit)
    - Multi-frame processing with history management
    - Efficient vision-language understanding
    - Habitat-Lab integration
    """

    def __init__(self, model_path: str, *args, **kwargs):
        """Initialize Qwen2.5-VL agent.
        
        Args:
            model_path: Path to the Qwen2.5-VL model checkpoint
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including:
                - quantization_bits: Quantization precision (4 or 8, default: 4)
                - quantization_type: Quantization type for 4-bit ('nf4', 'fp4', default: 'nf4')
                - num_frames: Number of frames to process (default: 8)
                - num_future_steps: Number of future steps to predict (default: 4)
                - num_history: Number of historical frames to maintain (default: 8)
                - auto_detect_attention: Whether to auto-detect best attention implementation (default: True)
                - force_attention_impl: Force specific attention implementation
                - model_max_length: Maximum sequence length (default: 2048)
        """
        # Initialize parameters
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frame and history parameters
        self.num_frames = kwargs.get('num_frames', 8)
        self.num_future_steps = kwargs.get('num_future_steps', 4)
        self.num_history = kwargs.get('num_history', 8)
        
        # Canonical action names for annotations-style indexing
        self.actions2idx = {'STOP': 0, 'FORWARD': 1, 'LEFT': 2, 'RIGHT': 3}
        
        # Memory management
        self.rgb_list = []
        self.time_ids = []
        self.action_seq = []
        self.output_ids = None
        self.past_key_values = None
        
        # Initialize conversation template
        self.conversation = [
            {
                "from": "human",
                "value": "You are a helpful assistant that helps with vision-language navigation tasks. You can see images and understand spatial relationships. Where should you go next to stay on track?"
            },
            {
                "from": "gpt",
                "value": "I'll help you navigate. Let me analyze the current situation and provide guidance."
            }
        ]
        
        # Load model and processor
        super().__init__(model_path, *args, **kwargs)

    def load_model_and_processor(self, model_path: str, **kwargs) -> Tuple[PreTrainedModel, Qwen25VLProcessor]:
        """Load Qwen2.5-VL model and processor.
        
        Args:
            model_path: Path to the model checkpoint
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (model, processor)
        """
        print(f"Loading Qwen2.5-VL model from: {model_path}")
        
        # Prepare model loading arguments
        model_kwargs = {
            'pretrained_model_name_or_path': model_path,
            'attn_implementation': detect_best_attention_implementation(),
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'device_map': 'auto',
            'trust_remote_code': True,
        }
        
        # Add quantization if enabled
        quantization_bits = kwargs.get('quantization_bits', None)
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

        model_max_length = kwargs.get('model_max_length', 2048)
        # Load processor
        hf_processor = AutoProcessor.from_pretrained(
            model_path,
            model_max_length=model_max_length,
            trust_remote_code=True
        )
        
        # Create Qwen25VLProcessor
        qwen_processor = Qwen25VLProcessor(processor=hf_processor)
        
        print("✅ Qwen2.5-VL model and processor loaded successfully!")
        return model, qwen_processor

    def parse_actions(self, text: str) -> List[int]:
        """Parse action sequence from generated text.
        
        Args:
            text: Generated text from the model
            
        Returns:
            List of action indices
        """
        # Define action patterns
        action_patterns = {
            r'\b(stop|halt|end|finish)\b': 0,  # STOP
            r'\b(forward|ahead|straight|move forward|go forward)\b': 1,  # FORWARD
            r'\b(left|turn left|go left|veer left)\b': 2,  # LEFT
            r'\b(right|turn right|go right|veer right)\b': 3,  # RIGHT
        }
        
        actions = []
        text_lower = text.lower()
        
        # Extract actions based on patterns
        for pattern, action_idx in action_patterns.items():
            matches = re.findall(pattern, text_lower)
            actions.extend([action_idx] * len(matches))
        
        # If no actions found, try to infer from context
        if not actions:
            if any(word in text_lower for word in ['forward', 'ahead', 'straight']):
                actions = [1]  # FORWARD
            elif any(word in text_lower for word in ['left', 'turn']):
                actions = [2]  # LEFT
            elif any(word in text_lower for word in ['right', 'turn']):
                actions = [3]  # RIGHT
            else:
                actions = [1]  # Default to FORWARD
        
        return actions

    def reset_for_env(self, env_id: int):
        """Reset cache for specified environment and synchronize model."""
        if hasattr(self.model, 'reset_for_env'):
            self.model.reset_for_env(env_id)
        self.output_ids = None
        self.past_key_values = None
        self.time_ids = []

    def reset(self, env_id: int):
        """Reset environment-related state."""
        self.reset_for_env(env_id)
        self.rgb_list = []
        self.action_seq = []

    def act(
        self,
        env_id: int,
        step_id: int,
        input_dict: Dict[str, Any]
    ) -> int:
        """Execute one generation step and parse actions, maintaining memory cache.
        
        Args:
            env_id: Environment identifier
            step_id: Current step identifier
            input_dict: Input dictionary containing:
                - instruction: Navigation instruction
                - rgb: RGB image observation
                - depth: Depth image observation
                - pose: Camera pose information
                - intrinsic: Camera intrinsic parameters
                
        Returns:
            Action index (0: STOP, 1: FORWARD, 2: LEFT, 3: RIGHT)
        """
        # Reset memory if needed
        if step_id != 0 and step_id % self.num_frames == 0:
            self.reset_for_env(env_id)
        
        # Extract instruction
        instruction = input_dict.pop('instruction', '')
        
        # Process inputs through processor
        processor_outputs = self.processor.prepare_from_inputs(input_dict)
        
        # Update memory
        self.time_ids.append(step_id)
        
        # Handle different input types
        if 'pixel_values' in processor_outputs:
            self.rgb_list.append(processor_outputs['pixel_values'])
        if 'depth' in input_dict:
            self.depth_list.append(input_dict['depth'])
        if 'pose' in input_dict:
            self.pose_list.append(input_dict['pose'])
        if 'intrinsic' in input_dict:
            self.intrinsic_list.append(input_dict['intrinsic'])
        
        # Return cached action if available
        if len(self.action_seq) != 0:
            return self.action_seq.pop(0)
        
        # Generate new action sequence
        if self.output_ids is None:
            # Prepare conversation with current instruction
            messages = copy.deepcopy(self.conversation)
            messages[0]["value"] = messages[0]["value"].replace(
                ' Where should you go next to stay on track?',
                f' Please devise an action sequence to follow the instruction: "{instruction}". Actions may include turning left or right by a certain degree, moving forward by a certain distance, or stopping once the task is complete.'
            )
            
            # Add history context if available
            if step_id != 0 and len(self.rgb_list) > 1:
                messages[0]["value"] += f' Consider your previous observations and current position.'
            
            # Remove image token placeholders
            messages[0]["value"] = messages[0]["value"].replace('<image>', '')
            
            add_system = True
        else:
            # Continue from previous generation
            messages = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False
        
        # Prepare model inputs
        model_inputs = self.processor.prepare_from_inputs(
            {'messages': [messages]}, 
            add_system=add_system
        )
        
        # Handle continuation from previous generation
        if self.output_ids is not None:
            input_ids = model_inputs['input_ids']
            if isinstance(input_ids, torch.Tensor):
                input_ids = torch.cat([self.output_ids, input_ids.to(self.output_ids.device)], dim=1)
            model_inputs['input_ids'] = input_ids
        
        # Move inputs to device
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=100,
                use_cache=True,
                return_dict_in_generate=True,
                past_key_values=self.past_key_values,
                pad_token_id=self.processor.processor.tokenizer.eos_token_id,
            )
        
        # Update state
        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values
        
        # Decode response
        decoded = self.processor.processor.tokenizer.decode(
            self.output_ids[0], 
            skip_special_tokens=True
        ).strip()
        
        # Parse actions
        self.action_seq = self.parse_actions(decoded)
        
        # Default to forward if no actions parsed
        if len(self.action_seq) == 0:
            self.action_seq = [1]  # FORWARD
        
        return self.action_seq.pop(0)
