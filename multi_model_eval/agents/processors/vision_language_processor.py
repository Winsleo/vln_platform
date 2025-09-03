"""Processor for vision-language models using a HuggingFace AutoProcessor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .base_processor import BaseProcessor, TensorDict
from qwen_vl_utils import process_vision_info


class Qwen25VLProcessor(BaseProcessor):
    """Vision-language processor that wraps an AutoProcessor.

    The provided processor must support both text and vision modalities.
    This implementation expects chat-style messages compatible with
    processor.apply_chat_template.
    """

    def __init__(
        self,
        processor: Optional[Any] = None,
    ) -> None:
        super().__init__(processor=processor)

    def prepare_from_inputs(
        self,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        *,
        add_generation_prompt: bool = True,
        add_system: bool = False,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: str = 'pt',
    ) -> TensorDict:
        """Prepare model-ready inputs from unified inputs (single or batch).
        
        Args:
            inputs: Single input dict or list of input dicts containing 'messages'/'text' and optionally 'images'/'videos'
            add_generation_prompt: Whether to add generation prompt
            add_system: Whether to add system prompt (currently unused)
            padding: Whether to enable padding
            device: Target device for tensors
            return_tensors: Format of returned tensors
            
        Returns:
            TensorDict ready for model.generate()
            
        Examples:
            # Single input
            inputs = {'messages': [{"role": "user", "content": [{"type": "image", "image": "path1.jpg"}, {"type": "text", "text": "What's this?"}]}]}
            result = processor.prepare_from_inputs(inputs)
            
            # Batch input
            inputs = [
                {'messages': [{"role": "user", "content": [{"type": "image", "image": "path1.jpg"}, {"type": "text", "text": "What's this?"}]}]},
                {'messages': [{"role": "user", "content": "Hello"}]}
            ]
            result = processor.prepare_from_inputs(inputs)
        """
        # Check if inputs is a list (batch mode) or single dict
        if isinstance(inputs, list):
            return self._prepare_batch_inputs(
                inputs,
                add_generation_prompt=add_generation_prompt,
                add_system=add_system,
                padding=padding,
                device=device,
                return_tensors=return_tensors
            )
        else:
            return self._prepare_single_input(
                inputs,
                add_generation_prompt=add_generation_prompt,
                add_system=add_system,
                padding=padding,
                device=device,
                return_tensors=return_tensors
            )

    def _prepare_single_input(
        self,
        input_dict: Dict[str, Any],
        *,
        add_generation_prompt: bool = True,
        add_system: bool = False,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: str = 'pt',
    ) -> TensorDict:
        """Prepare model-ready inputs from a single input dict.
        
        Args:
            input_dict: Dict containing 'messages' or 'text', and optionally 'images'/'videos'
            add_generation_prompt: Whether to add generation prompt
            add_system: Whether to add system prompt (currently unused)
            padding: Whether to enable padding
            device: Target device for tensors
            return_tensors: Format of returned tensors
            
        Returns:
            TensorDict ready for model.generate()
        """
        messages = input_dict.get('messages', input_dict.get('text', None))
        if messages is None:
            raise ValueError('messages or text is required')
        
        text_batch, padding = self.process_messages(messages, padding=padding, add_generation_prompt=add_generation_prompt)
        image_inputs = None
        video_inputs = None
        if 'images' in input_dict:
            image_inputs = input_dict['images']
            if isinstance(image_inputs, Sequence):
                image_inputs = list(image_inputs)
            else:
                image_inputs = [image_inputs]
        elif 'videos' in input_dict:
            video_inputs = input_dict['videos']
            if isinstance(video_inputs, Sequence):
                video_inputs = list(video_inputs)
            else:
                video_inputs = [video_inputs]
        else:
            image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = self.processor(
            text=text_batch,
            images=image_inputs,
            videos=video_inputs,
            padding=padding,
            return_tensors=return_tensors,
        )

        if device is not None:
            model_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in model_inputs.items()}

        return model_inputs

    def _prepare_batch_inputs(
        self,
        batch_inputs: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        add_system: bool = False,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: str = 'pt',
    ) -> TensorDict:
        """Prepare model-ready inputs from multiple input dicts (batch mode).
        
        Args:
            batch_inputs: List of input dicts, each containing 'messages'/'text' and optionally 'images'/'videos'
            add_generation_prompt: Whether to add generation prompt
            add_system: Whether to add system prompt (currently unused)
            padding: Whether to enable padding
            device: Target device for tensors
            return_tensors: Format of returned tensors
            
        Returns:
            TensorDict ready for model.generate()
        """
        # Process text for each input
        text_batch = []
        all_images = []
        all_videos = []
        
        for single_input in batch_inputs:
            messages = single_input.get('messages', single_input.get('text', None))
            
            # Process text
            text, padding = self.process_messages(messages, padding=padding, add_generation_prompt=add_generation_prompt)
            text_batch.append(text if text else '')
            
            # Process images/videos for this input
            if 'images' in single_input:
                images = single_input['images']
                if images is not None:
                    all_images.extend(images if isinstance(images, list) else [images])
            elif 'videos' in single_input:
                videos = single_input['videos']
                if videos is not None:
                    all_videos.extend(videos if isinstance(videos, list) else [videos])
            else:
                # Extract from messages
                image_inputs, video_inputs = process_vision_info(messages)
                if image_inputs is not None:
                    all_images.extend(image_inputs if isinstance(image_inputs, list) else [image_inputs])
                if video_inputs is not None:
                    all_videos.extend(video_inputs if isinstance(video_inputs, list) else [video_inputs])
        
        # Prepare final inputs
        final_images = all_images if all_images else None
        final_videos = all_videos if all_videos else None
        
        model_inputs = self.processor(
            text=text_batch,
            images=final_images,
            videos=final_videos,
            padding=padding,
            return_tensors=return_tensors,
        )

        if device is not None:
            model_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in model_inputs.items()}

        return model_inputs
