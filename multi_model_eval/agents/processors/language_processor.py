"""Processor for pure language models using a HuggingFace tokenizer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase

from .base_processor import BaseProcessor, TensorDict


class LanguageProcessor(BaseProcessor):
    """Language-only processor that wraps a tokenizer.

    This processor converts prompts or chat-style messages to input ids
    suitable for text-only models.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer=tokenizer)

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
            inputs: Single input dict or list of input dicts containing 'messages'/'text'
            add_generation_prompt: Whether to add generation prompt
            add_system: Whether to add system prompt (currently unused)
            padding: Whether to enable padding
            device: Target device for tensors
            return_tensors: Format of returned tensors
            
        Returns:
            TensorDict ready for model.generate()
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
        """Prepare model-ready inputs from a single input dict."""
        messages = input_dict.get('messages', input_dict.get('text', None))
        if messages is None:
            raise ValueError('messages or text is required')
        text_batch, padding = self.process_messages(messages, padding=padding, add_generation_prompt=add_generation_prompt)
        model_inputs = self.tokenizer(
            text_batch,
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
        """Prepare model-ready inputs from multiple input dicts (batch mode)."""
        # Process all texts at once for efficiency
        text_batch = []
        for input_dict in batch_inputs:
            messages = input_dict.get('messages', input_dict.get('text', None))
            if messages is None:
                raise ValueError('messages or text is required')
            text, _ = self.process_messages(messages, padding=padding, add_generation_prompt=add_generation_prompt)
            text_batch.append(text)
        
        model_inputs = self.tokenizer(
            text_batch,
            padding=padding,
            return_tensors=return_tensors,
        )

        if device is not None:
            model_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in model_inputs.items()}

        return model_inputs
