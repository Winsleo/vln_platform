"""Base processor interface for unified multimodal preprocessing.

Defines an abstract interface that prepares model-ready inputs from
raw user data (text, images, video). Implementations must return a dict
that can be passed directly to a HuggingFace model's generate method.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Union

import torch


TensorDict = Dict[str, Any]
Messages = Sequence[Dict[str, Any]]


class BaseProcessor(abc.ABC):
    """Abstract base class for all processors.

    Implementations should convert raw inputs into model-ready tensors
    for generation. The returned dict is expected to be directly
    consumable by model.generate(**returned_dict).
    """

    def __init__(
        self,
        *,
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else getattr(processor, 'tokenizer', None)
        self.image_processor = image_processor if image_processor is not None else getattr(processor, 'image_processor', None)

    def process_messages(self, messages: Union[str, Messages], *, padding: bool = True, add_generation_prompt: bool = True) -> List[str]:
        if isinstance(messages, str):
            text = messages
        else:
            # Try chat template if available
            try:
                if getattr(self,'tokenizer', None) is not None \
                and getattr(self.tokenizer, 'apply_chat_template', None) is not None:
                    text = self.tokenizer.apply_chat_template(
                        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
                    ) 
                elif getattr(self,'processor', None) is not None \
                and getattr(self.processor, 'apply_chat_template', None) is not None:
                    text = self.processor.apply_chat_template(
                        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
                    )
                else:
                    raise Exception('No available chat template found on tokenizer or processor')
            except Exception:
                # Fallback: concatenate plain text segments
                parts: List[str] = []
                for msg in messages:
                    role = msg.get('role', None)
                    if role is not None:
                        parts.append(str(role))
                    content = msg.get('content') or msg.get('value') or ''
                    if isinstance(content, list):
                        for it in content:
                            if it.get('type') == 'text':
                                parts.append(str(it.get('text', '')))
                    else:
                        parts.append(str(content))
                text = '\n'.join(parts)

        # Ensure a valid padding strategy for tokenizers without pad token (e.g., GPT-2)
        if padding:
            if getattr(self.tokenizer, 'pad_token', None) is None:
                eos_token = getattr(self.tokenizer, 'eos_token', None)
                if eos_token is not None:
                    # Use EOS as PAD for batching purposes
                    self.tokenizer.pad_token = eos_token  # type: ignore[assignment]
                else:
                    # As a last resort, disable padding
                    padding = False

        return text, padding

    def decode(
        self,
        generated_ids: torch.Tensor,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> List[str]:
        """Decode generated token ids to strings.
        """
        if getattr(self, 'tokenizer', None) is not None:
            return self.tokenizer.batch_decode(
                generated_ids,  # type: ignore[arg-type]
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

        if getattr(self, 'processor', None) is None or getattr(self.processor, 'batch_decode', None) is None:
            raise RuntimeError('No available decoder found on processor/tokenizer')

        return self.processor.batch_decode(
            generated_ids,  # type: ignore[arg-type]
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    @abc.abstractmethod
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
        """Prepare model-ready inputs from unified inputs.
        
        Args:
            inputs: Single input dict or list of input dicts containing 'messages'/'text' and optionally other modalities
            add_generation_prompt: Whether to add generation prompt
            add_system: Whether to add system prompt
            padding: Whether to enable padding
            device: Target device for tensors
            return_tensors: Format of returned tensors
            
        Returns:
            TensorDict ready for model.generate()
            
        Examples:
            # Single input
            inputs = {'messages': [{"role": "user", "content": "Hello"}]}
            result = processor.prepare_from_inputs(inputs)
            
            # Batch input
            inputs = [
                {'messages': [{"role": "user", "content": "Hello"}]},
                {'messages': [{"role": "user", "content": "Hi"}]}
            ]
            result = processor.prepare_from_inputs(inputs)
        """