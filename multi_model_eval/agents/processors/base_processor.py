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
    def process_messages(self, messages: Union[str, Messages], *, padding: bool = True, add_generation_prompt: bool = True, add_system: bool = False) -> str:
        if isinstance(messages, str):
            text = messages
        else:
            try:
                if hasattr(self, 'apply_chat_template') and callable(getattr(self, 'apply_chat_template')):
                    text = self.apply_chat_template(  # type: ignore[attr-defined]
                        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, add_system=add_system
                    )
                elif hasattr(self, 'tokenizer') and getattr(self, 'tokenizer', None) is not None \
                        and getattr(self.tokenizer, 'apply_chat_template', None) is not None:
                    text = self.tokenizer.apply_chat_template(
                        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, add_system=add_system
                    )
                else:
                    raise Exception('No available chat template found on processor/tokenizer')
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

        # 对无 PAD 的分词器（如 GPT-2）设置合理的 padding 策略
        if padding and getattr(self, 'tokenizer', None) is not None:
            tok = getattr(self, 'tokenizer')
            if getattr(tok, 'pad_token', None) is None:
                eos_token = getattr(tok, 'eos_token', None)
                if eos_token is not None:
                    try:
                        tok.pad_token = eos_token  # type: ignore[assignment]
                    except Exception:
                        padding = False
                else:
                    padding = False

        return text, padding

    def decode_trimmed(
        self,
        generated_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> List[str]:
        """Decode generated token ids to strings.
        """
        if input_ids is not None:
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
        else:
            generated_ids_trimmed = generated_ids
        # 优先使用 ProcessorMixin 的 batch_decode（由 tokenizer 提供）
        if hasattr(self, 'batch_decode'):
            try:
                return self.batch_decode(  # type: ignore[attr-defined]
                    generated_ids_trimmed,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
            except Exception:
                pass

        # 回退到 tokenizer 的 batch_decode
        if getattr(self, 'tokenizer', None) is not None:
            return self.tokenizer.batch_decode(
                generated_ids_trimmed,  # type: ignore[arg-type]
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

        raise RuntimeError('No available decoder found on processor/tokenizer')

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