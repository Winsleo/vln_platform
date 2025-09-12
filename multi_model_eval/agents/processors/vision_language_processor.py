"""Processor for vision-language models using a HuggingFace AutoProcessor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import Qwen2_5_VLProcessor

from .base_processor import BaseProcessor, TensorDict, Messages
from qwen_vl_utils import process_vision_info


class Qwen25VLProcessor(Qwen2_5_VLProcessor, BaseProcessor):
    """Vision-language processor that wraps an AutoProcessor.

    The provided processor must support both text and vision modalities.
    This implementation expects chat-style messages compatible with
    processor.apply_chat_template.
    """
    # 继承自 Qwen2_5_VLProcessor 已含 attributes/tokenizer_class 等定义
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        # 初始化父类 Qwen2_5_VLProcessor（包含 ProcessorMixin 初始化与类型校验）
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )
        self.chat_template = (
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
            "{% else %}{% for content in message['content'] %}"
            "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
            "{% set image_count.value = image_count.value + 1 %}"
            "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{% elif content['type'] == 'video' or 'video' in content %}"
            "{% set video_count.value = video_count.value + 1 %}"
            "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
            "{% endfor %}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        try:
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                self.tokenizer.chat_template = self.chat_template
        except Exception:
            pass

    def process_messages(self, messages: Union[str, Messages], *, padding: bool = True, add_generation_prompt: bool = True, add_system: bool = False):
        # 与 BaseProcessor 对齐：返回 (text, padding)
        text, padding = super().process_messages(
            messages,
            padding=padding,
            add_generation_prompt=add_generation_prompt,
            add_system=add_system,
        )
        if add_system:
            text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' + text
        return text, padding

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
        """将单条输入规范化为 text/images/videos 的组合并调用父处理器。

        规则：
        - 支持 input_dict 中的 'messages' 或 'text'；
        - 支持 'images' 或 'videos'，也可缺省；
        - 若存在图像/视频但文本缺失，则自动构造包含相应占位符的文本；
        - 若文本存在但未包含占位符且有图像/视频，自动在末尾追加占位符。
        """
        messages = input_dict.get('messages', None)
        raw_text = input_dict.get('text', None)

        # 解析视觉信息
        images = input_dict.get('images', None)
        videos = input_dict.get('videos', None)
        if images is None and videos is None and messages is not None:
            images, videos = process_vision_info(messages)

        # 规范化为列表
        if images is not None and not isinstance(images, Sequence):
            images = [images]
        if videos is not None and not isinstance(videos, Sequence):
            videos = [videos]

        # 生成文本（优先 messages）。若 raw_text 不是字符串（如会话列表），也走 process_messages。
        if raw_text is not None:
            if isinstance(raw_text, str):
                text = raw_text
            else:
                text, padding = self.process_messages(
                    raw_text,
                    padding=padding,
                    add_generation_prompt=add_generation_prompt,
                    add_system=add_system,
                )
        else:
            text, padding = self.process_messages(
                messages if messages is not None else "",
                padding=padding,
                add_generation_prompt=add_generation_prompt,
                add_system=add_system,
            )

        # 自动补齐视觉占位符
        def repeat_token(token: str, n: int) -> str:
            return "".join([f"<|vision_start|>{token}<|vision_end|>" for _ in range(max(0, n))])

        num_images = len(images) if images is not None else 0
        num_videos = len(videos) if videos is not None else 0
        needs_image_tokens = num_images > 0 and (text is None or (self.image_token not in str(text)))
        needs_video_tokens = num_videos > 0 and (text is None or (self.video_token not in str(text)))

        if needs_image_tokens or needs_video_tokens:
            suffix = ""
            if needs_image_tokens:
                suffix += repeat_token(self.image_token, num_images)
            if needs_video_tokens:
                suffix += repeat_token(self.video_token, num_videos)
            text = (text or "") + ("\n" if text else "") + suffix if suffix else (text or "")

        model_inputs = self.__call__(
            images=images,
            text=[text] if not isinstance(text, list) else text,
            videos=videos,
            return_tensors=return_tensors,
            padding=padding,
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
        # 聚合批量文本与视觉输入，确保占位符与图像/视频顺序一致
        text_list: List[str] = []
        flat_images: List[Any] = []
        flat_videos: List[Any] = []

        for single_input in batch_inputs:
            # 单条规范化
            messages = single_input.get('messages', None)
            raw_text = single_input.get('text', None)
            images = single_input.get('images', None)
            videos = single_input.get('videos', None)
            if images is None and videos is None and messages is not None:
                images, videos = process_vision_info(messages)
            if images is not None and not isinstance(images, Sequence):
                images = [images]
            if videos is not None and not isinstance(videos, Sequence):
                videos = [videos]

            if raw_text is not None and isinstance(raw_text, str):
                text = raw_text
            else:
                source_msgs = raw_text if raw_text is not None else (messages if messages is not None else "")
                text, _ = self.process_messages(
                    source_msgs,
                    padding=padding,
                    add_generation_prompt=add_generation_prompt,
                    add_system=add_system,
                )

            num_images = len(images) if images is not None else 0
            num_videos = len(videos) if videos is not None else 0
            needs_image_tokens = num_images > 0 and (text is None or (self.image_token not in str(text)))
            needs_video_tokens = num_videos > 0 and (text is None or (self.video_token not in str(text)))

            suffix = ""
            if needs_image_tokens:
                suffix += "".join([f"<|vision_start|>{self.image_token}<|vision_end|>" for _ in range(num_images)])
            if needs_video_tokens:
                suffix += "".join([f"<|vision_start|>{self.video_token}<|vision_end|>" for _ in range(num_videos)])
            enriched_text = (text or "") + ("\n" if text and suffix else "") + suffix
            text_list.append(enriched_text)

            if images:
                flat_images.extend(images)
            if videos:
                flat_videos.extend(videos)

        final_images = flat_images if len(flat_images) > 0 else None
        final_videos = flat_videos if len(flat_videos) > 0 else None

        model_inputs = self.__call__(
            text=text_list,
            images=final_images,
            videos=final_videos,
            return_tensors=return_tensors,
            padding=padding,
        )

        if device is not None:
            model_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in model_inputs.items()}

        return model_inputs
