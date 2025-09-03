"""Unified processor interfaces for language-only and vision-language models.

This module exposes:

- base_processor: abstract base class that defines the common prepare interface
- language_processor: processor for pure language models (tokenizer only)
- vision_language_processor: processor for vision-language models (AutoProcessor)
"""

from .base_processor import BaseProcessor

__all__ = [
    'BaseProcessor',
]
