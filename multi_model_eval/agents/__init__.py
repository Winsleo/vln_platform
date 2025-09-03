"""Unified agent interfaces for different types of vision-language models.

This module provides:

- base_agent: abstract base class that defines common interfaces
- streamvln_agent: dedicated agent for StreamVLN models
- navila_agent: dedicated agent for NaVILA models
- attention detection utilities for optimal performance
"""

from .base_agent import BaseAgent
from . import agent_factory
# Import processors and expose main components
from . import processors


__all__ = [
    'BaseAgent',
    # Expose processors module and main classes
    'processors',
    'agent_factory',
]

