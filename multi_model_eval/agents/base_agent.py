"""Base agent class: uses unified Processor to complete all preprocessing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from transformers import PreTrainedModel

from .processors import BaseProcessor


class BaseAgent(ABC):
    """Base Navigation agent class for all VLM model.

    - Only responsible for calling the model and organizing I/O;
    - All data preprocessing is handled by Processor: processor.prepare / processor.decode.
    """

    def __init__(self, model_path, *args, **kwargs):
        self.model, self.processor = self.load_model_and_processor(model_path=model_path, *args, **kwargs)

    @abstractmethod
    def load_model_and_processor(self, model_path: str, **kwargs) -> Tuple[PreTrainedModel, BaseProcessor]:
        """Load model and processor, implemented by specific agents."""
        raise NotImplementedError

    def prepare_from_inputs(
        self,
        inputs: Dict[str, Any],
        *,
        padding: bool = True,
        return_tensors: str = 'pt',
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Interface with BaseProcessor.prepare_from_inputs (unified inputs dictionary entry)."""
        model_inputs = self.processor.prepare_from_inputs(
            inputs,
            padding=padding,
            device=device,
            return_tensors=return_tensors,
        )
        return model_inputs

    @abstractmethod
    def act(
        self,
        env_id: int,
        step_id: int,
        input_dict: Dict[str, Any]
    ) -> int:
        """Unified action interface.

        - If dict is passed, use processor.prepare_from_inputs to preserve model-specific keys;
        - Otherwise call processor.prepare first for preprocessing, then generate.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_actions(self, output_text: str) -> List[int]:
        """Parse action sequence from generated text."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_id: int):
        """Reset environment-related state (if model requires)."""
        raise NotImplementedError

    def to(self, device: str | int | torch.device):
        self.model.to(device)

    def eval(self):
        self.model.eval()
        self.model.requires_grad_(False)