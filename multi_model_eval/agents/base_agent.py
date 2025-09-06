"""Base agent class: uses unified Processor to complete all preprocessing."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import PreTrainedModel

from .processors import BaseProcessor


@dataclass
class AgentConfig:
    """Configuration for creating a navigation agent."""
    model_path: str
    agent_type: str
    model_max_length: int = 1024
    device: Optional[Union[str, int, torch.device]] = None
    project_root: Optional[str] = None
    agent_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.model_path:
            raise ValueError("model_path is required")
        
        # Initialize agent_params if None
        if self.agent_params is None:
            self.agent_params = {}
    
    def get_agent_param(self, key: str, default: Any = None) -> Any:
        """Get an agent-specific parameter."""
        return self.agent_params.get(key, default)
    
    def set_agent_param(self, key: str, value: Any) -> None:
        """Set an agent-specific parameter."""
        if self.agent_params is None:
            self.agent_params = {}
        self.agent_params[key] = value


class BaseAgent(ABC):
    """Base Navigation agent class for all VLM model.

    - Only responsible for calling the model and organizing I/O;
    - All data preprocessing is handled by Processor: processor.prepare / processor.decode.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        raise NotImplementedError

    def __init__(self, config: AgentConfig, *args, **kwargs):
        self.device = config.device
        self.model, self.processor = self.load_model_and_processor(config=config, *args, **kwargs)

    @abstractmethod
    def load_model_and_processor(self, config: AgentConfig, **kwargs) -> Tuple[PreTrainedModel, BaseProcessor]:
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

    def to(self, device: Union[str, int, torch.device]):
        """Move the model to the specified device."""
        self.model.to(device)
        self.device = device

    def eval(self):
        self.model.eval()
        self.model.requires_grad_(False)