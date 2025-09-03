"""
Agent Factory

This module provides a clean factory interface for creating different types of agents
without exposing the complexity of model-specific construction details to users.
Users only need to work with the BaseAgent interface.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Optional, Type
from enum import Enum
from dataclasses import dataclass
import sys
import os

from . import BaseAgent


class AgentType(Enum):
    """Supported agent types."""
    STREAMVLN = "streamvln"
    NAVILA = "navila"


@dataclass
class AgentConfig:
    """Configuration for agent creation."""
    model_path: str
    agent_type: AgentType
    
    # Agent-specific parameters as a dictionary
    agent_params: Dict[str, Any] = None
    
    # Common parameters
    device: Optional[str] = None
    max_length: Optional[int] = None
    project_root: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.model_path:
            raise ValueError("model_path is required")
        
        if isinstance(self.agent_type, str):
            try:
                self.agent_type = AgentType(self.agent_type)
            except ValueError:
                raise ValueError(f"Unsupported agent type: {self.agent_type}")
        
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


class AgentFactory:
    """
    Factory class for creating agents with complete isolation and clean interface.
    
    This factory handles all the complexity of:
    - Package isolation between different agent types
    - Model-specific parameter handling
    - Import path management
    - Error handling and validation
    
    Users only need to provide an AgentConfig and get back a BaseAgent instance.
    """
    
    def __init__(self):
        """
        Initialize the agent factory.
        
        Args:
            project_root: Path to the project root.
        """

        self._agent_registry = {
            AgentType.STREAMVLN: self._create_streamvln_agent,
            AgentType.NAVILA: self._create_navila_agent,
        }
    
    def create_agent(self, config: AgentConfig) -> BaseAgent:
        """
        Create an agent instance based on the provided configuration.
        
        Args:
            config: Agent configuration specifying type and parameters
            
        Returns:
            BaseAgent: An initialized agent instance
            
        Raises:
            ValueError: If agent type is not supported
            ImportError: If agent dependencies are not available
            Exception: If agent creation fails
        """
        if config.agent_type not in self._agent_registry:
            raise ValueError(f"Unsupported agent type: {config.agent_type}")
        
        try:
            creator_func = self._agent_registry[config.agent_type]
            agent = creator_func(config)
            
            print(f"✓ Successfully created {config.agent_type.value} agent")
            return agent
            
        except ImportError as e:
            raise ImportError(
                f"Failed to create {config.agent_type.value} agent due to missing dependencies: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Failed to create {config.agent_type.value} agent: {e}"
            ) from e
    
    def _create_streamvln_agent(self, config: AgentConfig) -> BaseAgent:
        """Create StreamVLN agent with temporary path management."""
        # Add StreamVLN specific path if it exists
        streamvln_path = config.project_root

        path_added = False
        
        if os.path.exists(streamvln_path):
            if streamvln_path not in sys.path:
                sys.path.insert(0, streamvln_path)
                sys.path.insert(0, os.path.join(streamvln_path, 'streamvln'))
            path_added = True
        else:
            raise ValueError(f"StreamVLN path does not exist: {streamvln_path}")

        try:
            # Import and create StreamVLN agent
            from .streamvln_agent import StreamVLNAgent
            
            # Build kwargs from config
            streamvln_kwargs = {'model_path': config.model_path}
            
            # Add common parameters if provided
            if config.device is not None:
                streamvln_kwargs['device'] = config.device
            if config.max_length is not None:
                streamvln_kwargs['max_length'] = config.max_length
            
            # Add StreamVLN-specific parameters
            if 'quantization_bits' in config.agent_params:
                streamvln_kwargs['quantization_bits'] = config.agent_params['quantization_bits']
            if 'vision_tower_path' in config.agent_params:
                streamvln_kwargs['vision_tower_path'] = config.agent_params['vision_tower_path']
            
            agent = StreamVLNAgent(**streamvln_kwargs)
            return agent
            
        finally:
            # Remove the paths if we added them
            if path_added:
                if streamvln_path in sys.path:
                    sys.path.remove(streamvln_path)
                streamvln_inner_path = os.path.join(streamvln_path, 'streamvln')
                if streamvln_inner_path in sys.path:
                    sys.path.remove(streamvln_inner_path)
            pass
    
    def _create_navila_agent(self, config: AgentConfig) -> BaseAgent:
        """Create NaVILA agent with temporary path management."""
        # Add NaVILA specific path if it exists
        navila_path = config.project_root
        path_added = False
        
        if os.path.exists(navila_path) and navila_path not in sys.path:
            sys.path.insert(0, navila_path)
            path_added = True
        
        try:
            # Import and create NaVILA agent
            from .navila_agent import NaVILAAgent
            
            # Build kwargs from config
            navila_kwargs = {'model_path': config.model_path}
            
            # Add common parameters if provided
            if config.device is not None:
                navila_kwargs['device'] = config.device
            if config.max_length is not None:
                navila_kwargs['max_length'] = config.max_length
            
            # Add NaVILA-specific parameters
            if 'num_video_frames' in config.agent_params:
                navila_kwargs['num_video_frames'] = config.agent_params['num_video_frames']
            
            agent = NaVILAAgent(**navila_kwargs)
            return agent
            
        finally:
            # Remove the path if we added it
            if path_added and navila_path in sys.path:
                sys.path.remove(navila_path)
    
    def list_supported_agents(self) -> list[AgentType]:
        """List all supported agent types."""
        return list(self._agent_registry.keys())


# Convenience functions for easy usage
def create_agent(agent_type: str, model_path: str, agent_params: Dict[str, Any] = None, **kwargs) -> BaseAgent:
    """
    Convenience function to create an agent with minimal configuration.
    
    Args:
        agent_type: Type of agent ('streamvln' or 'navila')
        model_path: Path to the model
        agent_params: Agent-specific parameters as a dictionary
        **kwargs: Common parameters (device, max_length)
        
    Returns:
        BaseAgent: Initialized agent instance
    """
    config = AgentConfig(
        model_path=model_path,
        agent_type=AgentType(agent_type),
        agent_params=agent_params or {},
        **kwargs
    )
    
    factory = AgentFactory()
    return factory.create_agent(config)
