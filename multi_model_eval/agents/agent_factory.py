"""
Agent Factory

This module provides a clean factory interface for creating different types of agents
without exposing the complexity of model-specific construction details to users.
Users only need to work with the BaseAgent interface.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Union, List, Type
from enum import Enum
from dataclasses import dataclass
import sys
import os
import torch
from .base_agent import BaseAgent, AgentConfig


class AgentFactory:
    """Factory for creating navigation agents."""
    
    def __init__(self):
        """
        Initialize the agent factory.
        
        Args:
            project_root: Path to the project root.
        """

        self._agent_registry = {}
        self._register_agents()

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
            agent_class_path = self._agent_registry[config.agent_type]
            # Handle project root for specific agents
            if config.project_root and config.project_root not in sys.path:
                if config.agent_type in ['streamvln', 'navila']:
                    sys.path.insert(0, config.project_root)
                    if config.agent_type == 'streamvln':
                         sys.path.insert(0, os.path.join(config.project_root, 'streamvln'))

            # Dynamically import the agent class
            module_name, class_name = agent_class_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)

            # Create the agent instance
            agent = agent_class(config)
            
            print(f"✓ Successfully created {config.agent_type} agent")
            return agent
            
        except ImportError as e:
            raise ImportError(
                f"Failed to create {config.agent_type} agent due to missing dependencies: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Failed to create {config.agent_type} agent: {e}"
            ) from e

    def list_supported_agents(self) -> list[str]:
        """List all supported agent types."""
        return list(self._agent_registry.keys())

    def _register_agents(self):
        """Register all supported agent types and their creator functions."""
        self._agent_registry = {
            'qwen25vl': 'agents.qwen25vl_agent.Qwen25VLAgent',
            'streamvln': 'agents.streamvln_agent.StreamVLNAgent',
            'navila': 'agents.navila_agent.NaVILAAgent',
        }



