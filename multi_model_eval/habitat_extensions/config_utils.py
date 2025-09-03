#!/usr/bin/env python3

"""
Configuration utilities for RxR dataset parameter handling.
This module provides mechanisms to pass RxR-specific parameters through Habitat's configuration system.
"""

import os
from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig, OmegaConf


class RxRConfigExtension:
    """Utility class to handle RxR-specific configuration parameters."""
    
    # Default values for RxR parameters
    DEFAULT_ROLES = ["guide"]
    DEFAULT_LANGUAGES = ["en-US"]
    DEFAULT_CONTENT_SCENES = ["*"]
    DEFAULT_EPISODES_ALLOWED = ["*"]
    
    # All supported values
    ALL_ROLES = ["guide", "follower"]
    ALL_LANGUAGES = ["en-US", "en-IN", "hi-IN", "te-IN"]
    
    @classmethod
    def extend_habitat_config(cls, config: DictConfig, rxr_params: Optional[Dict[str, Any]] = None) -> DictConfig:
        """
        Extend Habitat configuration with RxR-specific parameters.
        
        Args:
            config: Base Habitat configuration
            rxr_params: Dictionary containing RxR-specific parameters
                       Keys: roles, languages, content_scenes, episodes_allowed, num_chunks, chunk_idx
        
        Returns:
            Extended configuration with RxR parameters
        """
        if rxr_params is None:
            rxr_params = {}
        
        # Create a copy to avoid modifying the original
        extended_config = OmegaConf.create(OmegaConf.to_yaml(config))
        
        # Disable struct mode to allow adding new keys
        OmegaConf.set_struct(extended_config, False)
        
        # Ensure dataset section exists
        if not hasattr(extended_config.habitat, 'dataset'):
            extended_config.habitat.dataset = {}
        
        # Add RxR-specific parameters
        extended_config.habitat.dataset.roles = rxr_params.get('roles', cls.DEFAULT_ROLES)
        extended_config.habitat.dataset.languages = rxr_params.get('languages', cls.DEFAULT_LANGUAGES)
        extended_config.habitat.dataset.content_scenes = rxr_params.get('content_scenes', cls.DEFAULT_CONTENT_SCENES)
        extended_config.habitat.dataset.episodes_allowed = rxr_params.get('episodes_allowed', cls.DEFAULT_EPISODES_ALLOWED)
        
        # Optional distributed training parameters
        if 'num_chunks' in rxr_params:
            extended_config.habitat.dataset.num_chunks = rxr_params['num_chunks']
        if 'chunk_idx' in rxr_params:
            extended_config.habitat.dataset.chunk_idx = rxr_params['chunk_idx']
        
        # Re-enable struct mode
        OmegaConf.set_struct(extended_config, True)
        
        return extended_config
    
    @classmethod
    def load_rxr_config_from_yaml(cls, config_path: str) -> Dict[str, Any]:
        """
        Load RxR configuration parameters from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing RxR parameters
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the configuration
        config = OmegaConf.load(config_path)
        
        # Extract RxR parameters from the config
        rxr_params = {}
        
        # Check if RxR parameters are defined in the config
        if hasattr(config, 'rxr'):
            rxr_section = config.rxr
            if hasattr(rxr_section, 'roles'):
                rxr_params['roles'] = rxr_section.roles
            if hasattr(rxr_section, 'languages'):
                rxr_params['languages'] = rxr_section.languages
            if hasattr(rxr_section, 'content_scenes'):
                rxr_params['content_scenes'] = rxr_section.content_scenes
            if hasattr(rxr_section, 'episodes_allowed'):
                rxr_params['episodes_allowed'] = rxr_section.episodes_allowed
            if hasattr(rxr_section, 'num_chunks'):
                rxr_params['num_chunks'] = rxr_section.num_chunks
            if hasattr(rxr_section, 'chunk_idx'):
                rxr_params['chunk_idx'] = rxr_section.chunk_idx
        
        # Also check if parameters are directly in dataset section (for backward compatibility)
        if hasattr(config, 'habitat') and hasattr(config.habitat, 'dataset'):
            dataset = config.habitat.dataset
            if hasattr(dataset, 'roles'):
                rxr_params['roles'] = dataset.roles
            if hasattr(dataset, 'languages'):
                rxr_params['languages'] = dataset.languages
            if hasattr(dataset, 'content_scenes'):
                rxr_params['content_scenes'] = dataset.content_scenes
            if hasattr(dataset, 'episodes_allowed'):
                rxr_params['episodes_allowed'] = dataset.episodes_allowed
            if hasattr(dataset, 'num_chunks'):
                rxr_params['num_chunks'] = dataset.num_chunks
            if hasattr(dataset, 'chunk_idx'):
                rxr_params['chunk_idx'] = dataset.chunk_idx
        
        return rxr_params
    
    @classmethod
    def load_rxr_config_from_env(cls) -> Dict[str, Any]:
        """
        Load RxR configuration parameters from environment variables.
        
        Returns:
            Dictionary containing RxR parameters from environment
        """
        rxr_params = {}
        
        # Load roles
        if 'RXR_ROLES' in os.environ:
            roles_str = os.environ['RXR_ROLES']
            if roles_str == '*':
                rxr_params['roles'] = cls.ALL_ROLES
            else:
                rxr_params['roles'] = [role.strip() for role in roles_str.split(',')]
        
        # Load languages
        if 'RXR_LANGUAGES' in os.environ:
            languages_str = os.environ['RXR_LANGUAGES']
            if languages_str == '*':
                rxr_params['languages'] = cls.ALL_LANGUAGES
            else:
                rxr_params['languages'] = [lang.strip() for lang in languages_str.split(',')]
        
        # Load content scenes
        if 'RXR_CONTENT_SCENES' in os.environ:
            scenes_str = os.environ['RXR_CONTENT_SCENES']
            if scenes_str == '*':
                rxr_params['content_scenes'] = ['*']
            else:
                rxr_params['content_scenes'] = [scene.strip() for scene in scenes_str.split(',')]
        
        # Load episodes allowed
        if 'RXR_EPISODES_ALLOWED' in os.environ:
            episodes_str = os.environ['RXR_EPISODES_ALLOWED']
            if episodes_str == '*':
                rxr_params['episodes_allowed'] = ['*']
            else:
                rxr_params['episodes_allowed'] = [ep.strip() for ep in episodes_str.split(',')]
        
        # Load distributed training parameters
        if 'RXR_NUM_CHUNKS' in os.environ:
            rxr_params['num_chunks'] = int(os.environ['RXR_NUM_CHUNKS'])
        
        if 'RXR_CHUNK_IDX' in os.environ:
            rxr_params['chunk_idx'] = int(os.environ['RXR_CHUNK_IDX'])
        
        return rxr_params
    
    @classmethod
    def validate_rxr_params(cls, rxr_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate RxR parameters and provide defaults for missing values.
        
        Args:
            rxr_params: Dictionary containing RxR parameters
            
        Returns:
            Validated and normalized RxR parameters
        """
        validated_params = {}
        
        # Validate roles
        roles = rxr_params.get('roles', cls.DEFAULT_ROLES)
        if isinstance(roles, str):
            roles = [roles]
        if '*' in roles:
            validated_params['roles'] = cls.ALL_ROLES
        else:
            invalid_roles = set(roles) - set(cls.ALL_ROLES)
            if invalid_roles:
                raise ValueError(f"Invalid roles: {invalid_roles}. Valid roles: {cls.ALL_ROLES}")
            validated_params['roles'] = roles
        
        # Validate languages
        languages = rxr_params.get('languages', cls.DEFAULT_LANGUAGES)
        if isinstance(languages, str):
            languages = [languages]
        if '*' in languages:
            validated_params['languages'] = cls.ALL_LANGUAGES
        else:
            invalid_languages = set(languages) - set(cls.ALL_LANGUAGES)
            if invalid_languages:
                raise ValueError(f"Invalid languages: {invalid_languages}. Valid languages: {cls.ALL_LANGUAGES}")
            validated_params['languages'] = languages
        
        # Validate content scenes
        content_scenes = rxr_params.get('content_scenes', cls.DEFAULT_CONTENT_SCENES)
        if isinstance(content_scenes, str):
            content_scenes = [content_scenes]
        validated_params['content_scenes'] = content_scenes
        
        # Validate episodes allowed
        episodes_allowed = rxr_params.get('episodes_allowed', cls.DEFAULT_EPISODES_ALLOWED)
        if isinstance(episodes_allowed, str):
            episodes_allowed = [episodes_allowed]
        validated_params['episodes_allowed'] = episodes_allowed
        
        # Validate distributed training parameters
        if 'num_chunks' in rxr_params:
            num_chunks = rxr_params['num_chunks']
            if not isinstance(num_chunks, int) or num_chunks < 1:
                raise ValueError(f"num_chunks must be a positive integer, got: {num_chunks}")
            validated_params['num_chunks'] = num_chunks
        
        if 'chunk_idx' in rxr_params:
            chunk_idx = rxr_params['chunk_idx']
            if not isinstance(chunk_idx, int) or chunk_idx < 0:
                raise ValueError(f"chunk_idx must be a non-negative integer, got: {chunk_idx}")
            validated_params['chunk_idx'] = chunk_idx
        
        # Validate chunk_idx is less than num_chunks if both are provided
        if 'num_chunks' in validated_params and 'chunk_idx' in validated_params:
            if validated_params['chunk_idx'] >= validated_params['num_chunks']:
                raise ValueError(f"chunk_idx ({validated_params['chunk_idx']}) must be less than num_chunks ({validated_params['num_chunks']})")
        
        return validated_params
    
    @classmethod
    def get_merged_rxr_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get merged RxR configuration from multiple sources with priority:
        1. Configuration file (highest priority)
        2. Environment variables
        3. Default values (lowest priority)
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Merged and validated RxR parameters
        """
        # Start with defaults
        merged_params = {
            'roles': cls.DEFAULT_ROLES,
            'languages': cls.DEFAULT_LANGUAGES,
            'content_scenes': cls.DEFAULT_CONTENT_SCENES,
            'episodes_allowed': cls.DEFAULT_EPISODES_ALLOWED,
        }
        
        # Override with environment variables
        env_params = cls.load_rxr_config_from_env()
        merged_params.update(env_params)
        
        # Override with config file (highest priority)
        if config_path and os.path.exists(config_path):
            try:
                file_params = cls.load_rxr_config_from_yaml(config_path)
                merged_params.update(file_params)
            except Exception as e:
                print(f"\033[91mWarning: Failed to load RxR config from {config_path}: {e}\033[0m")
        
        # Validate and return
        return cls.validate_rxr_params(merged_params)


def create_rxr_config_wrapper():
    """
    Create a wrapper function for get_habitat_config that supports RxR parameters.
    
    Returns:
        Wrapper function that can be used instead of get_habitat_config
    """
    def get_habitat_config_with_rxr(config_path: str, overrides: Optional[List[str]] = None):
        """
        Load Habitat configuration with RxR parameter support.
        
        Args:
            config_path: Path to the main Habitat configuration file
            overrides: Optional list of configuration overrides
            
        Returns:
            Extended Habitat configuration with RxR parameters
        """
        from habitat_baselines.config.default import get_config as get_habitat_config
        
        # Load base Habitat configuration
        base_config = get_habitat_config(config_path, overrides)
        
        # Get RxR parameters
        rxr_params = RxRConfigExtension.get_merged_rxr_config(config_path)
        
        # Extend configuration with RxR parameters
        extended_config = RxRConfigExtension.extend_habitat_config(base_config, rxr_params)
        
        print(f"✓ Loaded RxR configuration with parameters: {rxr_params}")
        
        return extended_config
    
    return get_habitat_config_with_rxr
