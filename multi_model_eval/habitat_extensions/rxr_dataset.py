#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
RxR VLN Dataset implementation for habitat-lab v0.2.4 compatibility.
This module provides RxRVLN-v1 dataset as habitat_extensions.
"""

import gzip
import json
import math
import os
from typing import Dict, List, Optional, Union

import attr
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    """Extended instruction data for RxR dataset with additional metadata."""
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(default=None)
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RxRVLNEpisode(VLNEpisode):
    """Extended VLN Episode for RxR dataset compatible with habitat-lab v0.2.4.
    
    This class extends the standard VLNEpisode to include RxR-specific features
    such as multilingual instructions and extended metadata.
    """
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(default=None, validator=not_none_validator)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from a list split into n chunks."""
    chunks = split_list(lst, n)
    return chunks[k]


@registry.register_dataset(name="RxRVLN-v1")
class RxRVLNDatasetV1(Dataset):
    """RxR VLN Dataset for habitat-lab v0.2.4 via habitat_extensions.
    
    This dataset loads the Room-across-Room (RxR) VLN dataset with support for:
    - Multiple languages (en-US, en-IN, hi-IN, te-IN)
    - Multiple annotation roles (guide, follower)
    - Extended instruction metadata
    - Compatibility with habitat-lab v0.2.4 API
    
    Usage:
        from habitat.core.registry import registry
        from habitat_extensions.rxr_dataset import RxRVLNDatasetV1
        
        # Get dataset class
        dataset_cls = registry.get_dataset("RxRVLN-v1")
        dataset = dataset_cls(config)
    """

    episodes: List[RxRVLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    def __init__(self, config=None) -> None:
        """Initialize RxR VLN dataset.
        
        Args:
            config: Dataset configuration object with the following expected fields:
                - data_path: Path template with {split} and {role} placeholders
                - scenes_dir: Directory containing scene files
                - split: Dataset split (train, val_seen, val_unseen, test)
                
        Optional environment variables:
            - RXR_ROLES: Comma-separated list of roles (default: "guide")
            - RXR_LANGUAGES: Comma-separated list of languages (default: "en-US")
            - RXR_CONTENT_SCENES: Comma-separated list of scenes (default: "*")
        """
        self.episodes = []
        self.config = config

        if config is None:
            return

        # Extract roles from config or environment variables
        roles = self.extract_roles_from_config(config)
        
        # Extract languages from config or environment variables
        languages = self.extract_languages_from_config(config)
        
        print(f"Loading RxR dataset with roles: {roles}, languages: {languages}")
        
        # Load data for each role
        for role in roles:
            data_path = config.data_path.format(split=config.split, role=role)
            if os.path.exists(data_path):
                print(f"Loading data from: {data_path}")
                with gzip.open(data_path, "rt") as f:
                    self.from_json(
                        f.read(),
                        scenes_dir=config.scenes_dir,
                        num_chunks=getattr(config, 'num_chunks', 1),
                        chunk_idx=getattr(config, 'chunk_idx', 0)
                    )
            else:
                print(f"\033[91mWarning: Data file not found: {data_path}\033[0m")

        # Apply filters
        self._apply_scene_filter(config)
        self._apply_language_filter(config, languages)
        self._apply_episode_filter(config)
        
        print(f"Loaded {len(self.episodes)} episodes total")

    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        num_chunks: Optional[int] = 1,
        chunk_idx: Optional[int] = 0,
    ) -> None:
        """Load episodes from JSON string.
        
        Args:
            json_str: JSON string containing episode data
            scenes_dir: Directory containing scene files
            num_chunks: Number of chunks for distributed processing
            chunk_idx: Index of chunk to load
        """
        deserialized = json.loads(json_str)

        # Load vocabulary if present
        if "instruction_vocab" in deserialized:
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )
        elif not hasattr(self, 'instruction_vocab'):
            # Create empty vocab if not present
            self.instruction_vocab = VocabDict(word_list=[])

        # Get chunked episodes
        chunked_episodes = get_chunk(deserialized["episodes"], num_chunks, chunk_idx)

        for episode in chunked_episodes:
            # Cast integer IDs to strings for consistency
            episode["episode_id"] = str(episode["episode_id"])
            if "trajectory_id" in episode and episode["trajectory_id"] is not None:
                episode["trajectory_id"] = str(episode["trajectory_id"])

            episode = RxRVLNEpisode(**episode)

            # Handle scene path
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # Create extended instruction data
            episode.instruction = ExtendedInstructionData(**episode.instruction)
            if self.config and hasattr(self.config, 'split'):
                episode.instruction.split = self.config.split

            # Handle goals if present
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)

            self.episodes.append(episode)

    def _apply_scene_filter(self, config):
        """Apply scene filtering based on config."""
        if hasattr(config, 'content_scenes') and ALL_SCENES_MASK not in config.content_scenes:
            scenes_to_load = set(config.content_scenes)
            self.episodes = [
                e for e in self.episodes 
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

    def _apply_language_filter(self, config, languages):
        """Apply language filtering based on config."""
        if ALL_LANGUAGES_MASK not in languages:
            languages_to_load = set(languages)
            self.episodes = [
                episode for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

    def _apply_episode_filter(self, config):
        """Apply episode filtering based on config."""
        if hasattr(config, 'episodes_allowed') and ALL_EPISODES_MASK not in config.episodes_allowed:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.episodes_allowed)
            self.episodes = [
                episode for episode in self.episodes 
                if episode.episode_id not in ep_ids_to_purge
            ]

    @classmethod
    def extract_roles_from_config(cls, config) -> List[str]:
        """Extract roles from config with multiple fallback sources."""
        # Priority 1: Direct config attribute
        if hasattr(config, 'roles') and config.roles is not None:
            roles = config.roles
            if isinstance(roles, str):
                roles = [roles]
            if ALL_ROLES_MASK in roles or '*' in roles:
                return cls.annotation_roles
            assert set(roles).issubset(set(cls.annotation_roles)), \
                f"Invalid roles: {set(roles) - set(cls.annotation_roles)}"
            return roles
        
        # Priority 2: Environment variable
        env_roles = os.environ.get('RXR_ROLES')
        if env_roles:
            if env_roles == '*':
                return cls.annotation_roles
            roles = [role.strip() for role in env_roles.split(',')]
            assert set(roles).issubset(set(cls.annotation_roles)), \
                f"Invalid roles in RXR_ROLES: {set(roles) - set(cls.annotation_roles)}"
            return roles
        
        # Priority 3: Default to guide role only
        return ["guide"]
    
    @classmethod
    def extract_languages_from_config(cls, config) -> List[str]:
        """Extract languages from config with multiple fallback sources."""
        # Priority 1: Direct config attribute
        if hasattr(config, 'languages') and config.languages is not None:
            languages = config.languages
            if isinstance(languages, str):
                languages = [languages]
            if ALL_LANGUAGES_MASK in languages or '*' in languages:
                return cls.languages
            assert set(languages).issubset(set(cls.languages)), \
                f"Invalid languages: {set(languages) - set(cls.languages)}"
            return languages
        
        # Priority 2: Environment variable
        env_languages = os.environ.get('RXR_LANGUAGES')
        if env_languages:
            if env_languages == '*':
                return cls.languages
            languages = [lang.strip() for lang in env_languages.split(',')]
            assert set(languages).issubset(set(cls.languages)), \
                f"Invalid languages in RXR_LANGUAGES: {set(languages) - set(cls.languages)}"
            return languages
        
        # Priority 3: Default to English only
        return ["en-US"]

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        """Return a sorted list of scenes to load."""
        assert cls.check_config_paths_exist(config), "Config paths do not exist"
        dataset = cls(config)
        return sorted({cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes})

    @classmethod
    def check_config_paths_exist(cls, config) -> bool:
        """Check if all required data files exist for the specified roles."""
        roles = cls.extract_roles_from_config(config)
        return all(
            os.path.exists(config.data_path.format(split=config.split, role=role))
            for role in roles
        ) and os.path.exists(config.scenes_dir)

    @staticmethod
    def _scene_from_episode(episode: RxRVLNEpisode) -> str:
        """Helper method to get the scene name from an episode."""
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: RxRVLNEpisode) -> str:
        """Extract language from episode instruction."""
        return episode.instruction.language if episode.instruction.language else "en-US"

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episodes)

    def __getitem__(self, index: int) -> RxRVLNEpisode:
        """Get episode by index."""
        return self.episodes[index]
