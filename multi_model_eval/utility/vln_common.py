import os
from typing import Any, Dict

from PIL import Image
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore

from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)


def save_rgb(observations: Dict[str, Any], out_dir: str, idx: int) -> None:
    """Save an RGB observation to a file."""
    rgb = observations['rgb']
    img = Image.fromarray(rgb).convert('RGB')
    img.save(os.path.join(out_dir, f'frame_{idx:04d}.jpg'))


def create_visualization_frame(info: Dict[str, Any], observations: Dict[str, Any]):
    """Create a visualization frame with overlays for logging/preview."""
    obs_for_vis = {k: v for k, v in observations.items() if hasattr(v, 'shape')}
    frame = observations_to_image(obs_for_vis, info)
    info.pop('top_down_map', None)
    frame = overlay_frame(frame, info)
    return frame


def is_rxr_config(config_path: str) -> bool:
    """Check if the configuration is for RxR dataset, robust to missing OmegaConf."""
    try:
        if OmegaConf is not None:
            config = OmegaConf.load(config_path)  # type: ignore
            if hasattr(config, 'habitat') and hasattr(config.habitat, 'dataset'):
                if hasattr(config.habitat.dataset, 'type'):
                    if str(config.habitat.dataset.type) == "RxRVLN-v1":
                        return True
            if hasattr(config, 'rxr'):
                return True
        # Fallback: filename/content heuristic
        return 'rxr' in str(config_path).lower()
    except Exception:
        return 'rxr' in str(config_path).lower()


def get_episode_instruction(episode) -> str:
    """Extract instruction text from an episode, compatible with RxR and R2R."""
    # RxR rich instruction
    if hasattr(episode, 'instruction'):
        instr = episode.instruction
        if hasattr(instr, 'instruction_text'):
            return instr.instruction_text
        if isinstance(instr, str):
            return instr
    # ObjectNav style
    if hasattr(episode, 'object_category'):
        return episode.object_category
    return "" 