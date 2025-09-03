"""Script to correct trajectories that have collisions."""
import os
import sys
import json
import argparse
from typing import List

import numpy as np

import habitat
from habitat import Env
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    CollisionsMeasurementConfig,
    FogOfWarConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


# Add multi_model_eval directory explicitly for habitat_extensions
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
multi_eval_path = os.path.join(project_root, "multi_model_eval")
if multi_eval_path not in sys.path:
    sys.path.insert(0, multi_eval_path)
# Keep project root as well for other relative imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from habitat_extensions.config_utils import create_rxr_config_wrapper
    from habitat_extensions import measures, rxr_dataset
    print("✓ RxR dataset support imported successfully")
    # Create RxR-aware config loader
    get_habitat_config_with_rxr = create_rxr_config_wrapper()
except ImportError as e:
    print(f"\033[91mWarning: Could not import RxR dataset support: {e}\033[0m")
    print("\033[91m  RxR evaluation may not work correctly\033[0m")
    get_habitat_config_with_rxr = None

# Import common helpers
from utility.vln_common import (
    save_rgb, 
    create_visualization_frame, 
    is_rxr_config, 
    get_episode_instruction
)


os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

# Discrete action mapping used in streamvln_eval.py
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


def load_actions_json(path: str):
    """Load actions and metadata from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    actions_field = data.get('actions', [])
    if (isinstance(actions_field, list) and len(actions_field) > 0 and
            isinstance(actions_field[0], dict)):
        actions = [int(a.get('action_idx', 0)) for a in actions_field]
    else:
        actions = [int(a) for a in actions_field]
    collision_flags = data.get('collision_flags', [])
    scene_id = data.get('scene_id')
    episode_id = int(data.get('episode_id'))
    image_dir = data.get('image_dir')
    instruction = data.get('instruction', '')
    return {
        'scene_id': scene_id,
        'episode_id': episode_id,
        'image_dir': image_dir,
        'instruction': instruction,
        'actions': actions,
        'collision_flags': [int(x) for x in collision_flags],
    }


def config_env(config_path: str, split: str) -> Env:
    """Configure and create a Habitat environment."""
    # Prefer RxR-aware loader when available and the config is RxR
    use_rxr_loader = (
        get_habitat_config_with_rxr is not None and is_rxr_config(config_path)
    )
    if use_rxr_loader:
        print("Using RxR-aware configuration loader")
        config = get_habitat_config_with_rxr(config_path)
    else:
        config = get_habitat_config(config_path)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        # enable top-down map for merged visualization frames
        config.habitat.task.measurements.update(
            {
                'top_down_map': TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True, visibility_dist=5.0, fov=90
                    ),
                ),
                'collisions': CollisionsMeasurementConfig(),
            }
        )
    env = Env(config=config)
    return env


def find_episode(env: Env, scene_id: str, episode_id: int):
    """Find a specific episode in the environment's dataset."""
    target_scene = str(scene_id)
    target_eid = str(episode_id)
    for ep in env.episodes:
        sid = ep.scene_id.split('/')[-2]
        if sid == target_scene and str(ep.episode_id) == target_eid:
            return ep
    raise RuntimeError(
        f'Episode {target_scene}_{target_eid} not found. '
        'Verify dataset split and episode ids.'
    )


def save_rgb(observations, out_dir: str, idx: int) -> None:
    """Save an RGB observation to a file."""
    rgb = observations['rgb']
    img = Image.fromarray(rgb).convert('RGB')
    img.save(os.path.join(out_dir, f'frame_{idx:04d}.jpg'))


def get_agent_position(env: Env) -> np.ndarray:
    """Get the agent's current position as a numpy array."""
    return np.array(env.sim.get_agent_state().position, dtype=np.float32)


class CollisionCorrectionRunner:
    """A class to run collision correction on a single trajectory."""

    def __init__(
        self,
        env: Env,
        episode_dir: str,
        actions_file: str = 'actions.json',
        output_name: str = 'actions_corrected.json',
        goal_radius: float = 1.0,
        save_video: bool = False
    ) -> None:
        """Initialize the runner."""
        self.env = env
        self.episode_dir = episode_dir
        self.actions_file = actions_file
        self.output_name = output_name
        self.goal_radius = goal_radius
        self.save_video = save_video
        self.episode = None
        self.meta = None
        self.cutoff = None
        self.frames_dir = os.path.join(self.episode_dir, 'frames_corrected')
        os.makedirs(self.frames_dir, exist_ok=True)
        try:
            self.follower = ShortestPathFollower(
                self.env.sim,
                goal_radius=self.goal_radius,
                return_one_hot=False
            )
        except Exception as e:
            print(f'Error initializing follower: {e}')
            raise

    def run(self) -> None:
        """Run the collision correction process."""
        actions_path = os.path.join(self.episode_dir, self.actions_file)
        self.meta = load_actions_json(actions_path)
        collisions = self.meta.get('collision_flags', [])
        if len(collisions) == 0:
            print('No collisions recorded. Nothing to correct.')
            return
        self.cutoff = int(min(collisions))

        try:
            self.episode = find_episode(
                self.env, self.meta['scene_id'], self.meta['episode_id']
            )
            self.env.current_episode = self.episode
            # Backfill instruction if missing (RxR/R2R compatible)
            if not self.meta.get('instruction'):
                inferred_instruction = get_episode_instruction(self.episode)
                if isinstance(inferred_instruction, str):
                    self.meta['instruction'] = inferred_instruction
        except Exception as e:
            print(f'Error finding episode: {e}')
            self.env.close()
            raise

        vis_frames: List[np.ndarray] = []

        observations = self.env.reset()
        if self.save_video:
            # Get metrics and build first frame
            info = self.env.get_metrics()
            frame = create_visualization_frame(info, observations)
            vis_frames.append(frame)

        step_id = 0
        save_rgb(observations, self.frames_dir, step_id)
        for a in self.meta['actions']:
            if step_id >= self.cutoff or self.env.episode_over:
                break
            observations = self.env.step(int(a))
            step_id += 1
            save_rgb(observations, self.frames_dir, step_id)

            if self.save_video:
                info = self.env.get_metrics()
                frame = create_visualization_frame(info, observations)
                vis_frames.append(frame)

        # Closed-loop correction with ShortestPathFollower
        correction_actions: List[int] = []
        max_correction_steps = 1000
        corr_iter = 0
        while not self.env.episode_over and corr_iter < max_correction_steps:
            raw = self.follower.get_next_action(self.episode.goals[0].position)
            if raw is None:
                break
            # Map follower action to evaluator indices
            if isinstance(raw, int):
                act = int(raw)
            else:
                mapping = {
                    'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2,
                    'TURN_RIGHT': 3
                }
                act = mapping.get(str(raw), ACTION_STOP)

            correction_actions.append(act)
            observations = self.env.step(int(act))
            step_id += 1
            save_rgb(observations, self.frames_dir, step_id)
            if self.save_video:
                info = self.env.get_metrics()
                frame = create_visualization_frame(info, observations)
                vis_frames.append(frame)

            # stop if near goal
            metrics = self.env.get_metrics()
            dist_to_goal = metrics.get('distance_to_goal', None)
            if (isinstance(dist_to_goal, (float, int)) and
                    float(dist_to_goal) < self.goal_radius):
                correction_actions.append(ACTION_STOP)
                break
            corr_iter += 1

        corrected_actions = self.meta['actions'][:self.cutoff] + \
            correction_actions

        out_path = os.path.join(self.episode_dir, self.output_name)
        out = {
            'scene_id': self.meta['scene_id'],
            'episode_id': self.meta['episode_id'],
            'image_dir': self.frames_dir,
            'instruction': self.meta.get('instruction', ''),
            'actions': corrected_actions,
            'collision_flags': self.meta.get('collision_flags', []),
            'cutoff': self.cutoff,
        }
        with open(out_path, 'w') as f:
            json.dump(out, f)
        print(f'Saved corrected actions to {out_path}')

        if self.save_video:
            images_to_video(
                vis_frames, self.episode_dir, 'correction_debug',
                fps=5, quality=9
            )
            print('Saved correction_debug.mp4 in episode directory.')


def main():
    """Main function to run the collision correction script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--habitat-config-path', '-c', type=str, default='config/vln_r2r.yaml'
    )
    parser.add_argument('--split', '-s', type=str, default='val_unseen')
    parser.add_argument(
        '--input-dir', '-i', type=str, required=True,
        help='Path to base directory of episode folders'
    )
    parser.add_argument('--actions-file', '-a', type=str, default='actions.json')
    parser.add_argument(
        '--output-name', '-o', type=str, default='actions_corrected.json'
    )
    parser.add_argument('--goal-radius', '-g', type=float, default=1.0)
    parser.add_argument(
        '--save-video', '-v', action='store_true', default=False
    )
    args = parser.parse_args()
    env = config_env(args.habitat_config_path, args.split)

    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory.")
        env.close()
        return

    subdirs = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])

    if not subdirs:
        print(f"No subdirectories found in '{args.input_dir}'.")

    for episode_dir in subdirs:
        actions_path = os.path.join(episode_dir, args.actions_file)
        if not os.path.isfile(actions_path):
            continue

        print(f'--> Processing {episode_dir}')
        try:
            runner = CollisionCorrectionRunner(
                env=env,
                episode_dir=episode_dir,
                actions_file=args.actions_file,
                output_name=args.output_name,
                goal_radius=args.goal_radius,
                save_video=args.save_video,
            )
            runner.run()
        except Exception as e:
            print(f'ERROR processing {episode_dir}: {e}')

    env.close()


if __name__ == '__main__':
    main()
