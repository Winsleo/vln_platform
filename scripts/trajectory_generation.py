import sys
import os
import re
import json
import argparse
from PIL import Image
import torch
import tqdm

from omegaconf import OmegaConf
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write
# Add multi_model_eval directory explicitly for habitat_extensions
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
multi_eval_path = os.path.join(project_root, "multi_model_eval")
if multi_eval_path not in sys.path:
    sys.path.insert(0, multi_eval_path)
# Import RxR dataset support from multi_model_eval habitat_extensions
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

from utility import init_distributed_mode
# Import common helpers
from utility.vln_common import (
    save_rgb, 
    is_rxr_config, 
    get_episode_instruction
)


class VLNGenerator:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cuda")
        self.config_path = args.config_path
        self.output_path = args.output_path
        self.data_path = args.data_path
        self.rank = args.rank
        self.world_size = args.world_size
        # Use RxR-aware config loader if available and dealing with RxR dataset
        if (get_habitat_config_with_rxr is not None and 
            ('rxr' in self.config_path.lower() or is_rxr_config(self.config_path))):
            print("Using RxR-aware configuration loader")
            self.config = get_habitat_config_with_rxr(self.config_path)
            self.is_rxr_dataset = True
        else:
            self.config = get_habitat_config(self.config_path)
            self.is_rxr_dataset = False
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.local_rank
                self.config.habitat.dataset.update(
                    {
                        "data_path": self.data_path,
                    }
                )
        self.data_path = self.config.habitat.dataset.data_path
        pattern = re.compile(r"datasets/([^/]+)")
        match = pattern.search(self.data_path)
        self.dataset_type = match.group(1)
        self.output_path = os.path.join(self.output_path, self.dataset_type)

    def config_env(self) -> habitat.Env:
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)

    def generate(self) -> None:
        os.makedirs(os.path.join(self.output_path), exist_ok=True)
        env = self.config_env()

        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        annotations = []
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split("/")[-2]
            print(f"scene = {scene}, scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[self.rank::self.world_size])), desc=f"scene {scene_id}")
            for episode in episodes[self.rank::self.world_size]:
                env.current_episode = episode
                agent = ShortestPathFollower(
                    sim=env.sim, goal_radius=0.5, return_one_hot=False)

                instructions = get_episode_instruction(episode)
                trajectory_id = episode.trajectory_id
                episode_id = int(episode.episode_id)
                ref_path = episode.reference_path

                observation = env.reset()

                # episode initialization
                rgb_list = []
                depth_list = []
                actions = [-1]
                next_waypoint_id = 1

                rgb_dir = os.path.join(
                    self.output_path, "images", f"{scene_id}_{self.dataset_type}_{episode_id:06d}", "rgb")
                os.makedirs(rgb_dir, exist_ok=True)
                while not env.episode_over:
                    rgb = observation["rgb"]
                    rgb_list.append(rgb)
                    Image.fromarray(rgb).convert("RGB").save(
                        os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg"))

                    next_action = agent.get_next_action(
                        ref_path[next_waypoint_id])

                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(
                                sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(
                            ref_path[next_waypoint_id])

                    if force_episode_over:
                        break

                    observation = env.step(next_action)
                    actions.append(next_action)

                process_bar.update(1)
                if len(actions) > 498:
                    continue  # Skip episodes with too many actions

                assert len(actions) == len(
                    rgb_list), f"Actions length {len(actions)} does not match RGB frames length {len(rgb_list)}"
                annotations.append({
                    "id": episode_id,
                    "video": os.path.join("images", f"{scene_id}_{self.dataset_type}_{episode_id:06d}"),
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                })

                with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                    result = {
                        "id": episode_id,
                        "video": os.path.join("images", f"{scene_id}_{self.dataset_type}_{episode_id:06d}"),
                        "instructions": instructions if isinstance(instructions, list) else [instructions],
                        "actions": actions,
                        "trajectory_id": trajectory_id,
                        "scene_id": scene_id,
                    }
                    f.write(json.dumps(result) + "\n")

        with open(os.path.join(self.output_path, f"annotations_{self.rank}.json"), "w") as f:
            json.dump(annotations, f, indent=4)

    def merge_annotations(self):
        if self.rank == 0:
            print("Merging annotation files...")
            all_annotations = []
            for i in range(self.world_size):
                file_path = os.path.join(self.output_path, f"annotations_{i}.json")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            all_annotations.extend(json.load(f))
                        os.remove(file_path)  # Delete the file after reading
                    except json.JSONDecodeError:
                        print(f"Warning: {file_path} is empty or not valid JSON. Skipping.")
                else:
                    print(f"Warning: {file_path} not found.")

            # Sort by id
            all_annotations.sort(key=lambda x: x['id'])

            # Write merged file
            with open(os.path.join(self.output_path, "annotations.json"), "w") as f:
                json.dump(all_annotations, f, indent=4)
            
            print("Annotation files merged successfully.")


def produce():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--config_path", type=str, default='config/vln_r2r.yaml',
                        help='Path to habitat config file. Use configs/vln_rxr.yaml for RxR dataset')
    parser.add_argument("--output_path", type=str, default='data/trajectory_data')
    parser.add_argument("--data_path", type=str, default=None)

    args = parser.parse_args()
    init_distributed_mode(args)
    generator = VLNGenerator(args)
    generator.generate()
    
    if args.world_size > 1:
        torch.distributed.barrier()
        generator.merge_annotations()


if __name__ == "__main__":
    produce()