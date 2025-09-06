import sys
import os

from PIL import Image
import tqdm
import torch
import json
import argparse
import quaternion
import numpy as np

from omegaconf import OmegaConf
from depth_camera_filtering import filter_depth

import habitat
from habitat import logger, Env
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image, overlay_frame
import torch.distributed as dist
# add multi_model_eval directory to Python path
projet_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projet_path)
from utility import get_rank, get_world_size, init_distributed_mode
from agents.base_agent import BaseAgent
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

# Import common helpers
from utility.vln_common import (
    save_rgb, 
    create_visualization_frame, 
    is_rxr_config, 
    get_episode_instruction
)


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        agent: BaseAgent = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.epoch = epoch
        self.config_path = config_path
        
        # Use RxR-aware config loader if available and dealing with RxR dataset
        if (get_habitat_config_with_rxr is not None and 
            ('rxr' in config_path.lower() or is_rxr_config(config_path))):
            print("Using RxR-aware configuration loader")
            self.config = get_habitat_config_with_rxr(config_path)
            self.dataset_type = "RxR"
            self.is_rxr_dataset = True
        else:
            self.config = get_habitat_config(config_path)
            self.dataset_type = "R2R"
            self.is_rxr_dataset = False
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        self.idx2action = {0: 'STOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT'}
        self.action_to_annotation_idx = {'STOP': 0, 'FORWARD': 1, 'LEFT': 2, 'RIGHT': 3}

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = get_rank()
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.num_frames = args.num_frames
        self.agent = agent
        
        self.output_path = os.path.join(output_path, self.dataset_type, split, self.agent.name)
        if self.is_rxr_dataset:
            print("✓ Detected RxR dataset - enabling multilingual support")
        else:
            print(f"✓ Detected dataset type: {self.dataset_type}")
    
    def _get_episode_metadata(self, episode) -> dict:
        """Get additional metadata from episode for RxR datasets."""
        metadata = {}
        
        if self.is_rxr_dataset and hasattr(episode, 'instruction'):
            instruction = episode.instruction
            if hasattr(instruction, 'language'):
                metadata['language'] = instruction.language
            if hasattr(instruction, 'annotator_id'):
                metadata['annotator_id'] = instruction.annotator_id
            if hasattr(instruction, 'instruction_id'):
                metadata['instruction_id'] = instruction.instruction_id
        
        return metadata

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, env_id) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        sucs, spls, oss, ones = [], [], [], []
        collision_counts, collision_any = [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            episode_id = 0
            process_bar = tqdm.tqdm(range(len(episodes[env_id::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[env_id::self.env_num]:
                episode_instruction = get_episode_instruction(episode)
                episode_metadata = self._get_episode_metadata(episode)
                
                # Log episode start with language info for RxR
                if self.is_rxr_dataset and 'language' in episode_metadata:
                    print(f"episode start [{episode_metadata['language']}]: {episode_instruction}")
                else:
                    print("episode start:", episode_instruction)
                
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                episode_dir = os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}')
                os.makedirs(episode_dir, exist_ok=True)
                frame_dir = os.path.join(episode_dir, 'frames')
                os.makedirs(frame_dir, exist_ok=True)
                actions_file = os.path.join(episode_dir, 'actions_human.json')

                step_id = 0
                save_rgb(observations, frame_dir, step_id)
                actions_taken = []
                # Record collision step ids only
                collisions_step_ids = []
                last_total_collision_count = 0
                vis_frames = []
                if self.save_video:
                    info = env.get_metrics()
                    frame = create_visualization_frame(info, observations)
                    vis_frames.append(frame)

                initial_height = env.sim.get_agent_state().position[1]

                self.agent.reset(env_id)
                while not env.episode_over:
                    self.agent.eval()
                    # ======================================
                    # environment observations preprocessing
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    pose_camera = tf_camera_to_episodic @ self.get_axis_align_matrix()
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion.as_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    inputs = {
                        "instruction": episode_instruction,
                        "rgb": image,
                        "depth": depth,
                        "pose": pose_camera,
                        "intrinsic": intrinsic_matrix,
                    }
                    
                    # Add RxR-specific metadata to inputs if available
                    if self.is_rxr_dataset and episode_metadata:
                        inputs.update(episode_metadata)
                    # =============================================================
                    with torch.inference_mode():
                        action = self.agent.act(env_id, step_id, inputs)
                    #==============================================================
                    # Save pre-action observation and log the action aligned with annotations indexing (0-based)
                    curr_step_id = step_id
                    action_name = self.idx2action.get(int(action), str(int(action)))
                    action_idx_for_annotations = self.action_to_annotation_idx.get(action_name, int(action))
                    actions_taken.append({
                        'step': int(curr_step_id),
                        'action': action_name,
                        'action_idx': int(action_idx_for_annotations)
                    })
                    # Execute action
                    observations = env.step(action)
                    step_id += 1
                    save_rgb(observations, frame_dir, step_id)
                    # After step: detect collision and record event mapped to the pre-action frame
                    step_metrics = env.get_metrics()
                    if self.save_video:
                        frame = create_visualization_frame(step_metrics, observations)
                        vis_frames.append(frame)
                    step_coll_count = 0
                    step_coll_flag = 0
                    if step_metrics.get('collisions') is not None:
                        collm = step_metrics['collisions']
                        step_coll_count = int(collm.get('count', 0)) if isinstance(collm, dict) else int(collm)
                        step_coll_flag = int(collm.get('is_collision', False)) if isinstance(collm, dict) else 0
                    prev_total = last_total_collision_count
                    inferred_delta = 1 if (step_coll_count > prev_total) else 0
                    collided_this_step = bool(step_coll_flag or inferred_delta)
                    if collided_this_step:
                        collisions_step_ids.append(int(curr_step_id))
                    last_total_collision_count = step_coll_count

                process_bar.update(1)
                metrics = env.get_metrics()
                if self.save_video:
                    # Save mp4 next to the 'frames' directory under each episode directory
                    images_to_video(
                        vis_frames, episode_dir, f'{scene_id}_{episode_id}', fps=5, quality=9
                    )
                    vis_frames.clear()
                # Save actions per episode (store image_dir once; collision_flags holds collided step ids)
                try:
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    image_dir_rel = os.path.relpath(frame_dir, start=project_root)
                    with open(actions_file, 'w') as f:
                        json.dump({
                            'scene_id': scene_id,
                            'episode_id': episode_id,
                            'image_dir': image_dir_rel,
                            'actions': actions_taken,
                            'collision_flags': collisions_step_ids
                        }, f)
                    # Also write a machine-friendly actions.json with only action_idx sequence
                    actions_idx_file = os.path.join(episode_dir, 'actions.json')
                    action_indices_only = [int(a['action_idx']) for a in actions_taken]
                    with open(actions_idx_file, 'w') as f2:
                        json.dump({
                            'scene_id': scene_id,
                            'episode_id': episode_id,
                            'image_dir': image_dir_rel,
                            'instruction': episode_instruction,
                            'actions': action_indices_only,
                            'collision_flags': collisions_step_ids
                        }, f2)
                except Exception as e:
                    logger.error(f'Failed to save actions for {scene_id}_{episode_id}: {e}')
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                coll_count = 0
                coll_flag = 0
                if metrics.get('collisions') is not None:
                    # Habitat 'collisions' measure returns a dict with keys like 'count' and 'is_collision'
                    coll = metrics['collisions']
                    coll_count = int(coll.get('count', 0)) if isinstance(coll, dict) else int(coll)
                    coll_flag = int(1 if (isinstance(coll, dict) and coll.get('count', 0) > 0) else 0)
                collision_counts.append(coll_count)
                collision_any.append(coll_flag)
                # Enhanced logging for RxR datasets
                if self.is_rxr_dataset and 'language' in episode_metadata:
                    print(f"scene_episode {scene_id}_{episode_id} [{episode_metadata['language']}] success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}, collisions: {coll_count}")
                else:
                    print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}, collisions: {coll_count}")
                
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "collisions": coll_count,
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                    "dataset_type": self.dataset_type
                }
                
                # Add RxR-specific metadata to results
                if self.is_rxr_dataset and episode_metadata:
                    result.update(episode_metadata)
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(ones).to(self.device),
            torch.tensor(collision_counts).to(self.device),
            torch.tensor(collision_any).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )    


def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml',
                        help='Path to habitat config file. Use configs/vln_rxr.yaml for RxR dataset')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--quantization_bits", type=int,
                        help="Quantization bits. 4 for 4-bit, 8 for 8-bit.")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--agent_type', type=str, default='streamvln', choices=['streamvln', 'navila', 'qwen25vl'],
                        help='Type of agent to use for evaluation')
    parser.add_argument('--vision_tower_path', type=str, default=None,
                        help='Path to vision tower model (e.g., checkpoints/google/siglip-so400m-patch14-384)')
    
    args = parser.parse_args()
    init_distributed_mode(args)

    evaluate(args)


def evaluate(args):
    local_rank = args.local_rank
    world_size = get_world_size()
    device = torch.device("cuda", local_rank)
    
    # Create agent using the factory pattern
    from agents.agent_factory import AgentFactory
    from agents.base_agent import AgentConfig
    
    try:
        # Create agent configuration
        agent_params = {}
        if hasattr(args, 'quantization_bits'):
            agent_params['quantization_bits'] = args.quantization_bits

        # Add agent-specific parameters based on type
        if args.agent_type == 'streamvln':
            if getattr(args, 'vision_tower_path', None):
                agent_params['vision_tower_path'] = args.vision_tower_path
            if hasattr(args, 'num_history'):
                agent_params['num_history'] = args.num_history
            if hasattr(args, 'num_frames'):
                agent_params['num_frames'] = args.num_frames
            if hasattr(args, 'num_future_steps'):
                agent_params['num_future_steps'] = args.num_future_steps
            project_root = os.path.join(os.path.dirname(projet_path), "StreamVLN")
        elif args.agent_type == 'navila':
            agent_params['num_video_frames'] = getattr(args, 'num_frames', 8)
            project_root = os.path.join(os.path.dirname(projet_path), "NaVILA")
        elif args.agent_type == 'qwen25vl':
            if hasattr(args, 'num_history'):
                agent_params['num_history'] = args.num_history
            project_root = os.path.dirname(os.path.abspath(__file__))

        config = AgentConfig(
            model_path=args.model_path,
            agent_type=args.agent_type,
            project_root=project_root,
            agent_params=agent_params,
            model_max_length=args.model_max_length,
            device=device,
        )
        
        # Create agent using factory
        factory = AgentFactory()  # Factory can be initialized without project_root now
        agent = factory.create_agent(config)
    except ValueError as e:
        print(f"\033[91m❌ Configuration Error: {e}\033[0m")
        sys.exit(1)
    except ImportError as e:
        print(f"\033[91m❌ Import Error: {e}\033[0m")
        print(f"\033[91m   Please check {args.agent_type} dependencies and configuration.\033[0m")
        sys.exit(1)
    except Exception as e:
        print(f"\033[91m❌ Agent Creation Error: {e}\033[0m")
        sys.exit(1)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        agent=agent,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, coll_counts, coll_any, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    coll_counts_all = [torch.zeros(ep_num_all[i], dtype=coll_counts.dtype).to(coll_counts.device) for i in range(world_size)]
    coll_any_all = [torch.zeros(ep_num_all[i], dtype=coll_any.dtype).to(coll_any.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.all_gather(coll_counts_all, coll_counts)
    dist.all_gather(coll_any_all, coll_any)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    coll_counts_all = torch.cat(coll_counts_all, dim=0)
    coll_any_all = torch.cat(coll_any_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    "collisions_avg": (sum(coll_counts_all)/len(coll_counts_all)).item(),
                    "collision_rate": (sum(coll_any_all)/len(coll_any_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))


if __name__ == "__main__":
    eval()
