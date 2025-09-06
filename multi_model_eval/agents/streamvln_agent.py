import sys
import os
from typing import List, Dict, Any, Sequence, Optional, Union
import re
import copy
import random
from collections import OrderedDict
import itertools
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import BitsAndBytesConfig
from transformers.image_utils import to_numpy_array
from .processors import BaseProcessor
from .base_agent import BaseAgent, AgentConfig
from utility import detect_best_attention_implementation, get_world_size
from streamvln.utils.utils import (dict_to_cuda,
                                    DEFAULT_IMAGE_TOKEN,
                                    IMAGE_TOKEN_INDEX,
                                    DEFAULT_MEMORY_TOKEN,
                                    MEMORY_TOKEN_INDEX,
                                    DEFAULT_VIDEO_TOKEN)
from streamvln.model.stream_video_vln import StreamVLNForCausalLM


class StreamVLNProcessor(BaseProcessor):
    """Processor that converts conversations to StreamVLN required `inputs` tensors.

    - Compatible with Qwen-style conversations;
    - When conversations contain image placeholders, insert 256 IMAGE_TOKEN_INDEX;
    - Use key 'inputs' in output dictionary (required by StreamVLN's generate interface).
    - Handle all input preprocessing for RGB, depth, pose, intrinsic parameters, etc.
    """
    def __init__(self, tokenizer, image_processor):
        super().__init__(tokenizer=tokenizer, image_processor=image_processor)

        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]

    def _process_conversation(
        self,
        messages,
        *,
        has_image: bool = True,
        max_len: int = 2048,
        system_message: str = 'You are a helpful assistant.',
        add_system: bool = False,
    ) -> List[int]:
        """Convert a Qwen-style conversation to a list of token ids."""
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}
        # import ipdb; ipdb.set_trace()
        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(self.tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(messages):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            

            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

    def prepare_from_inputs(
        self,
        inputs: Dict[str, Any],
        *,
        add_generation_prompt: bool = True,
        add_system: bool = False,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: str = 'pt',
    ) -> Dict[str, Any]:
        output_dict = {}
        images = inputs.get('rgb', None)
        if images is None:
            images = []
        if not isinstance(images, Sequence):
            images = [images]

        if len(images) > 0:
            images = [Image.fromarray(image).convert('RGB') if isinstance(image, np.ndarray) else image for image in images]
            if any(not isinstance(image, Image.Image) for image in images):
                raise ValueError("Invalid image type")
            pixel_values = self.image_processor.preprocess(images=images, return_tensors='pt')['pixel_values'][0]
            output_dict['pixel_values'] = pixel_values

        messages = inputs.get('messages', None)
        if messages is not None:
            input_ids, conversations = self._process_conversation(messages, has_image=True, add_system=add_system)
            output_dict['input_ids'] = input_ids

        resize_shape = None
        # depth
        depth = inputs.get('depth', None)
        if isinstance(depth, np.ndarray):
            depth_image = Image.fromarray(depth.astype(np.uint16), mode='I;16')
            depth_image, resize_shape = self.preprocess_depth_image(depth_image, do_depth_scale=True)
            depth_image = torch.from_numpy(depth_image).float()
            output_dict['depth'] = depth_image

        # pose
        pose = inputs.get('pose', None)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
            output_dict['pose'] = pose

        # intrinsic
        intrinsic = inputs.get('intrinsic', None)
        if isinstance(intrinsic, np.ndarray):
            if len(images) > 0 and resize_shape is not None:
                intrinsic = self.preprocess_intrinsic(intrinsic, images[0].size, resize_shape)
                intrinsic = torch.from_numpy(intrinsic).float()
                output_dict['intrinsic'] = intrinsic

        return output_dict

    def preprocess_depth_image(
        self,
        depth_image,
        *,
        do_depth_scale: bool = True,
        depth_scale: int = 1000,
    ):
        """Preprocess depth image by cropping to visual processor dimensions."""
        if self.image_processor is None:
            raise RuntimeError('Image processor not set')
        target_h = self.image_processor.crop_size['height']  # 384
        target_w = self.image_processor.crop_size['width']  # 384
        resized = depth_image.resize((target_w, target_h))
        arr = to_numpy_array(resized)
        if do_depth_scale:
            arr = arr / depth_scale
        return arr, (target_w, target_h)

    def preprocess_intrinsic(
        self,
        intrinsic: np.ndarray,
        ori_size,
        target_size,
    ) -> np.ndarray:
        """Adjust intrinsic parameters for size and cropping (consistent with evaluation)."""
        intr = intrinsic.copy()
        if len(intr.shape) == 2:
            intr = intr[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        intr[:, 0] /= ori_size[0] / target_size[0]  # width
        intr[:, 1] /= ori_size[1] / target_size[1]  # height
        intr[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
        if intr.shape[0] == 1:
            intr = intr.squeeze(0)
        return intr


class StreamVLNAgent(BaseAgent):
    """StreamVLN agent implementation based on the BaseAgent."""
    @property
    def name(self) -> str:
        return "streamvln"

    def __init__(self, config: AgentConfig, *args, **kwargs):
        """Initialize and load model and processor using model path."""
        self.config = config
        
        # Extract parameters from config
        params = config.agent_params or {}
        self.num_frames = params.get('num_frames', 8)
        self.num_future_steps = params.get('num_future_steps', 4)
        self.num_history = params.get('num_history', 8)

        super().__init__(config, *args, **kwargs)
        
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })

        self.num_frames = kwargs.get('num_frames', 8)
        self.num_future_steps = kwargs.get('num_future_steps', 4)
        self.num_history = kwargs.get('num_history', 8)
        
        # Canonical action names for annotations-style indexing
        self.idx2action = {0: 'STOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT'}
        self.action_to_annotation_idx = {'STOP': 0, 'FORWARD': 1, 'LEFT': 2, 'RIGHT': 3}
        self.rgb_list = []
        self.depth_list = []
        self.depth_images_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.past_key_values = None
        self.output_ids = None

    def load_model_and_processor(self, config: AgentConfig, *args, **kwargs):
        """Load StreamVLN model and custom Processor."""
        params = config.agent_params or {}
        num_history = params.get('num_history', 8)
        vision_tower_path = params.get('vision_tower_path', None)
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_path,
            model_max_length=config.model_max_length,
            padding_side='right',
        )
        auto_config = transformers.AutoConfig.from_pretrained(config.model_path)
        attn_implementation = detect_best_attention_implementation()
        
        # Prepare overwrite config for vision tower if specified
        if vision_tower_path:
            auto_config.mm_vision_tower = vision_tower_path
            print(f"✓ Overriding vision_tower path to: {vision_tower_path}")
        
        # Prepare model loading arguments
        model_kwargs = {
            'pretrained_model_name_or_path': config.model_path,  # Transformers API requires this key
            'attn_implementation': attn_implementation,
            'torch_dtype': torch.bfloat16,
            'config': auto_config,
            'low_cpu_mem_usage': True,
            'device_map': self.device if self.device is not None else 'auto',
        }
        
        # Add quantization if enabled
        quantization_bits = params.get('quantization_bits', None)
        if quantization_bits == 4:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            model_kwargs['quantization_config'] = qconf
            print(f"Using 4-bit quantization")
        elif quantization_bits == 8:
            qconf = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs['quantization_config'] = qconf
            print(f"Using 8-bit quantization")
        else:
            print(f"\033[91mWarning: Unsupported quantization bits: {quantization_bits}, falling back to no quantization\033[0m")

        model = StreamVLNForCausalLM.from_pretrained(**model_kwargs)

        if hasattr(model.model, 'num_history'):
            model.model.num_history = num_history
        world_size = get_world_size()
        model.reset(world_size)
        try:
            image_processor = model.get_vision_tower().image_processor
        except Exception as e:
            print(f'Error getting image processor: {e}')
            raise e

        processor = StreamVLNProcessor(tokenizer, image_processor)
        return model, processor

    def act(
        self,
        env_id: int,
        step_id: int,
        input_dict: Dict[str, Any]
    ) -> int:
        """Execute one generation step and parse actions, maintaining memory cache."""
        if step_id != 0 and step_id % self.num_frames == 0:
            self.reset_for_env(env_id)
        instruction = input_dict.pop('instruction')
        processor_outputs = self.processor.prepare_from_inputs(input_dict)
        self.time_ids.append(step_id)
        self.rgb_list.append(processor_outputs['pixel_values'])
        self.depth_list.append(processor_outputs['depth'])
        self.pose_list.append(processor_outputs['pose'])
        self.intrinsic_list.append(processor_outputs['intrinsic'])

        if len(self.action_seq) != 0:
            return self.action_seq.pop(0)

        if self.output_ids is None:
            messages = copy.deepcopy(self.conversation)
            messages[0]["value"] = messages[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
            if step_id != 0 :
                messages[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
            messages[0]["value"] = messages[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
            messages[0]["value"] = messages[0]["value"].replace('<instruction>.', instruction)
            add_system = True
        else:
            messages = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False
        messages = [messages]
        
        input_ids = self.processor.prepare_from_inputs({'messages': messages}, add_system=add_system)['input_ids']
        if self.output_ids is not None:
            input_ids = torch.cat([self.output_ids, input_ids.to(self.output_ids.device)], dim=1)

        images = self.rgb_list[-1:]
        depths = self.depth_list[-1:]
        poses = self.pose_list[-1:]
        intrinsics = self.intrinsic_list[-1:]
        if step_id != 0 and step_id % self.num_frames == 0:
            if self.num_history is None:
                # Use num_future_steps as step size, ensure step size > 0
                step_size = max(1, self.num_future_steps)
                # Calculate history indices using current time step
                history_ids = slice(0, step_id, step_size)
            else:
                # Calculate history indices using current time step, ensure step size > 0
                step_size = max(1, step_id // self.num_history)
                history_ids = slice(0, step_id, step_size)
            
            # Safely get history frames
            try:
                history_images = self.rgb_list[history_ids]
                history_depths = self.depth_list[history_ids]
                history_poses = self.pose_list[history_ids]
                history_intrinsics = self.intrinsic_list[history_ids]
                
                images = history_images + images
                depths = history_depths + depths
                poses = history_poses + poses
                intrinsics = history_intrinsics + intrinsics
                
                print(f"Successfully added {len(history_images)} history frames")
            except Exception as e:
                print(f"\033[91mWarning: Failed to get history frames: {e}\033[0m")
                print(f"\033[91mrgb_list length: {len(self.rgb_list)}\033[0m")
                # If getting history frames fails, only use current frame
                pass

        model_inputs = {
            'images': torch.stack(images).unsqueeze(0),
            'depths': torch.stack(depths).unsqueeze(0),
            'poses': torch.stack(poses).unsqueeze(0),
            'intrinsics': torch.stack(intrinsics).unsqueeze(0),
            'inputs': input_ids,
            'env_id': env_id,
            'time_ids': [self.time_ids],
            'task_type': [0]
        }
        if self.device is not None:
            model_inputs = dict_to_cuda(model_inputs, self.device)
        
        for key, value in model_inputs.items():
            if key in ['images', 'depths', 'poses', 'intrinsics']:
                model_inputs[key] = model_inputs[key].to(torch.bfloat16)

        outputs = self.model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=10000,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=self.past_key_values,
        )

        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values

        decoded = self.processor.decode(
            self.output_ids, skip_special_tokens=False
        )[0].strip()

        print("--------------------------------")
        print(f"llm output: {decoded}")
        print("--------------------------------")

        self.action_seq = self.parse_actions(decoded)
        if len(self.action_seq) == 0:
            self.action_seq = [0]

        return self.action_seq.pop(0)

    def reset_for_env(self, env_id: int):
        """Reset cache for specified environment and synchronize model."""
        if hasattr(self.model, 'reset_for_env'):
            self.model.reset_for_env(env_id)
        self.output_ids = None
        self.past_key_values = None
        self.time_ids = []

    def reset(self, env_id: int):
        self.reset_for_env(env_id)
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.action_seq = []

    def parse_actions(self, text: str) -> List[int]:
        """Parse action sequence from generated text, default to forward if no match."""
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(text)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
