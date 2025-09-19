import numpy as np
import json
import os
import random
import copy
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import SigLipImageProcessor
from model.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_MEMORY_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
from typing import Dict


def sample_indices(total_length, num_frames):
    """Sample indices to fixed number. If total_length < num_frames, return all indices."""
    if total_length < num_frames:
        return np.arange(total_length).tolist()
    return np.round(np.linspace(0, total_length - 1, num=num_frames)).astype(int).tolist()


def pad_tensors(tensors, lens=None, max_len=None, pad=0, padding_side: str = "right"):
    """B x [T, ...]

    当 pad_left=True 时，进行左侧填充，使得每个样本的有效时间步对齐到序列末尾。
    """
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        if padding_side == "left":
            output.data[i, -l:, ...] = t.data
        else:
            output.data[i, :l, ...] = t.data
    return output


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
    # stop_token_index = tokenizer.convert_tokens_to_ids("Ġstop")

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
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

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
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX
            
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def collate_fn(batch, tokenizer):
    # Support tuples with optional user_prompt and alt_instructions
    tuple_len = len(batch[0])
    if tuple_len == 5:
        input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(*batch)
        user_prompts, alt_instructions = None, None
    elif tuple_len == 7:
        input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch, user_prompts, alt_instructions = zip(*batch)
    else:
        raise ValueError(f"Unexpected batch item length: {tuple_len}")
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=IGNORE_INDEX)
    
    input_ids_batch = input_ids_batch[:, :tokenizer.model_max_length]
    labels_batch = labels_batch[:, :tokenizer.model_max_length]
    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)
    
    img_lens = np.array([i.size(0) for i in image_batch])

    if time_ids_batch[0] is not None:
        time_ids_batch = pad_sequence(time_ids_batch, batch_first=True, padding_value=-1)
    
    # 左侧填充图像，使得最新一帧固定位于末尾
    image_batch = pad_tensors(image_batch, img_lens, padding_side="left")
    
    # 图像注意力掩码：真实=1，补齐=0（与左填充对齐）
    image_attention_mask = torch.zeros((len(img_lens), image_batch.size(1)), dtype=torch.long, device=image_batch.device)
    for i, l in enumerate(img_lens):
        image_attention_mask[i, -l:] = 1

    out = {'images': image_batch, 
           'time_ids': time_ids_batch, 
           'attention_mask': attention_mask, 
           'input_ids': input_ids_batch, 
           'labels': labels_batch, 
           'image_attention_mask': image_attention_mask,
           'task_type': task_type_batch}
    if tuple_len == 7:
        out['user_prompts'] = list(user_prompts)
        out['alt_instructions'] = list(alt_instructions)
    return out


class VLNActionDataset(Dataset):
    def __init__(
        self,
        processor,
        tokenizer,
        data_args, 
        task_id
    ):
        super(VLNActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns
        self.enable_stopword_ignore = getattr(data_args, 'enable_stopword_ignore', False)

        self.video_folder = data_args.video_folder.split(',')

        self.nav_data =[]
        for vf in self.video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
        
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)
            if actions_len < self.num_future_steps:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]
                
            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)
                    if valid_idx != 0:
                        invalid_len += 1

                if actions_len - valid_idx < self.num_future_steps:
                    continue
                
                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))

        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.act_conjunctions = [
                                    'and then ', 
                                    'after that ', 
                                    'next ', 
                                    'the next action is ',
                                    'followed by ', 
                                    'leading to ', 
                                    'continuing ',
                                    'subsequently ', 
                                    'proceeding to '
                                ]
        
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

    def __len__(self):
        return len(self.data_list)
    
    @property
    def task(self):
        return self.task_id
    
    def actions2text(self, actions):
        converted_sequence = []         
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        
        text = ''.join(converted_sequence)
        return text
    
    def prepare_conversation(self, conversation, actions): 
        i = 0
        sources = []
        t = 0
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i+self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            source[1]["value"] = answer
            i += len(step_actions)
            t += 1
            sources.extend(source)
        return sources
    
    def __getitem__(self, i):
        ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions = data['actions'][1+valid_idx:] + [0]
        actions_len = len(actions)
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0
        actions = np.array(actions)[time_ids]

        start_idx, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
        sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]

        if time_ids[0] != 0:
            history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
            history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]
        else:
            history_frames = []
            
        images = []
        for image_file in history_frames + sample_frames:
            image = Image.open(image_file).convert('RGB')
            if self.transforms is not None:
                image = self.transforms(image)
            
            image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0] # [3, H, W]
            images.append(image)

        images = torch.stack(images)
        
        sources = copy.deepcopy(self.conversations)

        if start_idx != 0:
            sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'
        
        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instructions[ins_id])
        interleave_sources = self.prepare_conversation(sources, list(actions))
        
        data_dict = preprocess_qwen([interleave_sources], self.tokenizer, True)

        return data_dict["input_ids"][0], \
            data_dict["labels"][0], \
            images, \
            torch.tensor(time_ids), \
            self.task


class VLNTrajectoryDescriptionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_args,
        task_id
    ):
        super(VLNTrajectoryDescriptionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()
        # minimal English stopword set for label masking
        self.enable_stopword_ignore = data_args.enable_stopword_ignore
        self.stopwords = set([
            'a','an','the','and','or','but','if','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','can','will','just','don','should','now'
        ])
        self.punctuations = set(list(",.;:!?-—()[]{}'\""))

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        self.video_folder = data_args.video_folder.split(',')

        self.nav_data = []
        for vf in self.video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json

        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)

            if not isinstance(instructions, list):
                instructions = [instructions]

            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)

                # For full-trajectory description, require at least one effective action
                if actions_len - valid_idx < 1:
                    continue

                # One sample per instruction: full trajectory starting at 0 (post-cleaning)
                self.data_list.append((ep_id, ins_id, 0, valid_idx))

        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        prompt = (
            "You are an autonomous navigation assistant. "
            "Given a sequence of actions and corresponding observations, "
            "describe the trajectory in natural language (i.e., the likely instruction for the agent)."
        )
        answer = ""
        self.base_conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

    def __len__(self):
        return len(self.data_list)

    @property
    def task(self):
        return self.task_id

    def actions2text(self, actions):
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        text = ''.join(converted_sequence)
        return text

    def __getitem__(self, i):
        ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions_full = data['actions'][1 + valid_idx:] + [0]
        actions_len = len(actions_full)
        # 观测帧可用长度（从valid_idx开始）
        num_available_frames = max(len(video_frames) - valid_idx, 0)
        effective_len = min(actions_len, num_available_frames)

        # 使用sample_indices将观测降采样到固定帧数，但动作序列保持完整
        sampled_local_indices = sample_indices(effective_len, self.num_frames)
        time_ids = np.array(sampled_local_indices, dtype=np.int32)
        assert len(time_ids) > 0
        actions = np.array(actions_full)  # 动作序列不变，使用完整序列

        # 选取对应的观测帧（相对索引加上valid_idx得到全局帧索引）
        sample_step_ids = valid_idx + time_ids
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[idx]) for idx in sample_step_ids]

        images = []
        for image_file in sample_frames:
            image = Image.open(image_file).convert('RGB')
            if self.transforms is not None:
                image = self.transforms(image)
            image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
            images.append(image)

        images = torch.stack(images)

        # Build conversation: human provides observations and action sequence; assistant outputs instruction
        sources = copy.deepcopy(self.base_conversation)
        # Actions as input
        action_text = self.actions2text(list(actions))
        sources[0]["value"] += f" The action sequence is: {action_text}."
        # Observations tokens with unified wording
        obs_tokens = " ".join([DEFAULT_IMAGE_TOKEN for _ in range(len(sample_frames))]) if len(sample_frames) > 0 else DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" This is the observation sequence corresponding to the action sequence of the trajectory: {obs_tokens}."

        # Target is the natural language instruction (randomly sample one reference)
        if len(instructions) > 0:
            sampled_instruction = random.choice(instructions)
        else:
            sampled_instruction = ""
        sources[1]["value"] = sampled_instruction

        data_dict = preprocess_qwen([sources], self.tokenizer, True)

        # Stopword-ignored label masking on assistant tokens (dataset-side, optional)
        input_ids = data_dict['input_ids'][0]
        labels = data_dict['labels'][0]
        if self.enable_stopword_ignore:
            mask = labels != IGNORE_INDEX
            indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
            for idx in indices:
                tid = int(input_ids[idx])
                tok = self.tokenizer.convert_ids_to_tokens(tid)
                if tok is None:
                    continue
                tok_clean = tok.replace('Ġ','').replace('▁','').strip().nlower() if hasattr(str, 'nlower') else tok.replace('Ġ','').replace('▁','').strip().lower()
                if tok_clean == '' or tok_clean in self.stopwords or tok_clean in self.punctuations:
                    labels[idx] = IGNORE_INDEX

        # also return user prompt and all references for multi-reference min-NLL
        user_prompt = sources[0]["value"]
        all_instructions = instructions

        return (
            input_ids,
            labels,
            images,
            torch.tensor(time_ids),
            self.task,
            user_prompt,
            all_instructions,
        )