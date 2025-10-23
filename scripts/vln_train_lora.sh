#!/bin/bash
export HF_HUB_OFFLINE=1
export HF_HOME=$PWD/checkpoints/hf_home/
export TOKENIZERS_PARALLELISM=false
# Single-node multi-GPU NCCL / CUDA runtime tuning
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# Disable InfiniBand (IB) if not available in single-node docker
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# Use CUDA allocator with expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Encourage fair GPU kernel scheduling in DDP
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Distributed training parameters
NNODES=${NNODES:-1}                           # Number of nodes, default 1 (single node)
NPROC_PER_NODE=${NPROC_PER_NODE:-8}          # GPUs per node, default 8
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}        # Master node address, default localhost for single node
MASTER_PORT=${MASTER_PORT:-12000}  # Default port

# Auto-select a large free port if unset or occupied
if command -v python3 >/dev/null 2>&1; then
SEL_PORT=$(python3 - <<'PY'
import os, socket, random, sys

def is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False

env_port = os.environ.get("MASTER_PORT")
if env_port is not None:
    try:
        p = int(env_port)
        if 1024 <= p <= 65535 and is_free(p):
            print(p)
            sys.exit(0)
    except Exception:
        pass

start = int(os.environ.get("PORT_START", 29500))
end = int(os.environ.get("PORT_END", 65535))
for _ in range(2048):
    p = random.randint(start, end)
    if is_free(p):
        print(p)
        sys.exit(0)

print(0)
PY
)
    if [ -n "$SEL_PORT" ] && [ "$SEL_PORT" != "0" ]; then
        MASTER_PORT=$SEL_PORT
    fi
fi

echo "=== Distributed training config ==="
echo "Num nodes: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Master endpoint: $MASTER_ADDR:$MASTER_PORT"
echo "HF cache dir: $HF_HOME"

VIDEO_FOLDER="data/trajectory_data/R2R"
COLLISION_VIDEO_FOLDER="results/R2R/train/streamvln/vis_0"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# POSIX-compatible replacement: replace '/' with '_' using tr
LLM_VERSION_CLEAN="$(printf '%s' "$LLM_VERSION" | tr '/' '_')"
VISION_MODEL_VERSION="checkpoints/google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="$(printf '%s' "$VISION_MODEL_VERSION" | tr '/' '_')"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_lora_1epoch_196token_8history_32frame"
# PREV_STAGE_CHECKPOINT="checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2"
PREV_STAGE_CHECKPOINT="checkpoints/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Detect GPU count: prefer CUDA_VISIBLE_DEVICES, then nvidia-smi, then Python torch
GPU_COUNT=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _CVD_CLEAN=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]' | sed 's/,,*/,/g;s/^,//;s/,$//')
    if [ -n "$_CVD_CLEAN" ]; then
        GPU_COUNT=$(echo "$_CVD_CLEAN" | awk -F, '{print NF}')
    else
        GPU_COUNT=0
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
fi

if [ -z "$GPU_COUNT" ] || ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    if command -v python3 >/dev/null 2>&1; then
        GPU_COUNT=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(-1)
PY
)
        GPU_COUNT=$(echo "$GPU_COUNT" | tr -d '[:space:]')
    fi
fi

if ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    echo "Error: Unable to determine GPU count. Set CUDA_VISIBLE_DEVICES or install NVIDIA drivers."
    exit 1
fi

if [ "$GPU_COUNT" -le 0 ]; then
    echo "Error: No GPUs visible to the process."
    exit 1
fi

echo "Detected GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]; then
    echo "Warning: NPROC_PER_NODE ($NPROC_PER_NODE) > available GPUs ($GPU_COUNT)"
    echo "Auto-adjust NPROC_PER_NODE to available GPUs"
    NPROC_PER_NODE=$GPU_COUNT
fi
echo "========================"

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT StreamVLN/streamvln/streamvln_train.py \
    --deepspeed StreamVLN/scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --video_folder ${VIDEO_FOLDER} \
    --group_by_task False \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation True \
    --mm_tunable_parts="mm_mlp_adapter,mm_lora_layer" \
    --vision_tower ${PREV_STAGE_CHECKPOINT} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --fp16 True \
    --bf16 False \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --run_name $MID_RUN_NAME \
    --output_dir checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --mm_projector_lr 1e-5 \
    --mm_vision_tower_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 9e-06}' \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to tensorboard \
    --attn_implementation "sdpa" \
    --trajectory_description_training True \
    --collision_training True \
    --collision_video_folder $COLLISION_VIDEO_FOLDER \
    # --bits 8 \
    # --quant_type nf4 \
    # --double_quant True \

# Usage:
# 1. Single node (default): bash scripts/streamvln_train.sh
# 2. Specify GPUs per node: NPROC_PER_NODE=4 bash scripts/streamvln_train.sh
# 3. Multi-node example:
#    Master: MASTER_ADDR=192.168.1.100 NNODES=2 bash scripts/streamvln_train.sh
#    Worker: MASTER_ADDR=192.168.1.100 NNODES=2 bash scripts/streamvln_train.sh
# 4. Custom HF cache: HF_HOME=/path/to/cache bash scripts/streamvln_train.sh
