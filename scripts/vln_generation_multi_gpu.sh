# --- Configuration ---
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=${MASTER_PORT:-12000}  # Default port

# RxR dataset configuration via environment variables
export RXR_ROLES=${RXR_ROLES:-"guide,follower"}  # Default to both roles
export RXR_LANGUAGES=${RXR_LANGUAGES:-"*"}       # Default to all languages
export RXR_CONTENT_SCENES=${RXR_CONTENT_SCENES:-"*"}  # Default to all scenes

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

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
  NPROC=$(nvidia-smi -L | wc -l)
fi

# --- Evaluation Command ---
torchrun --nproc_per_node="$NPROC" \
    --master_port=$MASTER_PORT \
    scripts/trajectory_generation.py \
    --config_path config/vln_r2r.yaml \
    --output_path data/trajectory_data \
    --data_path data/datasets/scalevln/scalevln_subset_150k.json.gz