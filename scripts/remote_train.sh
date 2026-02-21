#!/usr/bin/env bash
# remote_train.sh — runs on the RunPod GPU Pod over SSH.
# Launched by pod_train.sh via: ssh ... bash -s < scripts/remote_train.sh
#
# Required env vars (passed over SSH):
#   CONFIG        — path to YAML config relative to repo root, e.g. configs/baseline_whisper_small.yaml
#   FORCE_RESTART — set to "1" to skip auto-resume from latest checkpoint (default: 0)
#   BRANCH        — git branch to pull (default: main)

set -euo pipefail

REPO=/workspace/childs_speech_recog_chall
LOG=/workspace/logs/current.log
CONFIG="${CONFIG:-configs/baseline_whisper_small.yaml}"
FORCE_RESTART="${FORCE_RESTART:-0}"
DEBUG="${DEBUG:-0}"
BRANCH="${BRANCH:-main}"

# Ensure log directory exists
mkdir -p /workspace/logs

# Ensure essential tools are installed
if ! command -v tmux &>/dev/null; then
    echo "Installing tmux..." | tee -a "$LOG"
    apt-get update -qq && apt-get install -y -qq tmux
fi
if ! command -v runpodctl &>/dev/null; then
    echo "Installing runpodctl..." | tee -a "$LOG"
    wget -q https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -O /usr/local/bin/runpodctl
    chmod +x /usr/local/bin/runpodctl
fi

# Clone repo on first run if it doesn't exist yet
if [[ ! -d "$REPO/.git" ]]; then
    echo "Cloning repo to $REPO..." | tee -a "$LOG"
    git clone https://github.com/ekshubina/childs_speech_recog_chall.git "$REPO"
fi

# Symlink network volume data → repo's data/ so config paths resolve correctly.
# pod_sync_data.sh uploads to s3://<bucket>/data/ which lands at /workspace/data/
# on pods created with --volumePath /workspace, or /runpod/data/ otherwise.
DATA_SRC=""
for candidate in /workspace/data /runpod/data; do
    if [[ -d "$candidate" ]]; then
        DATA_SRC="$candidate"
        break
    fi
done
if [[ -z "$DATA_SRC" ]]; then
    echo "ERROR: training data not found (checked /workspace/data and /runpod/data). Run ./scripts/pod_sync_data.sh first." | tee -a "$LOG"
    exit 1
fi
if [[ ! -L "$REPO/data" || "$(readlink "$REPO/data")" != "$DATA_SRC" ]]; then
    ln -sfn "$DATA_SRC" "$REPO/data"
    echo "Linked $REPO/data → $DATA_SRC" | tee -a "$LOG"
fi

# Derive OUTPUT_DIR from the YAML config — avoids hardcoding the run name
# Expects a line like:  output_dir: checkpoints/baseline_whisper_small
OUTPUT_DIR_REL=$(grep 'output_dir' "$REPO/$CONFIG" | head -1 | sed 's/.*output_dir[[:space:]]*:[[:space:]]*//')
# train.py appends the git branch at runtime (e.g. baseline_whisper_small_ep1).
# Mirror that logic here so checkpoint auto-detection looks in the right directory.
OUTPUT_DIR="$REPO/${OUTPUT_DIR_REL}_${BRANCH}"

# ── Auto-detect latest checkpoint ──────────────────────────────────────────
# Debug runs are smoke tests — never resume from a previous checkpoint.
if [[ "${FORCE_RESTART}" == "1" || "${DEBUG}" == "1" ]]; then
    RESUME_FLAG=""
    LATEST_CKPT=""
else
    LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_CKPT" ]]; then
        RESUME_FLAG="--resume \"$LATEST_CKPT\""
    else
        RESUME_FLAG=""
    fi
fi

# ── Write timestamped run header to log ────────────────────────────────────
{
    echo ""
    echo "====== $(date '+%Y-%m-%d %H:%M:%S') ======"
    echo "CONFIG: $CONFIG"
    if [[ "${FORCE_RESTART}" == "1" ]]; then
        echo "FORCE RESTART: checkpoints ignored"
    elif [[ -n "$LATEST_CKPT" ]]; then
        echo "RESUMING FROM: $LATEST_CKPT"
    else
        echo "STARTING FRESH"
    fi
} >> "$LOG"

# ── Write the training script so tmux can run it without quote escaping ────
TRAIN_SCRIPT=/workspace/logs/run_train.sh
cat > "$TRAIN_SCRIPT" << SCRIPT
#!/usr/bin/env bash
set -euo pipefail

LOG=$LOG
REPO=$REPO
CONFIG=$CONFIG
RESUME_FLAG='$RESUME_FLAG'
DEBUG=$DEBUG
BRANCH=$BRANCH
POD_ID=${POD_ID:-}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-}

# Always stop the Pod on exit (success, error, or signal) to prevent runaway billing
_stop_pod() {
    local exit_code=\$?
    echo "EXIT_CODE=\$exit_code — stopping Pod..." | tee -a "\$LOG"
    if [[ -n "\$RUNPOD_API_KEY" && -n "\$POD_ID" ]]; then
        runpodctl config --apiKey "\$RUNPOD_API_KEY" &>/dev/null || true
        runpodctl stop pod "\$POD_ID" || true
    else
        echo 'WARNING: RUNPOD_API_KEY or POD_ID not set — pod will not auto-stop.' | tee -a "\$LOG"
    fi
}
trap _stop_pod EXIT

cd "\$REPO"

# Sync latest code
git fetch origin
git checkout "\$BRANCH"
git pull origin "\$BRANCH"

# Install all project dependencies into the container's system Python.
# No venv needed — we're root in an isolated container.
echo 'Installing dependencies...' | tee -a "\$LOG"
pip install -q --root-user-action=ignore -r "\$REPO/requirements.txt" 2>&1 | tee -a "\$LOG"

# Build debug flag
DEBUG_FLAG=""
[[ "\$DEBUG" == "1" ]] && DEBUG_FLAG="--debug"

# Run training — all output appended to persistent log
# expandable_segments avoids OOM from CUDA allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
eval python scripts/train.py --config "\$CONFIG" \$RESUME_FLAG \$DEBUG_FLAG 2>&1 | tee -a "\$LOG"
TRAIN_EXIT=\${PIPESTATUS[0]}
echo "Training script finished with exit code \$TRAIN_EXIT" | tee -a "\$LOG"
SCRIPT
chmod +x "$TRAIN_SCRIPT"

# ── Kill any existing train session to prevent duplicate runs ──────────────
tmux kill-session -t train 2>/dev/null || true

# ── Launch detached tmux session ───────────────────────────────────────────
tmux new-session -d -s train "bash $TRAIN_SCRIPT"

echo "Training session started in tmux (session: train). SSH connection closing — training continues on Pod."
echo "Reattach log:  ssh root@<IP> -p <PORT> \"tail -f $LOG\""
echo "Reattach tmux: ssh root@<IP> -p <PORT> \"tmux attach -t train\""
