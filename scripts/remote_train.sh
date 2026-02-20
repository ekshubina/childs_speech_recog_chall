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

# Derive OUTPUT_DIR from the YAML config — avoids hardcoding the run name
# Expects a line like:  output_dir: checkpoints/baseline_whisper_small
OUTPUT_DIR_REL=$(grep 'output_dir' "$REPO/$CONFIG" | head -1 | sed 's/.*output_dir[[:space:]]*:[[:space:]]*//')
OUTPUT_DIR="$REPO/${OUTPUT_DIR_REL}"

# ── Auto-detect latest checkpoint ──────────────────────────────────────────
if [[ "${FORCE_RESTART}" == "1" ]]; then
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

# ── Kill any existing train session to prevent duplicate runs ──────────────
tmux kill-session -t train 2>/dev/null || true

# ── Launch detached tmux session ───────────────────────────────────────────
# The session: syncs code, activates venv (bootstrapping if needed),
# runs training, writes EXIT_CODE, then self-stops the Pod.
tmux new-session -d -s train "
set -euo pipefail

LOG=$LOG
REPO=$REPO
CONFIG=$CONFIG
RESUME_FLAG='$RESUME_FLAG'

DEBUG=$DEBUG

cd \$REPO

# Sync latest code
git fetch origin
git checkout ${BRANCH}
git pull origin ${BRANCH}

# Bootstrap venv on first run
if [[ ! -d venv ]]; then
    echo 'Creating Python virtual environment...' | tee -a \$LOG
    python -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# Build debug flag
DEBUG_FLAG=""
[[ "$DEBUG" == "1" ]] && DEBUG_FLAG="--debug"

# Run training — all output appended to persistent log
  eval python scripts/train.py --config "\$CONFIG" \$RESUME_FLAG \$DEBUG_FLAG 2>&1 | tee -a \$LOG
TRAIN_EXIT=\${PIPESTATUS[0]}
echo \"EXIT_CODE=\$TRAIN_EXIT\" >> \$LOG

# Self-stop Pod — GPU billing ends immediately
echo 'Training complete. Stopping Pod...' | tee -a \$LOG
runpodctl stop pod \$RUNPOD_POD_ID
"

echo "Training session started in tmux (session: train). SSH connection closing — training continues on Pod."
echo "Reattach log:  ssh root@<IP> -p <PORT> \"tail -f $LOG\""
echo "Reattach tmux: ssh root@<IP> -p <PORT> \"tmux attach -t train\""
