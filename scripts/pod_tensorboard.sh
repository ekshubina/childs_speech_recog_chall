#!/usr/bin/env bash
# pod_tensorboard.sh — open a local TensorBoard against a running RunPod Pod via SSH tunnel.
#
# Usage:
#   ./scripts/pod_tensorboard.sh                                           # default config + port 6006
#   ./scripts/pod_tensorboard.sh --config configs/baseline_whisper_medium.yaml
#   ./scripts/pod_tensorboard.sh --port 6007
#   ./scripts/pod_tensorboard.sh --config configs/baseline_whisper_medium.yaml --port 6007
#
# Ctrl+C closes the local tunnel only — TensorBoard keeps running on the Pod.
# Re-run the script anytime to reopen the tunnel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.runpod.env"

# ── Parse arguments ────────────────────────────────────────────────────────
CONFIG="configs/baseline_whisper_small.yaml"
LOCAL_PORT="6006"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --port)
            LOCAL_PORT="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            echo "Usage: $0 [--config <yaml>] [--port <local-port>]" >&2
            exit 1
            ;;
    esac
done

# ── Load environment ───────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Copy .runpod.env.example to .runpod.env and fill in values." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

# Strip any surrounding quotes from POD_ID (runpodctl sometimes includes them)
POD_ID="${POD_ID//\"/}"
SSH_KEY_PATH="${SSH_KEY_PATH:-~/.ssh/id_ed25519}"
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

if [[ -z "${POD_ID:-}" ]]; then
    echo "Error: POD_ID is not set in $ENV_FILE. Start a Pod first with pod_train.sh." >&2
    exit 1
fi

# ── Derive TB log dir from the YAML config (mirrors remote_train.sh) ──────
# Seq2SeqTrainingArguments defaults logging_dir to {output_dir}/runs when
# logging_dir is not explicitly passed, which is the case in trainer.py.
OUTPUT_DIR_REL=$(grep 'output_dir' "$REPO_ROOT/$CONFIG" | head -1 | sed 's/.*output_dir[[:space:]]*:[[:space:]]*//')
TB_LOGDIR="/workspace/childs_speech_recog_chall/${OUTPUT_DIR_REL}/runs"

# ── Detect SSH connection mode (proxy vs direct) — mirrors pod_train.sh ───
SSH_OPTS="-i ${SSH_KEY_PATH} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SSH_USER=""
SSH_HOST=""
SSH_PORT="22"

CONNECT_CMD=$(runpodctl ssh connect "$POD_ID" 2>&1 || true)

# Secure cloud proxy: "ssh <user>@ssh.runpod.io ..."
if echo "$CONNECT_CMD" | grep -q '@ssh.runpod.io'; then
    SSH_USER=$(echo "$CONNECT_CMD" | grep -oE '[^ ]+@ssh\.runpod\.io' | head -1)
    SSH_HOST="ssh.runpod.io"
    echo "SSH mode: proxy ($SSH_USER)"

# Direct TCP: "ssh root@<IP> -p <PORT>"
elif echo "$CONNECT_CMD" | grep -qE '^ssh root@[0-9]+\.[0-9]+'; then
    SSH_HOST=$(echo "$CONNECT_CMD" | grep -oE '@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tr -d '@' | head -1)
    SSH_PORT=$(echo "$CONNECT_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}' | head -1)
    SSH_PORT="${SSH_PORT:-22}"
    echo "SSH mode: direct ($SSH_HOST:$SSH_PORT)"

# Fallback: parse from pod info table
else
    POD_INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null || true)
    SSH_HOST=$(echo "$POD_INFO" | grep -oE 'IP:[[:space:]]+[^ ]+' | awk '{print $NF}' | head -1 || true)
    SSH_PORT=$(echo "$POD_INFO" | grep -oE '[0-9]+->22/tcp' | grep -oE '^[0-9]+' | head -1 || true)
    SSH_PORT="${SSH_PORT:-22}"
    if [[ -n "$SSH_HOST" ]]; then
        echo "SSH mode: direct fallback ($SSH_HOST:$SSH_PORT)"
    fi
fi

if [[ -z "$SSH_HOST" && -z "$SSH_USER" ]]; then
    echo "Error: Could not determine Pod SSH endpoint. Is Pod $POD_ID running?" >&2
    echo "Check: runpodctl ssh connect $POD_ID" >&2
    exit 1
fi

if [[ -n "$SSH_USER" && "$SSH_HOST" == "ssh.runpod.io" ]]; then
    SSH_TARGET="$SSH_USER"
else
    SSH_TARGET="root@$SSH_HOST -p $SSH_PORT"
fi

# ── Start TensorBoard on Pod if not already running ───────────────────────
echo "Ensuring TensorBoard is running on Pod (logdir: $TB_LOGDIR)..."
# shellcheck disable=SC2086
ssh $SSH_OPTS $SSH_TARGET "
    tmux has-session -t tb 2>/dev/null && echo 'TensorBoard already running.' || \
    tmux new-session -d -s tb \
        'tensorboard --logdir $TB_LOGDIR --host 0.0.0.0 --port 6006 2>&1 | tee /workspace/logs/tensorboard.log'
"

# ── Open SSH port-forward tunnel ──────────────────────────────────────────
echo ""
echo "TensorBoard: http://localhost:$LOCAL_PORT"
echo "Ctrl+C closes this tunnel — TensorBoard keeps running on Pod."
echo "Re-run this script anytime to reopen the tunnel."
echo ""

# shellcheck disable=SC2086
ssh -N \
    -L "${LOCAL_PORT}:localhost:6006" \
    $SSH_OPTS \
    $SSH_TARGET
