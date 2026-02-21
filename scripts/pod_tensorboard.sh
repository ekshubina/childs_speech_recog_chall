#!/usr/bin/env bash
# pod_tensorboard.sh — open a local TensorBoard against a running RunPod Pod via SSH tunnel.
#
# Usage:
#   ./scripts/pod_tensorboard.sh                        # all branches (parent logs/ dir)
#   ./scripts/pod_tensorboard.sh --branch ep1           # single branch only
#   ./scripts/pod_tensorboard.sh --config configs/baseline_whisper_medium.yaml
#   ./scripts/pod_tensorboard.sh --port 6007
#
# Omitting --branch points TensorBoard at the parent logs/ directory so all
# branch runs appear together and can be compared side-by-side.
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
BRANCH=""  # empty = all branches

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
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            echo "Usage: $0 [--config <yaml>] [--branch <name>] [--port <local-port>]" >&2
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
# train.py appends the git branch to both output_dir and logging_dir at runtime.
# When logging_dir is set in the YAML, TF events land there (not output_dir/runs).
#
# No --branch: point at the repo root so ALL runs are visible — both old runs that
#              wrote to checkpoints/<name>/runs/ and new runs that write to logs/<name>/.
# --branch X : watch only that branch's exact logging directory.
LOGGING_DIR_REL=$(grep 'logging_dir' "$REPO_ROOT/$CONFIG" | head -1 | sed 's/.*logging_dir[[:space:]]*:[[:space:]]*//' || true)
if [[ -n "$BRANCH" ]]; then
    if [[ -n "$LOGGING_DIR_REL" ]]; then
        TB_LOGDIR="/workspace/childs_speech_recog_chall/${LOGGING_DIR_REL}_${BRANCH}"
    else
        OUTPUT_DIR_REL=$(grep 'output_dir' "$REPO_ROOT/$CONFIG" | head -1 | sed 's/.*output_dir[[:space:]]*:[[:space:]]*//')
        TB_LOGDIR="/workspace/childs_speech_recog_chall/${OUTPUT_DIR_REL}_${BRANCH}/runs"
    fi
else
    # No branch specified — scan the whole repo so both checkpoints/*/runs/ and logs/
    # directories are visible together. TensorBoard only displays dirs with event files.
    TB_LOGDIR="/workspace/childs_speech_recog_chall"
fi
echo "TensorBoard logdir: $TB_LOGDIR"

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

# ── Start TensorBoard on Pod (always restart so logdir is current) ────────
echo "Starting TensorBoard on Pod (logdir: $TB_LOGDIR)..."
# shellcheck disable=SC2086
ssh $SSH_OPTS $SSH_TARGET "
    tmux kill-session -t tb 2>/dev/null || true
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
