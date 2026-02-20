#!/usr/bin/env bash
# pod_tensorboard.sh — open a local TensorBoard against a running RunPod Pod via SSH tunnel.
#
# Usage:
#   ./scripts/pod_tensorboard.sh              # tunnel on localhost:6006 (default)
#   ./scripts/pod_tensorboard.sh 6007         # tunnel on localhost:6007
#
# Ctrl+C closes the local tunnel only — TensorBoard keeps running on the Pod.
# Re-run the script anytime to reopen the tunnel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.runpod.env"

LOCAL_PORT="${1:-6006}"
TB_LOGDIR="/workspace/childs_speech_recog_chall/logs/baseline_whisper_small"

# ── Load environment ───────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Copy .runpod.env.example to .runpod.env and fill in values." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

SSH_KEY_PATH="${SSH_KEY_PATH:-~/.ssh/id_ed25519}"

if [[ -z "${POD_ID:-}" ]]; then
    echo "Error: POD_ID is not set in $ENV_FILE. Start a Pod first with pod_train.sh." >&2
    exit 1
fi

# ── Get live Pod IP and port ───────────────────────────────────────────────
POD_INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null)
POD_IP=$(echo "$POD_INFO" | grep -oP '(?<=IP:\s)\S+' | head -1 || true)
POD_PORT=$(echo "$POD_INFO" | grep -oP '\d+(?=->22/tcp)' | head -1 || true)

if [[ -z "$POD_IP" || -z "$POD_PORT" ]]; then
    echo "Error: Could not determine Pod IP/port. Is Pod $POD_ID running?" >&2
    echo "Check: runpodctl get pod $POD_ID" >&2
    exit 1
fi

SSH_OPTS="-i ${SSH_KEY_PATH} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SSH_TARGET="root@$POD_IP -p $POD_PORT"

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
