#!/usr/bin/env bash
# pod_train.sh — local entry point for one-command RunPod training.
#
# Usage:
#   ./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml
#   ./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --force-restart
#
# Reads .runpod.env for RUNPOD_API_KEY, POD_ID, NETWORK_VOLUME_ID, SSH_KEY_PATH, GPU_TYPE.
# On first run (POD_ID unset): creates a new Pod and writes its ID back to .runpod.env.
# On subsequent runs: resumes the stopped Pod.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.runpod.env"

# ── Usage ──────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --config <config-path> [--force-restart]"
    echo ""
    echo "Options:"
    echo "  --config <path>    Path to YAML config (required), e.g. configs/baseline_whisper_small.yaml"
    echo "  --force-restart    Ignore existing checkpoints; start training from epoch 0"
    echo "  --debug            Run with 100 train / 20 val samples (smoke test)"
    echo "  --help             Show this message"
    exit 1
}

# ── Parse arguments ────────────────────────────────────────────────────────
CONFIG=""
FORCE_RESTART=0
DEBUG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --force-restart)
            FORCE_RESTART=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required." >&2
    usage
fi

# ── Load environment ───────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Copy .runpod.env.example to .runpod.env and fill in values." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

SSH_KEY_PATH="${SSH_KEY_PATH:-~/.ssh/id_ed25519}"
GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 4090}"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "Error: RUNPOD_API_KEY is not set in $ENV_FILE" >&2
    exit 1
fi
if [[ -z "${NETWORK_VOLUME_ID:-}" ]]; then
    echo "Error: NETWORK_VOLUME_ID is not set in $ENV_FILE" >&2
    exit 1
fi

# ── Create or resume Pod ───────────────────────────────────────────────────
if [[ -z "${POD_ID:-}" ]]; then
    echo "No POD_ID set — creating a new Pod..."
    # Note: --cost flag may need to be --bidPerGpu on some runpodctl versions.
    # Run 'runpodctl create pod --help' to confirm the correct flag name.
    POD_ID=$(runpodctl create pod \
        --name "whisper-training" \
        --gpuType "$GPU_TYPE" \
        --communityCloud \
        --cost 0.50 \
        --imageName "runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204" \
        --containerDiskSize 50 \
        --networkVolumeId "$NETWORK_VOLUME_ID" \
        --ports "22/tcp" \
        --env "HF_HOME=/workspace/.cache/huggingface" \
        --env "PYTHONPATH=/workspace/childs_speech_recog_chall" \
        2>&1 | grep -oP '(?<=pod )\S+' | head -1)

    if [[ -z "$POD_ID" ]]; then
        echo "Error: Failed to parse POD_ID from runpodctl output." >&2
        exit 1
    fi

    echo "Pod created: $POD_ID"
    # Write POD_ID back to .runpod.env
    if grep -q '^POD_ID=' "$ENV_FILE"; then
        sed -i '' "s|^POD_ID=.*|POD_ID=$POD_ID|" "$ENV_FILE"
    else
        echo "POD_ID=$POD_ID" >> "$ENV_FILE"
    fi
    echo "POD_ID saved to $ENV_FILE"
else
    echo "Resuming Pod $POD_ID..."
    runpodctl start pod "$POD_ID"
fi

# ── Poll until TCP-22 port is live ─────────────────────────────────────────
echo -n "Waiting for SSH on Pod $POD_ID"
POD_IP=""
POD_PORT=""
for i in $(seq 1 60); do
    POD_INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null || true)
    # Parse IP and TCP-22 port — format varies; try common patterns
    POD_IP=$(echo "$POD_INFO" | grep -oP '(?<=IP:\s)\S+' | head -1 || true)
    POD_PORT=$(echo "$POD_INFO" | grep -oP '\d+(?=->22/tcp)' | head -1 || true)

    if [[ -n "$POD_IP" && -n "$POD_PORT" ]]; then
        echo ""
        echo "SSH ready: $POD_IP:$POD_PORT"
        break
    fi
    echo -n "."
    sleep 10
done

if [[ -z "$POD_IP" || -z "$POD_PORT" ]]; then
    echo ""
    echo "Error: Timed out waiting for Pod SSH port. Check 'runpodctl get pod $POD_ID'." >&2
    exit 1
fi

SSH_OPTS="-i ${SSH_KEY_PATH} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SSH_TARGET="root@$POD_IP -p $POD_PORT"

# ── Launch remote training via stdin ──────────────────────────────────────
echo "Launching training on Pod (config: $CONFIG, force-restart: $FORCE_RESTART, debug: $DEBUG)..."
# shellcheck disable=SC2086
CONFIG="$CONFIG" FORCE_RESTART="$FORCE_RESTART" DEBUG="$DEBUG" \
    ssh $SSH_OPTS $SSH_TARGET "CONFIG='$CONFIG' FORCE_RESTART='$FORCE_RESTART' DEBUG='$DEBUG' bash -s" \
    < "$SCRIPT_DIR/remote_train.sh"

# ── Stream log — Ctrl+C disconnects tail only ─────────────────────────────
TAIL_PID=""
cleanup() {
    if [[ -n "$TAIL_PID" ]]; then
        kill "$TAIL_PID" 2>/dev/null || true
    fi
    echo ""
    echo "Log tail disconnected. Training continues on Pod."
    echo ""
    echo "Reattach log:  ssh $SSH_OPTS $SSH_TARGET 'tail -f /workspace/logs/current.log'"
    echo "Reattach tmux: ssh $SSH_OPTS $SSH_TARGET 'tmux attach -t train'"
    echo "Stop Pod now:  runpodctl stop pod $POD_ID"
}
trap cleanup INT TERM

echo ""
echo "─── Live log (Ctrl+C to disconnect, training keeps running) ───"
# shellcheck disable=SC2086
ssh $SSH_OPTS $SSH_TARGET "tail -f /workspace/logs/current.log" &
TAIL_PID=$!
wait "$TAIL_PID" || true
TAIL_PID=""
