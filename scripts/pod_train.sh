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
# Strip any surrounding quotes from POD_ID (runpodctl sometimes includes them)
POD_ID="${POD_ID//\"/}"

SSH_KEY_PATH="${SSH_KEY_PATH:-~/.ssh/id_ed25519}"
# Expand tilde manually so it works inside variables
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"
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
    CREATE_OUTPUT=$(runpodctl create pod \
        --name "whisper-training" \
        --gpuType "$GPU_TYPE" \
        --secureCloud \
        --cost 0.80 \
        --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
        --containerDiskSize 50 \
        --networkVolumeId "$NETWORK_VOLUME_ID" \
        --startSSH \
        --ports "22/tcp" \
        --env "HF_HOME=/workspace/.cache/huggingface" \
        --env "PYTHONPATH=/workspace/childs_speech_recog_chall" \
        2>&1) || true
    echo "runpodctl output: $CREATE_OUTPUT"

    # Try to extract pod ID — runpodctl typically prints: pod "abc123" created
    POD_ID=$(echo "$CREATE_OUTPUT" | grep -oE 'pod "[^"]+"' | grep -oE '"[^"]+"' | tr -d '"' | head -1 || true)
    # Fallback: unquoted format "pod abc123 created"
    if [[ -z "$POD_ID" ]]; then
        POD_ID=$(echo "$CREATE_OUTPUT" | grep -oE 'pod [^ "]+' | awk '{print $2}' | head -1 || true)
    fi

    if [[ -z "$POD_ID" ]]; then
        echo "Error: Failed to parse POD_ID from runpodctl output above." >&2
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

# ── Poll until SSH is ready (supports both community and secure cloud) ─────
# Secure cloud: RunPod proxy  → user@ssh.runpod.io  (no direct IP/port)
# Community cloud: direct TCP → IP:port
echo -n "Waiting for SSH on Pod $POD_ID"
SSH_USER=""
SSH_HOST=""
SSH_PORT="22"
for i in $(seq 1 60); do
    # runpodctl ssh connect prints the full ssh command once the pod is ready
    CONNECT_CMD=$(runpodctl ssh connect "$POD_ID" 2>&1 || true)

    # Secure cloud proxy: "ssh <user>@ssh.runpod.io ..."
    if echo "$CONNECT_CMD" | grep -q '@ssh.runpod.io'; then
        SSH_USER=$(echo "$CONNECT_CMD" | grep -oE '[^ ]+@ssh\.runpod\.io' | head -1)
        SSH_HOST="ssh.runpod.io"
        echo ""
        echo "SSH ready (proxy): $SSH_USER"
        break
    fi

    # Direct TCP: "ssh root@<IP> -p <PORT>"
    if echo "$CONNECT_CMD" | grep -qE '^ssh root@[0-9]+\.[0-9]+'; then
        SSH_HOST=$(echo "$CONNECT_CMD" | grep -oE '@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tr -d '@' | head -1)
        SSH_PORT=$(echo "$CONNECT_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}' | head -1)
        SSH_PORT="${SSH_PORT:-22}"
        echo ""
        echo "SSH ready (direct): $SSH_HOST:$SSH_PORT"
        break
    fi

    # Community cloud fallback: try parsing IP:port from pod info table
    POD_INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null || true)
    POD_IP=$(echo "$POD_INFO" | grep -oE 'IP:[[:space:]]+[^ ]+' | awk '{print $NF}' | head -1 || true)
    POD_PORT_MAPPED=$(echo "$POD_INFO" | grep -oE '[0-9]+->22/tcp' | grep -oE '^[0-9]+' | head -1 || true)
    if [[ -n "$POD_IP" && -n "$POD_PORT_MAPPED" ]]; then
        SSH_HOST="$POD_IP"
        SSH_PORT="$POD_PORT_MAPPED"
        echo ""
        echo "SSH ready (direct): $SSH_HOST:$SSH_PORT"
        break
    fi

    echo -n "."
    sleep 10
done

if [[ -z "$SSH_HOST" ]]; then
    echo ""
    echo "Error: Timed out waiting for Pod SSH. Check 'runpodctl get pod $POD_ID'." >&2
    exit 1
fi

SSH_OPTS="-i ${SSH_KEY_PATH} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
if [[ -n "$SSH_USER" && "$SSH_HOST" == "ssh.runpod.io" ]]; then
    # Proxy mode: user already contains user@host
    SSH_TARGET="$SSH_USER"
else
    SSH_TARGET="root@$SSH_HOST -p $SSH_PORT"
fi

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
    echo "Stop Pod now:  runpodctl stop pod ${POD_ID}"
}
trap cleanup INT TERM

echo ""
echo "─── Live log (Ctrl+C to disconnect, training keeps running) ───"
# shellcheck disable=SC2086
ssh $SSH_OPTS $SSH_TARGET "tail -f /workspace/logs/current.log" &
TAIL_PID=$!
wait "$TAIL_PID" || true
TAIL_PID=""
