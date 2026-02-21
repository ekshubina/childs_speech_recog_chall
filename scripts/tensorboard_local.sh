#!/usr/bin/env bash
# Run TensorBoard locally against downloaded checkpoint logs.
#
# Lookup order:
#   1. checkpoints/<run>/runs/  (local — synced via pod_storage.py get)
#   2. Google Drive for Desktop (macOS CloudStorage mount)
#
# Usage: ./scripts/tensorboard_local.sh [run_name] [port]
#   run_name defaults to baseline_whisper_small
#   port     defaults to 6006
#
# To download only the TensorBoard logs from S3 (fast — ~91 KB):
#   python scripts/pod_storage.py get --checkpoints-only --run baseline_whisper_small --subdir runs

set -euo pipefail

RUN_NAME="${1:-baseline_whisper_small}"
PORT="${2:-6006}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

LOGDIR=""

# ── 1. Local checkpoints/ (preferred) ───────────────────────────────────────
LOCAL_LOGDIR="$REPO_ROOT/checkpoints/$RUN_NAME/runs"
if [[ -d "$LOCAL_LOGDIR" ]]; then
    LOGDIR="$LOCAL_LOGDIR"
    echo "✓  Using local logdir : $LOGDIR"
fi

# ── 2. Google Drive fallback ─────────────────────────────────────────────────
if [[ -z "$LOGDIR" ]]; then
    GDRIVE_ROOT=""
    while IFS= read -r candidate; do
        if [[ -d "$candidate" ]]; then
            GDRIVE_ROOT="$candidate"
            break
        fi
    done < <(find "$HOME/Library/CloudStorage" -maxdepth 2 -name "My Drive" 2>/dev/null | sort)

    if [[ -z "$GDRIVE_ROOT" && -d "/Volumes/GoogleDrive/My Drive" ]]; then
        GDRIVE_ROOT="/Volumes/GoogleDrive/My Drive"
    fi

    if [[ -n "$GDRIVE_ROOT" ]]; then
        GDRIVE_LOGDIR="$GDRIVE_ROOT/childs_speech_recog_chall/checkpoints/$RUN_NAME/runs"
        if [[ -d "$GDRIVE_LOGDIR" ]]; then
            LOGDIR="$GDRIVE_LOGDIR"
            echo "✓  Using Google Drive logdir : $LOGDIR"
        fi
    fi
fi

if [[ -z "$LOGDIR" ]]; then
    echo "❌  No TensorBoard logs found for run '$RUN_NAME'."
    echo ""
    echo "Download them first (fast — only ~91 KB):"
    echo "  python scripts/pod_storage.py get --checkpoints-only --run $RUN_NAME --subdir runs"
    echo ""
    echo "Then re-run: $0 $RUN_NAME $PORT"
    exit 1
fi

echo "✓  Starting TensorBoard on http://localhost:$PORT"
echo ""

# Activate venv if present
VENV="$(dirname "$0")/../venv/bin/activate"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
fi

tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
