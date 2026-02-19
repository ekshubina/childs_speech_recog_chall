#!/usr/bin/env bash
# Run TensorBoard locally against the Google Drive checkpoint logs.
# Works with Google Drive for Desktop on macOS (CloudStorage mount).

set -euo pipefail

RUN_NAME="${1:-baseline_whisper_small}"
PORT="${2:-6006}"

# ── Locate Google Drive "My Drive" on macOS ──────────────────────────────────
GDRIVE_ROOT=""

# 1. Google Drive for Desktop (modern) — ~/Library/CloudStorage/GoogleDrive-*/My Drive
while IFS= read -r candidate; do
    if [[ -d "$candidate" ]]; then
        GDRIVE_ROOT="$candidate"
        break
    fi
done < <(find "$HOME/Library/CloudStorage" -maxdepth 2 -name "My Drive" 2>/dev/null | sort)

# 2. Legacy mount point
if [[ -z "$GDRIVE_ROOT" && -d "/Volumes/GoogleDrive/My Drive" ]]; then
    GDRIVE_ROOT="/Volumes/GoogleDrive/My Drive"
fi

if [[ -z "$GDRIVE_ROOT" ]]; then
    echo "❌  Google Drive 'My Drive' not found."
    echo "    Make sure Google Drive for Desktop is running and signed in."
    exit 1
fi

LOGDIR="$GDRIVE_ROOT/childs_speech_recog_chall/checkpoints/$RUN_NAME/runs"

if [[ ! -d "$LOGDIR" ]]; then
    echo "❌  Log directory not found:"
    echo "    $LOGDIR"
    echo ""
    echo "    Has training started yet?  Check that the run name is correct."
    echo "    Usage: $0 [run_name] [port]"
    echo "    Default run_name: baseline_whisper_small"
    exit 1
fi

echo "✓  Google Drive found : $GDRIVE_ROOT"
echo "✓  TensorBoard logdir : $LOGDIR"
echo "✓  Starting TensorBoard on http://localhost:$PORT"
echo ""

# Activate venv if present
VENV="$(dirname "$0")/../venv/bin/activate"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
fi

tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
