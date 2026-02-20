#!/usr/bin/env bash
# pod_get_results.sh — download training results from the RunPod network volume via S3 API.
#
# Usage:
#   ./scripts/pod_get_results.sh                          # download checkpoints + logs
#   ./scripts/pod_get_results.sh --checkpoints-only       # skip logs
#   ./scripts/pod_get_results.sh --logs-only              # skip checkpoints
#   ./scripts/pod_get_results.sh --run baseline_whisper_small   # specific run only
#
# Works WITHOUT a running Pod — reads directly from the network volume S3 bucket.
# Requires the same S3 credentials used by pod_sync_data.sh in .runpod.env.
#
# Downloaded files land in:
#   checkpoints/<run_name>/   — model checkpoints (checkpoint-N/ + final_model/)
#   logs/                     — training logs (current.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.runpod.env"

# ── Defaults ────────────────────────────────────────────────────────────────
DOWNLOAD_CHECKPOINTS=true
DOWNLOAD_LOGS=true
RUN_NAME=""       # empty = all runs

usage() {
    echo "Usage: $0 [--checkpoints-only | --logs-only] [--run <run_name>]"
    echo ""
    echo "Options:"
    echo "  --checkpoints-only   Download only model checkpoints (skip logs)"
    echo "  --logs-only          Download only training logs (skip checkpoints)"
    echo "  --run <name>         Download a specific run (e.g. baseline_whisper_small)"
    echo "  --help               Show this message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints-only)
            DOWNLOAD_LOGS=false
            shift ;;
        --logs-only)
            DOWNLOAD_CHECKPOINTS=false
            shift ;;
        --run)
            RUN_NAME="$2"
            shift 2 ;;
        --help|-h)
            usage ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            usage ;;
    esac
done

# ── Load credentials ────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

for var in RUNPOD_S3_BUCKET RUNPOD_S3_ENDPOINT RUNPOD_S3_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: $var is not set in $ENV_FILE" >&2
        echo "Get S3 credentials from: RunPod console → Network Volumes → your volume → S3 API Access" >&2
        exit 1
    fi
done
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

if ! command -v aws &>/dev/null; then
    echo "Error: aws CLI not found. Install with: brew install awscli" >&2
    exit 1
fi

S3_ARGS="--region $RUNPOD_S3_REGION --endpoint-url $RUNPOD_S3_ENDPOINT"
# Network volume root maps to s3://<bucket>/
# /workspace/childs_speech_recog_chall/ → s3://<bucket>/childs_speech_recog_chall/
# /workspace/logs/                       → s3://<bucket>/logs/
BUCKET="s3://${RUNPOD_S3_BUCKET}"

echo ""
echo "Bucket:   $BUCKET"
echo "Endpoint: $RUNPOD_S3_ENDPOINT"
echo ""

# ── Download checkpoints ────────────────────────────────────────────────────
if $DOWNLOAD_CHECKPOINTS; then
    if [[ -n "$RUN_NAME" ]]; then
        REMOTE_CKPT_PREFIX="childs_speech_recog_chall/checkpoints/$RUN_NAME"
        LOCAL_CKPT_DIR="$REPO_ROOT/checkpoints/$RUN_NAME"
        echo ">>> Downloading checkpoints/$RUN_NAME/ ..."
        mkdir -p "$LOCAL_CKPT_DIR"
        # shellcheck disable=SC2086
        aws s3 cp \
            "$BUCKET/$REMOTE_CKPT_PREFIX" "$LOCAL_CKPT_DIR" \
            --recursive \
            $S3_ARGS
    else
        REMOTE_CKPT_PREFIX="childs_speech_recog_chall/checkpoints"
        LOCAL_CKPT_DIR="$REPO_ROOT/checkpoints"
        echo ">>> Downloading all checkpoints/ ..."
        mkdir -p "$LOCAL_CKPT_DIR"
        # shellcheck disable=SC2086
        aws s3 cp \
            "$BUCKET/$REMOTE_CKPT_PREFIX" "$LOCAL_CKPT_DIR" \
            --recursive \
            $S3_ARGS
    fi
    echo "    Saved to: $LOCAL_CKPT_DIR"
fi

# ── Download logs ───────────────────────────────────────────────────────────
if $DOWNLOAD_LOGS; then
    LOCAL_LOG_DIR="$REPO_ROOT/logs"
    echo ">>> Downloading logs/ ..."
    mkdir -p "$LOCAL_LOG_DIR"
    # shellcheck disable=SC2086
    aws s3 cp \
        "$BUCKET/logs" "$LOCAL_LOG_DIR" \
        --recursive \
        $S3_ARGS
    echo "    Saved to: $LOCAL_LOG_DIR"
fi

echo ""
echo "Done. Verify checkpoint contents:"
if [[ -n "$RUN_NAME" ]]; then
    ls "$REPO_ROOT/checkpoints/$RUN_NAME/" 2>/dev/null || echo "  (no local files — check if run name is correct)"
else
    ls "$REPO_ROOT/checkpoints/" 2>/dev/null || echo "  (no checkpoints downloaded)"
fi
