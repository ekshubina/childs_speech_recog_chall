#!/usr/bin/env bash
# pod_sync_data.sh — one-time upload of training data to the RunPod network volume.
#
# Usage:
#   ./scripts/pod_sync_data.sh
#
# Uploads directly via the RunPod S3 API — NO Pod needs to be running.
# This avoids GPU billing during upload entirely.
#
# Requires:
#   - aws CLI: brew install awscli
#   - S3 credentials in .runpod.env: RUNPOD_S3_BUCKET, RUNPOD_S3_ENDPOINT,
#     RUNPOD_S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#   (Get credentials from: RunPod console → Network Volumes → your volume → S3 API Access)
#
# Files are uploaded to s3://<bucket>/data/ which maps to /workspace/data/ on the Pod.
# Run once before the first training run. Subsequent pod_train.sh runs skip data transfer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.runpod.env"

# ── Load environment ────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Copy .runpod.env.example to .runpod.env and fill in values." >&2
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

S3_BASE="s3://${RUNPOD_S3_BUCKET}/data"
S3_ARGS="--region $RUNPOD_S3_REGION --endpoint-url $RUNPOD_S3_ENDPOINT"

if ! command -v aws &>/dev/null; then
    echo "Error: aws CLI not found. Install with: brew install awscli" >&2
    exit 1
fi

# ── Upload via S3 API (no Pod required) ───────────────────────────────
echo ""
echo "Uploading to: $S3_BASE/"
echo "Endpoint:     $RUNPOD_S3_ENDPOINT"
echo "No Pod needed — upload goes directly to the network volume via S3 API."
echo ""

# Upload audio directories (parallel multipart, auto-resume on retry)
for dir in audio_0 audio_1 audio_2; do
    echo ">>> Syncing $dir/ ..."
    # shellcheck disable=SC2086
    aws s3 sync \
        "$REPO_ROOT/data/$dir" "$S3_BASE/$dir" \
        $S3_ARGS \
        --no-progress \
        --page-size 100
    echo "    Done: $dir"
done

# Upload manifest
echo ">>> Uploading train_word_transcripts.jsonl ..."
# shellcheck disable=SC2086
aws s3 cp \
    "$REPO_ROOT/data/train_word_transcripts.jsonl" "$S3_BASE/train_word_transcripts.jsonl" \
    $S3_ARGS

# ── Verify ────────────────────────────────────────────────────────────────────────
echo ""
echo "Verifying upload..."
for dir in audio_0 audio_1 audio_2; do
    COUNT=$(aws s3 ls "$S3_BASE/$dir/" $S3_ARGS --recursive | wc -l | tr -d ' ')
    echo "  $dir: $COUNT files"
done
echo ""
echo "Upload complete. Files are immediately available at /workspace/data/ when a Pod mounts the volume."
