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

MARKER_DIR="$REPO_ROOT/.sync_markers"
mkdir -p "$MARKER_DIR"

# Upload audio directories (parallel multipart, auto-resume on retry)
# Marker files in .sync_markers/ record completed dirs so re-runs skip the
# remote listing phase (which triggers RunPod S3 pagination bugs on large dirs).
for dir in audio_0 audio_1 audio_2; do
    MARKER="$MARKER_DIR/$dir.done"
    if [[ -f "$MARKER" ]]; then
        echo ">>> Skipping $dir/ (already uploaded — delete $MARKER to re-sync)"
        continue
    fi
    echo ">>> Syncing $dir/ ..."
    # shellcheck disable=SC2086
    TOTAL=$(find "$REPO_ROOT/data/$dir" -type f | wc -l | tr -d ' ')
    echo ">>> Uploading $dir/ ($TOTAL files) ..."
    # Use cp --recursive instead of sync: sync must LIST the remote bucket first,
    # which triggers RunPod S3's broken pagination on large directories.
    # cp --recursive reads only local files and uploads — no remote listing needed.
    # shellcheck disable=SC2086
    aws s3 cp \
        "$REPO_ROOT/data/$dir" "$S3_BASE/$dir" \
        --recursive \
        $S3_ARGS \
        --no-progress \
        | awk -v total="$TOTAL" '
            /^upload:/ { n++; printf "\r    %d / %d uploaded", n, total; fflush() }
            END { printf "\n" }
        '
    touch "$MARKER"
    echo "    Done: $dir"
done

# Upload manifest
echo ">>> Uploading train_word_transcripts.jsonl ..."
# shellcheck disable=SC2086
aws s3 cp \
    "$REPO_ROOT/data/train_word_transcripts.jsonl" "$S3_BASE/train_word_transcripts.jsonl" \
    $S3_ARGS \
    --no-progress

# ── Verify ────────────────────────────────────────────────────────────────────────
# head-object on first+last local file per dir: 2 API calls each, no listing.
echo ""
echo "Verifying upload (spot-checking first + last file per dir)..."
ALL_OK=true
for dir in audio_0 audio_1 audio_2; do
    # Disable pipefail temporarily: find|sort|head -1 causes SIGPIPE (exit 141) under pipefail
    set +o pipefail
    FIRST=$(find "$REPO_ROOT/data/$dir" -type f | sort | head -1)
    LAST=$(find "$REPO_ROOT/data/$dir" -type f | sort | tail -1)
    TOTAL=$(find "$REPO_ROOT/data/$dir" -type f | wc -l | tr -d ' ')
    set -o pipefail
    OK=true
    for f in "$FIRST" "$LAST"; do
        KEY="data/$dir/${f##*/}"
        # shellcheck disable=SC2086
        if ! aws s3api head-object \
                --bucket "$RUNPOD_S3_BUCKET" \
                --key "$KEY" \
                $S3_ARGS \
                --output text \
                --query 'ContentLength' &>/dev/null; then
            echo "  MISSING: $KEY"
            OK=false; ALL_OK=false
        fi
    done
    $OK && echo "  $dir: OK  ($TOTAL local files; first+last confirmed in S3)"
done
$ALL_OK || { echo "Some files missing — re-run to retry."; exit 1; }
echo ""
echo "Upload complete. Files are immediately available at /workspace/data/ when a Pod mounts the volume."
