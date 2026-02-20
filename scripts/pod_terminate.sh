#!/usr/bin/env bash
# pod_terminate.sh — one-command Pod termination and cleanup.
#

set -euo pipefail

ENV_FILE=".runpod.env"

source "$ENV_FILE" && runpodctl remove pod "$POD_ID" 2>&1 && sed -i '' 's|^POD_ID=.*|POD_ID=|' "$ENV_FILE" && echo "Done"
