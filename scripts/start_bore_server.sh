#!/bin/bash

# Navigate to data directory
cd "$(dirname "$0")/../data" || exit 1

echo "Starting HTTP server on port 8080..."
# Start HTTP server in background
python3 -m http.server 8080 &
HTTP_PID=$!

# Wait a moment for server to start
sleep 2

echo "Starting bore tunnel..."
echo "========================================"
# Start bore in foreground (will show the port)
bore local 8080 --to bore.pub

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down HTTP server..."
    kill $HTTP_PID 2>/dev/null
    exit 0
}

# Trap SIGINT and SIGTERM to cleanup
trap cleanup SIGINT SIGTERM
