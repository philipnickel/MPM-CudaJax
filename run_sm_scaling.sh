#!/usr/bin/env bash
# Sweep CUDA MPS active-thread percentage with fresh simulate.py processes.
# CUDA reads CUDA_MPS_ACTIVE_THREAD_PERCENTAGE when JAX creates its context.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$ROOT_DIR/.mps/pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$ROOT_DIR/.mps/log}"

mps_up() {
    echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1
}

STARTED_MPS=0
cleanup() {
    if [ "$STARTED_MPS" = "1" ]; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if ! mps_up; then
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    STARTED_MPS=1
    for _ in 1 2 3 4 5; do
        mps_up && break
        sleep 0.5
    done
fi

PERCENTS="${MPS_PERCENTS:-10 20 40 60 80 100}"
for pct in $PERCENTS; do
    echo ">>> CUDA MPS active thread percentage: ${pct}%"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
        python simulate.py -cn sweep sweep=sm_scaling "mps_thread_percent=${pct}" "$@"
done
