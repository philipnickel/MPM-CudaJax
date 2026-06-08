#!/usr/bin/env bash
# Sweep Nsight roofline over MPS thread percent.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$ROOT_DIR/.mps/pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$ROOT_DIR/.mps/log}"

REAL_NCU="$(command -v ncu)"
NCU_WRAPPER_DIR="$(mktemp -d)"
# NCU in MPS control mode.
cat >"$NCU_WRAPPER_DIR/ncu" <<EOF
#!/usr/bin/env bash
exec "$REAL_NCU" --mps=control "\$@"
EOF
chmod +x "$NCU_WRAPPER_DIR/ncu"
export PATH="$NCU_WRAPPER_DIR:$PATH"

mps_up() {
    echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1
}

STARTED_MPS=0
cleanup() {
    rm -rf "$NCU_WRAPPER_DIR"
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
    echo ">>> Nsight CUDA MPS active thread percentage: ${pct}%"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
        python profile_nsight.py -cn nsight_profile nsight_sweep=sm_scaling \
            "mps_thread_percent=${pct}" "$@"
done
