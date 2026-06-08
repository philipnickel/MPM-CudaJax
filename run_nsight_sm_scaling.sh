#!/usr/bin/env bash
# Nsight roofline trajectories swept over CUDA MPS active-thread percentage.
#
# The profiled process must start with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE already
# set, so this loops over percentages and launches normal profile_nsight.py
# multiruns. The nsight_sweep=sm_scaling config has a static sweep directory,
# allowing every percentage to append to one results.parquet and re-render the
# combined roofline trajectory.
#
#   pixi run nsight-sweep-sm
#   MPS_PERCENTS="20 40 80" pixi run nsight-sweep-sm nsight_metrics=roofline nsight_plot=roofline_only
#   pixi run nsight-sweep-sm backend=cuda_v3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$ROOT_DIR/.mps/pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$ROOT_DIR/.mps/log}"

REAL_NCU="$(command -v ncu)"
NCU_WRAPPER_DIR="$(mktemp -d)"
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
