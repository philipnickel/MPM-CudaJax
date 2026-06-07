#!/usr/bin/env bash
# Strong-scaling-via-MPS sweep.
#
# CUDA reads CUDA_MPS_ACTIVE_THREAD_PERCENTAGE once, when JAX creates its
# context, so the MPS axis can't be a single-process Hydra multirun. This script
# runs the `sweep=sm_scaling` config once per percentage — each a fresh process,
# so each picks up its own SM clamp. The config's sweep dir is static, so all
# percentages aggregate there and the ScalingPlotCallback renders the combined
# result after the last run.
#
# MPS is started here (only for this sweep) and stopped at the end, so ordinary
# `pixi run` commands never depend on a running MPS daemon.
#
#   pixi run sweep-sm                          # all backends, 10/25/50/75/100%
#   MPS_PERCENTS="20 100" pixi run sweep-sm    # custom percentages
#   pixi run sweep-sm backend=cutile_v3        # restrict backends (extra args pass through)
set -euo pipefail

mps_up() { echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; }

if ! mps_up; then
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    for _ in 1 2 3 4 5; do mps_up && break; sleep 0.5; done
    STARTED_MPS=1
fi

PERCENTS="${MPS_PERCENTS:-10 25 50 75 100}"
for pct in $PERCENTS; do
    echo ">>> MPS thread percentage: ${pct}%"
    # Set the clamp in the env before launching: each run is a fresh process,
    # and CUDA reads this once when JAX creates its context. mps_thread_percent
    # is also passed so the value is recorded in metrics for the plot x-axis.
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${pct}" \
        python simulate.py -cn sweep sweep=sm_scaling "mps_thread_percent=${pct}" "$@"
done

# Leave the GPU as we found it: only stop MPS if this script started it.
if [ "${STARTED_MPS:-0}" = "1" ]; then
    echo quit | nvidia-cuda-mps-control
fi
