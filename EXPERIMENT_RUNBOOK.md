# Experiment Commands

## Timing Sweeps

```bash
# Weak scaling: N=250k,500k,1M,5M,10M,15M,20M,25M,30M; G grows with N.
pixi run sweep-weak

# Grid/density scaling: fixed N, varying G.
pixi run sweep-density

# Particle-count scaling: fixed G=128, same N axis as above.
pixi run sweep-particles

# SM strong scaling with ordinary simulate.py under CUDA MPS.
# Fixed G=128, N=10M; MPS defaults to 10 20 40 60 80 100.
pixi run sweep-sm

# Timing sweep plots.
pixi run plot-sweeps
```

## Nsight Roofline Trajectories

```bash
# Particle-increase roofline trajectory: fixed G=128, same N axis as above.
pixi run python profile_nsight.py -cn nsight_profile \
    nsight_sweep=particle_count \
    nsight_metrics=roofline \
    nsight_plot=roofline_only

# SM-percentage roofline trajectory under CUDA MPS.
# Fixed G=128, N=10M; MPS defaults to 10 20 40 60 80 100.
pixi run nsight-sweep-sm \
    nsight_metrics=roofline \
    nsight_plot=roofline_only
```
