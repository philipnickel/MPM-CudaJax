# Regression Baseline

Captured on 2026-06-08 with an NVIDIA A100-SXM4-40GB.

Command:

```bash
pixi run python simulate.py -m sim=benchmark backend=jax,cuda_v1,cuda_v2,cuda_v3,CuTile material=jelly
```

Benchmark preset:

- `sim=benchmark`
- `n_particles=10000000`
- `num_grids=128`
- `num_frames=1`
- `steps_per_frame=10`
- `render.enabled=false`
- material: `jelly`

Results:

| Backend | Total Steps | ms/step | Particles/s |
| --- | ---: | ---: | ---: |
| `jax` | 10 | 98.909 | 1.011e+08 |
| `cuda_v1` | 10 | 60.764 | 1.646e+08 |
| `cuda_v2` | 10 | 55.256 | 1.810e+08 |
| `cuda_v3` | 10 | 51.427 | 1.945e+08 |
| `CuTile` | 10 | 40.002 | 2.500e+08 |

Output sweep:

```text
outputs/sweeps/nvidia_a100_sxm4_40gb/runs/2026-06-08/05-06-24
```
