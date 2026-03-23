import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


def get_cube(center, size, num, add_noise=False):
    start = np.array(center) - np.array(size) / 2
    end = np.array(center) + np.array(size) / 2
    x = np.linspace(start[0], end[0], num)
    y = np.linspace(start[1], end[1], num)
    z = np.linspace(start[2], end[2], num)
    cube = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
    if add_noise:
        noisy = start + np.random.rand(*cube.shape) * (end - start)
        cube = np.concatenate([cube, noisy], axis=0)
    return cube


def visualize_frames(frames, export_path, center=[0.5, 0.5, 0.5],
                     size=[2.0, 2.0, 2.0], c='blue', s=20, fps=30):
    xlim = [center[0] - size[0]/2, center[0] + size[0]/2]
    ylim = [center[1] - size[1]/2, center[1] + size[1]/2]
    zlim = [center[2] - size[2]/2, center[2] + size[2]/2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

    def update(frame):
        ax.cla()
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.scatter(frames[frame][:, 0], frames[frame][:, 1], frames[frame][:, 2], s=s, c=c)
        ax.set_title(f'Frame {frame}')

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()


# ---------------------------------------------------------------------------
# Per-stage timing helpers
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates per-stage wall-clock times across substeps."""

    def __init__(self):
        self.stages = {}
        self._start = None
        self._current = None

    def start(self, name):
        self._current = name
        self._start = time.perf_counter()

    def stop(self):
        elapsed = time.perf_counter() - self._start
        if self._current not in self.stages:
            self.stages[self._current] = []
        self.stages[self._current].append(elapsed)
        self._current = None

    def flush_frame(self):
        """Pop accumulated times for the current frame and return per-stage totals in ms."""
        out = {}
        for name, times in self.stages.items():
            out[name] = sum(times) * 1000  # ms
        self.stages.clear()
        return out

    def summary_from_frames(self, frame_timings):
        """Compute overall summary from a list of per-frame dicts."""
        all_stages = {}
        for ft in frame_timings:
            for name, ms in ft.items():
                all_stages.setdefault(name, []).append(ms)
        out = {}
        for name, vals in all_stages.items():
            arr = np.array(vals)
            out[name] = {
                'mean_ms': float(arr.mean()),
                'std_ms': float(arr.std()),
                'total_ms': float(arr.sum()),
                'count': len(vals),
            }
        return out


# ---------------------------------------------------------------------------
# Backend-specific runners
# ---------------------------------------------------------------------------

def run_jax(cfg: DictConfig):
    import jax
    import jax.numpy as jnp
    from mpm_jax.solver import MPMState, make_params, step, simulate_frame
    from mpm_jax.solver import compute_weights_and_indices, p2g_compute, p2g_scatter, grid_update, g2p
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.boundary import build_boundary_fns

    sim = cfg.sim
    mat = cfg.material
    bench = cfg.get('benchmark', False)
    kernel_name = cfg.get('kernel', {}).get('name', 'jax')

    # Build p2g_fn based on kernel config
    p2g_fn = None  # None = default JAX implementation
    if kernel_name == 'cuda_v1':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='scatter')
        if p2g_fn is None:
            print(f"WARNING: kernel={kernel_name} requested but CUDA not available, falling back to JAX")
        else:
            print(f"Using CUDA P2G scatter kernel (v1)")
    elif kernel_name == 'cuda_v2':
        from mpm_jax.cuda.p2g_cuda import is_available
        if not is_available('fused'):
            print(f"WARNING: kernel={kernel_name} requested but CUDA not available, falling back to JAX")
        else:
            print(f"Using CUDA fused P2G kernel (v2)")
    else:
        print("Using JAX P2G kernel")

    cube_np = get_cube(center=list(sim.center), size=[0.5, 0.5, 0.5], num=10, add_noise=True)
    particles = jnp.array(cube_np, dtype=jnp.float32)
    n = particles.shape[0]

    params = make_params(
        n_particles=n, num_grids=sim.num_grids, dt=sim.dt,
        gravity=list(sim.gravity), rho=sim.rho,
        clip_bound=sim.clip_bound, damping=sim.damping,
        center=list(sim.center), size=list(sim.size),
    )

    g = jnp.arange(params.num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing='ij')
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    pre_fn, post_fn = build_boundary_fns(
        list(sim.boundary_conditions), grid_x, params.dx,
        particles, params.dt, params.p_mass,
    )

    elasticity_fn = get_constitutive(mat.elasticity)
    plasticity_fn = get_constitutive(mat.plasticity)

    state = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    # Warmup JIT
    for _ in range(3):
        stress = elasticity_fn(state.F)
        state = step(params, state, stress, pre_fn, post_fn, 0.0, p2g_fn=p2g_fn)
        state = state._replace(F=plasticity_fn(state.F))
    jax.block_until_ready(state.x)

    # Reset
    state = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    timer = StageTimer()
    frames = []
    frame_metrics = []  # collect metrics outside timing
    frame_timings = []  # per-frame stage breakdown
    sim_time = 0.0
    t0 = time.perf_counter()

    for frame in tqdm(range(sim.num_frames), desc='JAX'):
        if not bench:
            frames.append(np.array(state.x))
        for _ in range(sim.steps_per_frame):
            # --- P2G: stress + weights + per-particle compute + scatter ---
            timer.start('p2g_compute')
            stress = elasticity_fn(state.F)
            x, v = pre_fn(state.x, state.v, sim_time)
            weight, dweight, dpos, index = compute_weights_and_indices(
                x, params.inv_dx, params.dx, params.num_grids)
            mv, m = p2g_compute(v, state.C, stress, weight, dweight, dpos,
                                params.dt, params.vol, params.p_mass)
            jax.block_until_ready(mv)
            timer.stop()

            timer.start('p2g_scatter')
            if p2g_fn is not None:
                # CUDA kernel: p2g_fn does compute+scatter, but we already
                # computed mv/m above, so call scatter directly
                from mpm_jax.cuda.p2g_cuda import cuda_p2g_scatter
                grid_mv, grid_m = cuda_p2g_scatter(mv, m, index, params.num_grids)
            else:
                grid_mv, grid_m = p2g_scatter(mv, m, index, params.num_grids)
            jax.block_until_ready(grid_mv)
            timer.stop()

            # --- Grid update ---
            timer.start('grid_update')
            grid_mv = grid_update(grid_mv, grid_m, params.gravity, params.dt, params.damping)
            grid_mv = post_fn(grid_mv, grid_m, sim_time)
            jax.block_until_ready(grid_mv)
            timer.stop()

            # --- G2P ---
            timer.start('g2p')
            new_x, new_v, new_C, new_F = g2p(grid_mv, weight, dweight, dpos, index,
                                               state.F, x, params.dt, params.inv_dx, params.clip_bound)
            new_F = plasticity_fn(new_F)
            jax.block_until_ready(new_x)
            timer.stop()

            state = MPMState(x=new_x, v=new_v, C=new_C, F=new_F)
            sim_time += params.dt

        # Flush per-frame stage timings (sums substeps within this frame)
        ft = timer.flush_frame()
        frame_ms = sum(ft.values())
        frame_timings.append(ft)
        frame_metrics.append({
            'x_mean_z': float(state.x[:, 2].mean()),
            'v_max': float(jnp.abs(state.v).max()),
            'frame_ms': frame_ms,
            **{f'{k}_ms': v for k, v in ft.items()},
        })

    jax.block_until_ready(state.x)
    elapsed = time.perf_counter() - t0

    total_steps = sim.num_frames * sim.steps_per_frame
    return frames, elapsed, total_steps, timer.summary_from_frames(frame_timings), frame_metrics


def run_pytorch(cfg: DictConfig):
    import sys
    vendor_path = os.path.join(os.path.dirname(__file__), "vendor", "MPM-PyTorch")
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    import torch
    from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive

    sim = cfg.sim
    mat = cfg.material
    bench = cfg.get('benchmark', False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cube_np = get_cube(center=list(sim.center), size=[0.5, 0.5, 0.5], num=10, add_noise=True)
    particles = torch.tensor(cube_np, dtype=torch.float32, device=device)
    n = particles.shape[0]

    solver = MPMSolver(particles, enable_train=False, device=device)
    set_boundary_conditions(solver, sim.boundary_conditions)

    elasticity_name = mat.elasticity.name
    plasticity_name = mat.plasticity.name
    elasticity = get_constitutive(elasticity_name, device=device)
    plasticity = get_constitutive(plasticity_name, device=device)

    x = particles.clone()
    v = torch.stack([torch.tensor(list(sim.initial_velocity), device=device) for _ in range(n)])
    C = torch.zeros((n, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n, 1, 1)

    timer = StageTimer()
    frames = []
    frame_metrics = []
    frame_timings = []
    t0 = time.perf_counter()

    for frame in tqdm(range(sim.num_frames), desc='PyTorch'):
        if not bench:
            frames.append(x.detach().cpu().numpy())
        for _ in range(sim.steps_per_frame):
            # Full timestep: stress + P2G + grid + G2P + plasticity
            # PyTorch solver bundles P2G/grid/G2P internally so we
            # can't split to match JAX's 3-stage breakdown.
            timer.start('timestep')
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timer.stop()

        ft = timer.flush_frame()
        frame_ms = sum(ft.values())
        frame_timings.append(ft)
        frame_metrics.append({
            'x_mean_z': float(x[:, 2].mean().item()),
            'v_max': float(torch.abs(v).max().item()),
            'frame_ms': frame_ms,
            **{f'{k}_ms': v for k, v in ft.items()},
        })

    elapsed = time.perf_counter() - t0
    total_steps = sim.num_frames * sim.steps_per_frame
    return frames, elapsed, total_steps, timer.summary_from_frames(frame_timings), frame_metrics


# ---------------------------------------------------------------------------
# Wandb logging (all after timing is complete)
# ---------------------------------------------------------------------------

def log_results(backend, elapsed, total_steps, summary, frame_metrics, frames, cfg, export_path):
    """Log all metrics to wandb. Called only after timing is done."""
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    steps_per_frame = cfg.sim.steps_per_frame

    # Per-frame time series (stage timings + physics metrics)
    for i, fm in enumerate(frame_metrics):
        step_idx = (i + 1) * steps_per_frame
        wandb.log({k: v for k, v in fm.items()}, step=step_idx)

    # Summary scalars
    n_particles = frames[0].shape[0] if frames else 0
    wandb.log({
        'summary/total_steps': total_steps,
        'summary/elapsed_s': elapsed,
        'summary/steps_per_sec': steps_per_sec,
        'summary/ms_per_step': ms_per_step,
        'summary/n_particles': n_particles,
    })

    # Per-stage breakdown table
    stage_table = wandb.Table(
        columns=["stage", "mean_ms", "std_ms", "total_ms", "count", "pct"],
    )
    total_ms = sum(s['total_ms'] for s in summary.values())
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        pct = stats['total_ms'] / total_ms * 100 if total_ms > 0 else 0
        stage_table.add_data(stage, round(stats['mean_ms'], 4), round(stats['std_ms'], 4),
                             round(stats['total_ms'], 2), stats['count'], round(pct, 1))
        wandb.log({
            f'stage/{stage}_mean_ms': stats['mean_ms'],
            f'stage/{stage}_pct': pct,
        })
    wandb.log({'stage_breakdown': stage_table})

    # Animation
    if export_path and os.path.exists(export_path):
        wandb.log({'animation': wandb.Video(export_path, format='gif')})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    backend = cfg.backend.name
    print(f"Backend: {backend}")

    # Init wandb
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="MPM-CudaJAX",
        name=f"{cfg.tag}_{backend}",
        config=wandb_cfg,
        tags=[backend, cfg.tag],
    )

    # Run simulation (timing-critical — no wandb calls inside)
    if backend == "jax":
        frames, elapsed, total_steps, summary, frame_metrics = run_jax(cfg)
    elif backend == "pytorch":
        frames, elapsed, total_steps, summary, frame_metrics = run_pytorch(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Print timing summary
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    print(f"\n{backend}: {total_steps} steps in {elapsed:.2f}s ({steps_per_sec:.1f} steps/s, {ms_per_step:.2f} ms/step)")

    total_ms = sum(s['total_ms'] for s in summary.values())
    print(f"\nPer-stage timing (per frame, {cfg.sim.steps_per_frame} substeps each):")
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        pct = stats['total_ms'] / total_ms * 100 if total_ms > 0 else 0
        print(f"  {stage:15s}: {stats['mean_ms']:8.3f} ms/frame ({pct:5.1f}%  std={stats['std_ms']:.3f}  n={stats['count']})")

    # Render GIF (skip in benchmark mode)
    export_path = None
    if not cfg.get('benchmark', False) and frames:
        orig_cwd = hydra.utils.get_original_cwd()
        output_dir = os.path.join(orig_cwd, cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, f"{cfg.tag}_{backend}.gif")
        print(f"\nRendering to {export_path}...")
        visualize_frames(frames, export_path, size=[1, 1, 1], c=cfg.material.color)
    elif cfg.get('benchmark', False):
        print("\nBenchmark mode: skipping GIF rendering.")

    # Log everything to wandb (after all timing is done)
    log_results(backend, elapsed, total_steps, summary, frame_metrics, frames, cfg, export_path)
    wandb.finish()


if __name__ == "__main__":
    main()
