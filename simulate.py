"""MLS-MPM simulation with pluggable P2G kernels."""
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import jax
import jax.numpy as jnp

from mpm_jax.state import MPMState, make_params
from mpm_jax.grid_update import grid_update
from mpm_jax.g2p import g2p
from mpm_jax.p2g import get_p2g_fn
from mpm_jax.boundary import build_boundary_fns


def get_particles(n_particles, center, size, initial_velocity):
    """Create particles on a regular grid with noisy duplicates (matches PyTorch).

    Creates num^3 particles on a regular grid, then adds num^3 random particles,
    where num = round(n_particles/2)^(1/3). Total ≈ n_particles.
    """
    num = round((n_particles / 2) ** (1/3))
    center = jnp.array(center)
    size = jnp.array(size)
    start = center - size / 2
    end = center + size / 2

    # Regular grid
    lx = jnp.linspace(start[0], end[0], num)
    ly = jnp.linspace(start[1], end[1], num)
    lz = jnp.linspace(start[2], end[2], num)
    gx, gy, gz = jnp.meshgrid(lx, ly, lz, indexing='ij')
    grid_pts = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # Noisy duplicates
    key = jax.random.PRNGKey(0)
    noisy_pts = start + jax.random.uniform(key, grid_pts.shape) * size

    x = jnp.concatenate([grid_pts, noisy_pts], axis=0)
    n = x.shape[0]
    v = jnp.tile(jnp.array(initial_velocity), (n, 1))
    C = jnp.zeros((n, 3, 3))
    F = jnp.tile(jnp.eye(3), (n, 1, 1))
    return MPMState(x=x, v=v, C=C, F=F)


def visualize_frames(frames, export_path, center=[0.5, 0.5, 0.5],
                     size=[2.0, 2.0, 2.0], c='blue', s=20, fps=30):
    """Render frames as a 3D scatter plot GIF."""
    xlim = [center[0] - size[0]/2, center[0] + size[0]/2]
    ylim = [center[1] - size[1]/2, center[1] + size[1]/2]
    zlim = [center[2] - size[2]/2, center[2] + size[2]/2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    def update(frame):
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.scatter(frames[frame][:, 0], frames[frame][:, 1], frames[frame][:, 2], s=s, c=c)
        ax.set_title(f'Frame {frame}')

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()


def simulate(cfg):
    """Run the simulation. Returns (step_timings, frames, total_time)."""
    sim = cfg.sim
    bench = cfg.get('benchmark', False)

    # Build initial state first (actual N may differ from config due to grid sampling)
    state = get_particles(
        sim.n_particles,
        center=list(sim.center),
        size=list(sim.size),
        initial_velocity=list(sim.initial_velocity),
    )
    actual_n = state.x.shape[0]

    # Build params with actual particle count (for correct vol = prod(size)/N)
    params = make_params(
        n_particles=actual_n,
        num_grids=sim.num_grids,
        dt=sim.dt,
        gravity=list(sim.gravity),
        rho=sim.rho,
        clip_bound=sim.get("clip_bound", 0.5),
        damping=sim.get("damping", 1.0),
        center=list(sim.center),
        size=list(sim.size),
    )

    # Build boundary conditions
    G = int(params.num_grids)
    grid_x = jnp.stack(jnp.meshgrid(
        jnp.arange(G), jnp.arange(G), jnp.arange(G), indexing='ij'
    ), axis=-1).reshape(-1, 3).astype(jnp.float32)
    pre_particle_fn, post_grid_fn = build_boundary_fns(
        sim.get("boundary_conditions", []),
        grid_x, params.dx, state.x, params.dt, params.p_mass,
    )

    # Initialize CUDA runtime if needed
    runtime = None
    if cfg.kernel.name.startswith("cuda"):
        from mpm_jax.cuda.runtime import CudaRuntime
        runtime = CudaRuntime()

    # Build P2G function
    p2g_fn = get_p2g_fn(cfg, runtime)
    print(f"N={sim.n_particles}, G={sim.num_grids}, kernel={cfg.kernel.name}")

    # Warmup: run one step to trigger JIT compilation
    grid_mv, grid_m = p2g_fn(state, params)
    grid_v = grid_update(grid_mv, grid_m, params)
    warmup_state = g2p(state, grid_v, params)
    jax.block_until_ready(warmup_state.x)

    # Reset state for timed run (same particles as before warmup)
    state = get_particles(
        sim.n_particles,
        center=list(sim.center),
        size=list(sim.size),
        initial_velocity=list(sim.initial_velocity),
    )
    assert state.x.shape[0] == actual_n

    step_timings = []
    frames = []
    t0 = time.perf_counter()

    for frame in tqdm(range(sim.num_frames), desc='Simulating'):
        if not bench:
            frames.append(np.array(state.x))

        for step in range(sim.steps_per_frame):
            step_t0 = time.perf_counter()

            # Pre-particle BCs
            sim_time = (frame * sim.steps_per_frame + step) * params.dt
            new_x, new_v = pre_particle_fn(state.x, state.v, sim_time)
            state = state._replace(x=new_x, v=new_v)

            # P2G
            t_p2g_start = time.perf_counter()
            grid_mv, grid_m = p2g_fn(state, params)
            jax.block_until_ready((grid_mv, grid_m))
            t_p2g_end = time.perf_counter()

            # Grid update + post-grid BCs
            t_grid_start = time.perf_counter()
            grid_v = grid_update(grid_mv, grid_m, params)
            grid_v = post_grid_fn(grid_v, grid_m, sim_time)
            jax.block_until_ready(grid_v)
            t_grid_end = time.perf_counter()

            # G2P
            t_g2p_start = time.perf_counter()
            state = g2p(state, grid_v, params)
            jax.block_until_ready(state.x)
            t_g2p_end = time.perf_counter()

            step_timings.append({
                "p2g_ms": (t_p2g_end - t_p2g_start) * 1000,
                "grid_update_ms": (t_grid_end - t_grid_start) * 1000,
                "g2p_ms": (t_g2p_end - t_g2p_start) * 1000,
                "step_ms": (time.perf_counter() - step_t0) * 1000,
            })

    total_time = time.perf_counter() - t0
    return step_timings, frames, total_time


def log_to_wandb(cfg, step_timings, total_time, gif_path=None):
    """Log per-frame aggregated timings, summary, and animation to wandb."""
    name = f"{cfg.kernel.name}_{cfg.tag}_N{cfg.sim.n_particles}_G{cfg.sim.num_grids}"
    group = cfg.get("group", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="mpm-cuda", name=name, group=group, config=OmegaConf.to_container(cfg))
    sim = cfg.sim
    spf = sim.steps_per_frame
    for f in range(sim.num_frames):
        fs = step_timings[f*spf:(f+1)*spf]
        wandb.log({
            "frame_p2g_ms": sum(t["p2g_ms"] for t in fs),
            "frame_grid_update_ms": sum(t["grid_update_ms"] for t in fs),
            "frame_g2p_ms": sum(t["g2p_ms"] for t in fs),
            "frame_step_ms": sum(t["step_ms"] for t in fs),
        })
    total_steps = len(step_timings)
    wandb.summary.update({
        "mean_p2g_ms": np.mean([t["p2g_ms"] for t in step_timings]),
        "mean_grid_update_ms": np.mean([t["grid_update_ms"] for t in step_timings]),
        "mean_g2p_ms": np.mean([t["g2p_ms"] for t in step_timings]),
        "mean_step_ms": np.mean([t["step_ms"] for t in step_timings]),
        "total_steps": total_steps,
        "total_elapsed_s": total_time,
        "steps_per_sec": total_steps / total_time,
        "n_particles": cfg.sim.n_particles,
        "kernel": cfg.kernel.name,
    })
    if gif_path and os.path.exists(gif_path):
        wandb.log({"animation": wandb.Video(gif_path, format="gif")})
    wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Run simulation
    step_timings, frames, total_time = simulate(cfg)

    # Print summary
    total_steps = len(step_timings)
    steps_per_sec = total_steps / total_time
    ms_per_step = total_time / total_steps * 1000
    print(f"\n{cfg.kernel.name}: {total_steps} steps in {total_time:.2f}s "
          f"({steps_per_sec:.1f} steps/s, {ms_per_step:.2f} ms/step)")
    print(f"  mean p2g:         {np.mean([t['p2g_ms'] for t in step_timings]):.3f} ms")
    print(f"  mean grid_update: {np.mean([t['grid_update_ms'] for t in step_timings]):.3f} ms")
    print(f"  mean g2p:         {np.mean([t['g2p_ms'] for t in step_timings]):.3f} ms")
    print(f"  mean step:        {np.mean([t['step_ms'] for t in step_timings]):.3f} ms")

    # Render GIF to temp file and upload to wandb (skip in benchmark mode)
    gif_path = None
    if not cfg.get('benchmark', False) and frames:
        import tempfile
        gif_path = os.path.join(tempfile.gettempdir(), f"{cfg.tag}_{cfg.kernel.name}.gif")
        print(f"\nRendering animation...")
        visualize_frames(frames, gif_path, size=[1, 1, 1], c=cfg.material.color)
    elif cfg.get('benchmark', False):
        print("\nBenchmark mode: skipping GIF rendering.")

    # Log to wandb (includes animation if rendered)
    log_to_wandb(cfg, step_timings, total_time, gif_path=gif_path)


if __name__ == "__main__":
    main()
