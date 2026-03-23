import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import hydra
from omegaconf import DictConfig


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
# Backend-specific runners
# ---------------------------------------------------------------------------

def run_jax(cfg: DictConfig):
    import jax
    import jax.numpy as jnp
    from mpm_jax.solver import MPMState, make_params, step, simulate_frame
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.boundary import build_boundary_fns

    sim = cfg.sim
    mat = cfg.material

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
        state = step(params, state, stress, pre_fn, post_fn, 0.0)
        state = state._replace(F=plasticity_fn(state.F))
    jax.block_until_ready(state.x)

    # Reset
    state = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    frames = []
    sim_time = 0.0
    t0 = time.perf_counter()
    for frame in tqdm(range(sim.num_frames), desc='JAX'):
        frames.append(np.array(state.x))
        state, sim_time = simulate_frame(
            params, state, elasticity_fn, plasticity_fn,
            pre_fn, post_fn, sim.steps_per_frame, sim_time,
        )
    jax.block_until_ready(state.x)
    elapsed = time.perf_counter() - t0

    total_steps = sim.num_frames * sim.steps_per_frame
    print(f"JAX: {total_steps} steps in {elapsed:.2f}s ({total_steps/elapsed:.1f} steps/s, {elapsed/total_steps*1000:.2f} ms/step)")
    return frames


def run_pytorch(cfg: DictConfig):
    import sys
    # Add the vendored MPM-PyTorch to the path
    vendor_path = os.path.join(os.path.dirname(__file__), "vendor", "MPM-PyTorch")
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    import torch
    from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive

    sim = cfg.sim
    mat = cfg.material

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

    frames = []
    t0 = time.perf_counter()
    for frame in tqdm(range(sim.num_frames), desc='PyTorch'):
        frames.append(x.cpu().numpy())
        for _ in range(sim.steps_per_frame):
            stress = elasticity(F)
            x, v, C, F = solver(x, v, C, F, stress)
            F = plasticity(F)
    elapsed = time.perf_counter() - t0

    total_steps = sim.num_frames * sim.steps_per_frame
    print(f"PyTorch: {total_steps} steps in {elapsed:.2f}s ({total_steps/elapsed:.1f} steps/s, {elapsed/total_steps*1000:.2f} ms/step)")
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    backend = cfg.backend.name
    print(f"Backend: {backend}")

    if backend == "jax":
        frames = run_jax(cfg)
    elif backend == "pytorch":
        frames = run_pytorch(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    orig_cwd = hydra.utils.get_original_cwd()
    output_dir = os.path.join(orig_cwd, cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, f"{cfg.tag}_{backend}.gif")
    print(f"Rendering to {export_path}...")
    visualize_frames(frames, export_path, size=[1, 1, 1], c=cfg.material.color)


if __name__ == "__main__":
    main()
