"""Profile JAX MPM step to identify host-device transfers and bottlenecks.

Generates a trace viewable in chrome://tracing or Perfetto UI.
"""
import jax
import jax.numpy as jnp
import time
from omegaconf import OmegaConf
from mpm_jax.solver import MPMState, make_params, step
from mpm_jax.constitutive import get_constitutive

print(f"Device: {jax.devices()[0]}")
print(f"Backend: {jax.default_backend()}")

n = 2000
params = make_params(n_particles=n, num_grids=64)
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)
efn = get_constitutive(OmegaConf.create({"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4}))
pfn = get_constitutive(OmegaConf.create({"name": "IdentityPlasticity"}))
noop = lambda x, v, t: (x, v)
noop2 = lambda gv, gm, t: gv

# Warmup
print("Warming up JIT...")
for _ in range(5):
    state = step(params, state, efn(state.F), noop, noop2, 0.0)
    state = state._replace(F=pfn(state.F))
jax.block_until_ready(state.x)
print("Warmup done.")

# Check where arrays live
print(f"\nArray devices:")
print(f"  state.x: {state.x.devices()}")
print(f"  state.F: {state.F.devices()}")
print(f"  params.gravity: {params.gravity.devices()}")

# Dump XLA HLO for one step to see what XLA compiles
print("\n--- HLO summary (first 50 lines) ---")
stress = efn(state.F)
# We can't easily lower step() since it's not jitted as a whole,
# but we can check the individual pieces
from mpm_jax.solver import compute_weights_and_indices, p2g_compute, p2g_scatter, grid_update, g2p

lowered = jax.jit(lambda s: efn(s)).lower(state.F)
hlo = lowered.as_text()
n_lines = len(hlo.splitlines())
print(f"Stress HLO: {n_lines} lines")

# Profile with JAX profiler
print("\n--- Profiling 20 steps ---")
trace_dir = "/tmp/jax_mpm_trace"
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)

jax.profiler.start_trace(trace_dir)
for i in range(20):
    stress = efn(state.F)
    state = step(params, state, stress, noop, noop2, 0.0)
    state = state._replace(F=pfn(state.F))
jax.block_until_ready(state.x)
jax.profiler.stop_trace()
print(f"Trace saved to {trace_dir}/")
print("View with: tensorboard --logdir /tmp/jax_mpm_trace")

# Also time individual pieces with sync between each
print("\n--- Per-operation timing (with sync, 10 steps) ---")
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)

timings = {}
for _ in range(10):
    # stress
    t0 = time.perf_counter()
    stress = efn(state.F)
    jax.block_until_ready(stress)
    timings.setdefault("stress", []).append(time.perf_counter() - t0)

    # pre-BC
    t0 = time.perf_counter()
    x, v = noop(state.x, state.v, 0.0)
    timings.setdefault("pre_bc", []).append(time.perf_counter() - t0)

    # weights
    t0 = time.perf_counter()
    w, dw, dpos, idx = compute_weights_and_indices(x, params.inv_dx, params.dx, params.num_grids)
    jax.block_until_ready(w)
    timings.setdefault("weights", []).append(time.perf_counter() - t0)

    # p2g compute
    t0 = time.perf_counter()
    mv, m = p2g_compute(v, state.C, stress, w, dw, dpos, params.dt, params.vol, params.p_mass)
    jax.block_until_ready(mv)
    timings.setdefault("p2g_compute", []).append(time.perf_counter() - t0)

    # p2g scatter
    t0 = time.perf_counter()
    grid_mv, grid_m = p2g_scatter(mv, m, idx, params.num_grids)
    jax.block_until_ready(grid_mv)
    timings.setdefault("p2g_scatter", []).append(time.perf_counter() - t0)

    # grid update
    t0 = time.perf_counter()
    grid_mv = grid_update(grid_mv, grid_m, params.gravity, params.dt, params.damping)
    jax.block_until_ready(grid_mv)
    timings.setdefault("grid_update", []).append(time.perf_counter() - t0)

    # g2p
    t0 = time.perf_counter()
    new_x, new_v, new_C, new_F = g2p(grid_mv, w, dw, dpos, idx, state.F, x, params.dt, params.inv_dx, params.clip_bound)
    jax.block_until_ready(new_x)
    timings.setdefault("g2p", []).append(time.perf_counter() - t0)

    # plasticity
    t0 = time.perf_counter()
    new_F = pfn(new_F)
    jax.block_until_ready(new_F)
    timings.setdefault("plasticity", []).append(time.perf_counter() - t0)

    state = MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

import numpy as np
total = sum(np.mean(v) for v in timings.values())
print(f"{'stage':20s} {'mean_ms':>10s} {'std_ms':>10s} {'pct':>8s}")
for name, vals in sorted(timings.items(), key=lambda x: -np.mean(x[1])):
    mean = np.mean(vals) * 1000
    std = np.std(vals) * 1000
    pct = np.mean(vals) / total * 100
    print(f"{name:20s} {mean:10.3f} {std:10.3f} {pct:7.1f}%")
print(f"{'TOTAL':20s} {total*1000:10.3f}")
