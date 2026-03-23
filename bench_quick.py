"""Quick GPU throughput benchmark — full timestep JIT-compiled via lax.scan."""
import jax
import jax.numpy as jnp
import time
from omegaconf import OmegaConf
from mpm_jax.solver import MPMState, make_params, build_jit_step, build_jit_frame
from mpm_jax.constitutive import get_constitutive

print(f"Device: {jax.devices()[0]}")
print(f"Backend: {jax.default_backend()}")

n = 2000
G = 64
params = make_params(n_particles=n, num_grids=G)
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

print(f"N={n}, G={G}")

# Build JIT-compiled step and frame
jit_step = build_jit_step(params, efn, pfn, noop, noop2)
jit_frame = build_jit_frame(params, efn, pfn, noop, noop2, steps_per_frame=10)

# Warmup (triggers JIT compilation)
print("Compiling...")
state = jit_step(state)
jax.block_until_ready(state.x)

state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)
state = jit_frame(state)
jax.block_until_ready(state.x)
print("Compilation done.")

# Benchmark single steps
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)
t0 = time.perf_counter()
for _ in range(100):
    state = jit_step(state)
jax.block_until_ready(state.x)
e = time.perf_counter() - t0
print(f"\njit_step:  100 steps in {e:.3f}s ({100/e:.1f} steps/s, {e/100*1000:.2f} ms/step)")

# Benchmark lax.scan frames (10 substeps each)
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)
t0 = time.perf_counter()
for _ in range(10):
    state = jit_frame(state)
jax.block_until_ready(state.x)
e = time.perf_counter() - t0
print(f"jit_frame: 100 steps in {e:.3f}s ({100/e:.1f} steps/s, {e/100*1000:.2f} ms/step)")
