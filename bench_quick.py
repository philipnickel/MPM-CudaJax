"""Quick GPU throughput benchmark — no per-stage sync overhead."""
import jax
import jax.numpy as jnp
import time
from mpm_jax.solver import MPMState, make_params, step
from mpm_jax.constitutive import get_constitutive

n = 2000
params = make_params(n_particles=n, num_grids=64)
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)
efn = get_constitutive({"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4})
pfn = get_constitutive({"name": "IdentityPlasticity"})
noop = lambda x, v, t: (x, v)
noop2 = lambda gv, gm, t: gv

print(f"Device: {jax.devices()[0]}")
print(f"N={n}, G=64")

# Warmup
for _ in range(5):
    state = step(params, state, efn(state.F), noop, noop2, 0.0)
    state = state._replace(F=pfn(state.F))
jax.block_until_ready(state.x)

# Reset
state = MPMState(
    x=jax.random.uniform(jax.random.PRNGKey(0), (n, 3), minval=0.2, maxval=0.8),
    v=jnp.zeros((n, 3)),
    C=jnp.zeros((n, 3, 3)),
    F=jnp.tile(jnp.eye(3), (n, 1, 1)),
)

# Time 100 steps, single sync at the end
t0 = time.perf_counter()
for _ in range(100):
    state = step(params, state, efn(state.F), noop, noop2, 0.0)
    state = state._replace(F=pfn(state.F))
jax.block_until_ready(state.x)
e = time.perf_counter() - t0
print(f"100 steps in {e:.3f}s ({100/e:.1f} steps/s, {e/100*1000:.2f} ms/step)")
