"""Grid update: normalize momentum by mass, apply gravity and damping."""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMParams


@jax.jit
def grid_update(
    grid_mv: jnp.ndarray,
    grid_m: jnp.ndarray,
    params: MPMParams,
) -> jnp.ndarray:
    """Normalize momentum by mass, apply gravity and damping.

    Matches solver.py:212-221 logic exactly:
    - Divide momentum by mass where mass > 1e-15
    - Apply damping factor and gravity

    Args:
        grid_mv: (num_grids^3, 3) momentum
        grid_m: (num_grids^3,) mass
        params: MPMParams

    Returns:
        grid_v: (num_grids^3, 3) velocity
    """
    valid = grid_m > 1e-15
    grid_v = jnp.where(valid[:, None], grid_mv / grid_m[:, None], grid_mv)
    grid_v = params.damping * (grid_v + params.dt * params.gravity)
    return grid_v
