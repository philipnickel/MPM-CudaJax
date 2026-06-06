from dataclasses import dataclass

import jax.numpy as jnp


def _identity_particles(x, v, time):
    return x, v


@dataclass(frozen=True)
class StickyPlane:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    start_time: float
    end_time: float

    def bind_grid(self, grid_x, dx):
        point = jnp.asarray(self.point, dtype=jnp.float32)
        normal = jnp.asarray(self.normal, dtype=jnp.float32)
        normal = normal / jnp.linalg.norm(normal)
        blocked = jnp.sum((grid_x * dx - point) * normal, axis=1) < 0.0

        def post_grid(grid_mv, grid_m, time):
            active = (time >= self.start_time) & (time < self.end_time)
            return jnp.where(active & blocked[:, None], 0.0, grid_mv)

        return post_grid


def bind_boundaries(boundaries, grid_x, dx):
    """Bind configured sticky planes to the current grid."""
    post_grid_fns = [boundary.bind_grid(grid_x, dx) for boundary in boundaries]

    def post_grid_fn(grid_mv, grid_m, time):
        for fn in post_grid_fns:
            grid_mv = fn(grid_mv, grid_m, time)
        return grid_mv

    return _identity_particles, post_grid_fn
