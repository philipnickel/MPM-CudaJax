from dataclasses import dataclass

import jax.numpy as jnp


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
