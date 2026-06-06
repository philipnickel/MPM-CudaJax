import jax.numpy as jnp

_STICKY_SURFACE_KEYS = {"point", "normal", "start_time", "end_time"}


def _sticky_surface(point, normal, grid_x, dx, start_time, end_time):
    point = jnp.array(point, dtype=jnp.float32)
    normal = jnp.array(normal, dtype=jnp.float32)
    normal = normal / jnp.linalg.norm(normal)
    signed_distance = jnp.sum((grid_x * dx - point) * normal, axis=1)
    masked = signed_distance < 0.0

    def apply(grid_mv, grid_m, time):
        active = (time >= start_time) & (time < end_time)
        stuck_mv = jnp.where(masked[:, None], 0.0, grid_mv)
        return jnp.where(active, stuck_mv, grid_mv)

    return apply


def _sticky_surface_from_config(bc, grid_x, dx):
    keys = set(bc.keys())
    missing = _STICKY_SURFACE_KEYS - keys
    extra = keys - _STICKY_SURFACE_KEYS
    if missing or extra:
        expected = ", ".join(sorted(_STICKY_SURFACE_KEYS))
        problems = []
        if missing:
            problems.append("missing " + ", ".join(sorted(missing)))
        if extra:
            problems.append("unexpected " + ", ".join(sorted(extra)))
        raise ValueError(
            "Sticky boundary configs must contain exactly "
            f"{expected}; got {'; '.join(problems)}."
        )
    return _sticky_surface(
        bc["point"],
        bc["normal"],
        grid_x,
        dx,
        bc["start_time"],
        bc["end_time"],
    )


def build_boundary_fns(bc_configs, grid_x, dx):
    """Build sticky plane boundary callbacks for the benchmark."""
    post_grid_fns = [
        _sticky_surface_from_config(bc, grid_x, dx) for bc in bc_configs
    ]

    def pre_particle_fn(x, v, time):
        return x, v

    def post_grid_fn(grid_mv, grid_m, time):
        for fn in post_grid_fns:
            grid_mv = fn(grid_mv, grid_m, time)
        return grid_mv

    return pre_particle_fn, post_grid_fn
