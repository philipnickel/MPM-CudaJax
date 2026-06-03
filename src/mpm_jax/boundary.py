import jax.numpy as jnp


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


def build_boundary_fns(bc_configs, grid_x, dx, init_pos=None, dt=None, p_mass=1.0):
    """Build boundary callbacks for the sand benchmark.

    The supported boundary is a sticky `surface_collider`, used as the floor in
    the default and benchmark configs. Extra parameters are accepted for
    compatibility with older call sites but are not used.
    """
    post_grid_fns = []

    for bc in bc_configs:
        bc_type = bc["type"] if isinstance(bc, dict) else bc.type
        bc_dict = dict(bc) if not isinstance(bc, dict) else bc

        if bc_type != "surface_collider":
            raise ValueError(
                f"Unsupported boundary type {bc_type!r}; only "
                "'surface_collider' is supported."
            )
        surface = bc_dict.get("surface", "sticky")
        if surface != "sticky":
            raise ValueError(
                f"Unsupported surface mode {surface!r}; only 'sticky' is supported."
            )
        post_grid_fns.append(_sticky_surface(
            bc_dict["point"], bc_dict["normal"], grid_x, dx,
            bc_dict["start_time"], bc_dict["end_time"],
        ))

    def pre_particle_fn(x, v, time):
        return x, v

    def post_grid_fn(grid_mv, grid_m, time):
        for fn in post_grid_fns:
            grid_mv = fn(grid_mv, grid_m, time)
        return grid_mv

    return pre_particle_fn, post_grid_fn
