import jax.numpy as jnp


def _surface_collider(point, normal, surface, grid_x, dx, start_time, end_time, friction=0.0):
    point = jnp.array(point, dtype=jnp.float32)
    normal = jnp.array(normal, dtype=jnp.float32)
    normal = normal / jnp.linalg.norm(normal)
    offset = grid_x * dx - point
    dotproduct = jnp.sum(offset * normal, axis=1)
    mask = dotproduct < 0.0

    def apply(grid_mv, grid_m, time):
        active = (time >= start_time) & (time < end_time)
        if surface == "sticky":
            new_mv = jnp.where(mask[:, None], 0.0, grid_mv)
        elif surface == "slip":
            proj = jnp.sum(grid_mv * normal, axis=1, keepdims=True)
            new_mv = jnp.where(mask[:, None], grid_mv - normal * proj, grid_mv)
        elif surface == "collide":
            proj = jnp.sum(grid_mv * normal, axis=1, keepdims=True)
            new_mv = jnp.where(mask[:, None], grid_mv - 2.0 * normal * proj, grid_mv)
        else:
            new_mv = grid_mv
        return jnp.where(active, new_mv, grid_mv)

    return apply


def _cuboid(point, size, velocity, grid_x, dx, start_time, end_time, reset, dt):
    point = jnp.array(point, dtype=jnp.float32)
    size = jnp.array(size, dtype=jnp.float32)
    velocity = jnp.array(velocity, dtype=jnp.float32)
    offset = grid_x * dx - point
    mask = jnp.all(jnp.abs(offset) < size, axis=1)

    def apply(grid_mv, grid_m, time):
        in_window = (time >= start_time) & (time <= end_time)
        in_reset = reset & (time > end_time) & (time < end_time + 15.0 * dt)
        new_mv = jnp.where(mask[:, None], velocity, grid_mv)
        reset_mv = jnp.where(mask[:, None], 0.0, grid_mv)
        result = jnp.where(in_window, new_mv, grid_mv)
        result = jnp.where(in_reset & ~in_window, reset_mv, result)
        return result

    return apply


def _sdf_collider(bound, dim, start_time, end_time, dt):
    sdf_grad = jnp.zeros(3).at[dim].set(1.0)

    def apply(x, v, time):
        active = (time >= start_time) & (time < end_time)
        sdf_lo = x[:, dim] - bound
        below = sdf_lo < 0
        x_lo = x - below[:, None] * sdf_lo[:, None] * sdf_grad
        v_lo = v - below[:, None] * sdf_lo[:, None] * sdf_grad / dt

        up_bound = 1.0 - bound
        sdf_hi = x_lo[:, dim] - up_bound
        above = sdf_hi > 0
        x_hi = x_lo - above[:, None] * sdf_hi[:, None] * sdf_grad
        v_hi = v_lo - above[:, None] * sdf_hi[:, None] * sdf_grad / dt

        return (
            jnp.where(active, x_hi, x),
            jnp.where(active, v_hi, v),
        )

    return apply


def _particle_impulse(force, point, size, init_pos, dt, num_dt, start_time, p_mass):
    point = jnp.array(point, dtype=jnp.float32)
    size = jnp.array(size, dtype=jnp.float32)
    force = jnp.array(force, dtype=jnp.float32)
    offset = init_pos - point
    mask = jnp.all(jnp.abs(offset) < size, axis=1)
    end_time = start_time + num_dt * dt

    def apply(x, v, time):
        active = (time >= start_time) & (time < end_time)
        dv = force / p_mass * dt
        new_v = jnp.where(mask[:, None], v + dv, v)
        return x, jnp.where(active, new_v, v)

    return apply


def _enforce_particle_translation(point, size, velocity, init_pos, start_time, end_time):
    point = jnp.array(point, dtype=jnp.float32)
    size = jnp.array(size, dtype=jnp.float32)
    velocity = jnp.array(velocity, dtype=jnp.float32)
    offset = init_pos - point
    mask = jnp.all(jnp.abs(offset) < size, axis=1)

    def apply(x, v, time):
        active = (time >= start_time) & (time < end_time)
        new_v = jnp.where(mask[:, None], velocity, v)
        return x, jnp.where(active, new_v, v)

    return apply


def _enforce_particle_velocity_rotation(
    point, normal, half_height_and_radius, rotation_scale, translation_scale,
    init_pos, start_time, end_time,
):
    point = jnp.array(point, dtype=jnp.float32)
    normal = jnp.array(normal, dtype=jnp.float32)
    normal = normal / jnp.linalg.norm(normal)
    hhr = jnp.array(half_height_and_radius, dtype=jnp.float32)

    h1 = jnp.array([1.0, 1.0, 1.0])
    h1 = jnp.where(jnp.abs(jnp.dot(normal, h1)) < 0.01, jnp.array([0.72, 0.37, -0.67]), h1)
    h1 = h1 - jnp.dot(h1, normal) * normal
    h1 = h1 / jnp.linalg.norm(h1)
    h2 = jnp.cross(h1, normal)

    offset = init_pos - point
    vert_dist = jnp.abs(jnp.dot(offset, normal))
    horiz_offset = offset - jnp.outer(jnp.dot(offset, normal), normal)
    horiz_dist = jnp.linalg.norm(horiz_offset, axis=-1)
    mask = (vert_dist < hhr[0]) & (horiz_dist < hhr[1])

    def apply(x, v, time):
        active = (time >= start_time) & (time < end_time)
        off = x - point
        h_off = off - jnp.outer(jnp.dot(off, normal), normal)
        h_dist = jnp.clip(jnp.linalg.norm(h_off, axis=-1, keepdims=True), 1e-10)
        cos_theta = jnp.dot(off, h1).reshape(-1, 1) / h_dist
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
        theta = jnp.arccos(cos_theta)
        sign = jnp.where(jnp.dot(off, h2).reshape(-1, 1) > 0, 1.0, -1.0)
        theta = theta * sign
        a1 = -h_dist * jnp.sin(theta) * rotation_scale
        a2 = h_dist * jnp.cos(theta) * rotation_scale
        new_v = a1 * h1 + a2 * h2 + translation_scale * normal
        new_v = jnp.where(mask[:, None], new_v, v)
        return x, jnp.where(active, new_v, v)

    return apply


def build_boundary_fns(bc_configs, grid_x, dx, init_pos, dt, p_mass=1.0):
    post_grid_fns = []
    pre_particle_fns = []

    for bc in bc_configs:
        bc_type = bc["type"] if isinstance(bc, dict) else bc.type
        bc_dict = dict(bc) if not isinstance(bc, dict) else bc

        if bc_type == "surface_collider":
            post_grid_fns.append(_surface_collider(
                bc_dict["point"], bc_dict["normal"], bc_dict["surface"],
                grid_x, dx, bc_dict["start_time"], bc_dict["end_time"],
                bc_dict.get("friction", 0.0),
            ))
        elif bc_type == "cuboid":
            post_grid_fns.append(_cuboid(
                bc_dict["point"], bc_dict["size"], bc_dict["velocity"],
                grid_x, dx, bc_dict["start_time"], bc_dict["end_time"],
                bc_dict.get("reset", False), dt,
            ))
        elif bc_type == "sdf_collider":
            pre_particle_fns.append(_sdf_collider(
                bc_dict["bound"], bc_dict["dim"],
                bc_dict["start_time"], bc_dict["end_time"], dt,
            ))
        elif bc_type == "particle_impulse":
            pre_particle_fns.append(_particle_impulse(
                bc_dict["force"], bc_dict.get("point", [1, 1, 1]),
                bc_dict.get("size", [1, 1, 1]), init_pos, dt,
                bc_dict.get("num_dt", 1), bc_dict.get("start_time", 0.0), p_mass,
            ))
        elif bc_type == "enforce_particle_translation":
            pre_particle_fns.append(_enforce_particle_translation(
                bc_dict["point"], bc_dict["size"], bc_dict["velocity"],
                init_pos, bc_dict["start_time"], bc_dict["end_time"],
            ))
        elif bc_type == "enforce_particle_velocity_rotation":
            pre_particle_fns.append(_enforce_particle_velocity_rotation(
                bc_dict["point"], bc_dict["normal"],
                bc_dict["half_height_and_radius"],
                bc_dict["rotation_scale"], bc_dict["translation_scale"],
                init_pos, bc_dict["start_time"], bc_dict["end_time"],
            ))

    def pre_particle_fn(x, v, time):
        for fn in pre_particle_fns:
            x, v = fn(x, v, time)
        return x, v

    def post_grid_fn(grid_mv, grid_m, time):
        for fn in post_grid_fns:
            grid_mv = fn(grid_mv, grid_m, time)
        return grid_mv

    return pre_particle_fn, post_grid_fn
