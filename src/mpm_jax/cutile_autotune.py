"""Per-GPU occupancy autotuning for the cuTile P2G kernels.

A hardcoded ``occupancy`` hint is only valid for one GPU -- ncu flags the arena
kernels as register-limited, and the best CTAs/SM depends on the register file and
SM config of each architecture. For a *fair* multi-machine comparison every GPU
must therefore get its own value, not inherit H100's.

This module sweeps the occupancy candidates once per (GPU, kernel), timing the real
JAX-FFI P2G on a small well-resolved problem, picks the fastest, and caches the
result on disk so later runs (and the nsight sweep) reuse it instantly. cuTile's
``kernel.replace_hints(occupancy=...)`` applies the chosen value; ``None`` leaves it
to the compiler. The calibration grid is small and independent of the production
problem because occupancy is a per-SM (register-driven) property, not an N/grid one.
"""

import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

_CACHE_PATH = os.path.expanduser("~/.cache/mpm_cudajax/cutile_occupancy.json")
_CANDIDATES = (None, 4, 6, 8)         # CTAs/SM to try (None = compiler default)
_CAL_GRID = 48                        # small well-resolved calibration grid (% 4 == 0)
_REPEATS = 15


def _gpu_name():
    try:
        return jax.devices()[0].device_kind
    except Exception:  # pragma: no cover - CPU / no device
        return "unknown"


def _load_cache():
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _time_ms(fn):
    out = fn()
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(_REPEATS):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / _REPEATS * 1000.0


def autotuned_kernel(base_kernel, name, make_timing_fn, *, force=False):
    """Return ``base_kernel`` with the best occupancy hint for this GPU.

    ``make_timing_fn(kernel) -> callable`` builds a 0-arg callable that runs one P2G
    with the given kernel (used only for timing). The winning occupancy is cached on
    disk keyed by ``(gpu, name)``; pass ``force=True`` to re-tune and overwrite.
    """
    key = f"{_gpu_name()} :: {name}"
    cache = _load_cache()
    if force or key not in cache:
        times = {}
        for occ in _CANDIDATES:
            kernel = base_kernel if occ is None else base_kernel.replace_hints(occupancy=occ)
            try:
                times[occ] = _time_ms(make_timing_fn(kernel))
            except Exception as exc:  # a candidate may fail to compile
                times[occ] = float("inf")
                print(f"[cutile autotune] {key} occupancy={occ} failed: "
                      f"{type(exc).__name__}")
        best = min(times, key=times.get)
        cache[key] = best
        _save_cache(cache)
        summary = ", ".join(
            f"{('auto' if o is None else o)}={times[o]:.2f}ms" for o in _CANDIDATES)
        print(f"[cutile autotune] {key} -> occupancy={best}  ({summary})")
    best = cache[key]
    return base_kernel if best is None else base_kernel.replace_hints(occupancy=best)


def _arena_timing_inputs(super_cell_width):
    """A small, well-resolved (~8 ppc) sorted problem for timing the arena P2G."""
    from mpm_jax.blocks.sort import home_super_cell_id  # noqa: E402

    g = _CAL_GRID
    n = int((0.8 * g) ** 3 * 8)
    rng = np.random.RandomState(0)
    x = jnp.asarray((rng.rand(n, 3) * 0.8 + 0.1).astype(np.float32))
    v = jnp.zeros((n, 3), jnp.float32)
    cmat = jnp.zeros((n, 3, 3), jnp.float32)
    stress = jnp.asarray((rng.randn(n, 3, 3) * 0.01).astype(np.float32))
    inv_dx = float(g)
    sid = home_super_cell_id(x, inv_dx, g, super_cell_width)
    order = jnp.argsort(sid)
    gs = g // super_cell_width
    cell_start = jnp.searchsorted(
        sid[order], jnp.arange(gs ** 3 + 1, dtype=jnp.int32)).astype(jnp.int32)
    return (x[order], v[order], cmat[order], stress[order], cell_start, g, inv_dx)


def tuned_atomic_tile_kernel(force=False):
    """Occupancy-autotuned ``_p2g_atomic_tile_kernel`` (cutile_v6) for this GPU."""
    from mpm_jax.cutile_p2g import (  # noqa: E402
        ARENA_SC, _p2g_atomic_tile_kernel, cutile_p2g_atomic_tile,
    )

    xs, vs, cs, ss, cell_start, g, inv_dx = _arena_timing_inputs(ARENA_SC)

    def make_timing_fn(kernel):
        return jax.jit(lambda: cutile_p2g_atomic_tile(
            xs, vs, cs, ss, cell_start, g, 5e-5, 1.0, 1.0, inv_dx, 1.0 / g, kernel=kernel))

    return autotuned_kernel(
        _p2g_atomic_tile_kernel, "v6_atomic_tile", make_timing_fn, force=force)
