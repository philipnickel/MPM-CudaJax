"""Offline cuTile P2G tuner.

This intentionally avoids Hydra. It times the production JAX/cuTile call path
for a small finite set of tile sizes and compiler hints, then writes the
ranking as JSON so the winning constants can be baked into the kernel modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.p2g.backends.common import home_cell_order
from mpm_jax.p2g.cutile.v1 import (
    PARTICLES_PER_TILE as V1_PARTICLES_PER_TILE,
    cutile_p2g_v1,
)
from mpm_jax.p2g.cutile.v3 import (
    PARTICLES_PER_TILE as V3_PARTICLES_PER_TILE,
    cutile_p2g_v3,
)
from mpm_jax.types import MPMParams, MPMState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuneConfig:
    backend: str
    particles_per_tile: int
    num_ctas: int | None
    occupancy: int | None

    def hints(self):
        return {
            key: value
            for key, value in {
                "num_ctas": self.num_ctas,
                "occupancy": self.occupancy,
            }.items()
            if value is not None
        }


def _parse_ints(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parse_optional_ints(value):
    parsed = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.append(None if item in {"auto", "none"} else int(item))
    return tuple(parsed)


def _sample_particles(n_particles, center, size):
    start = np.asarray(center, dtype=np.float32) - np.asarray(size, dtype=np.float32) / 2
    end = np.asarray(center, dtype=np.float32) + np.asarray(size, dtype=np.float32) / 2
    rng = np.random.RandomState(42)
    return start + rng.rand(n_particles, 3).astype(np.float32) * (end - start)


def _sim_config(args):
    n_particles = args.n_particles
    if n_particles is None:
        n_particles = 8 * int(args.num_grids) ** 3
    return SimpleNamespace(
        n_particles=int(n_particles),
        num_grids=int(args.num_grids),
        dt=float(args.dt),
        gravity=(0.0, 0.0, -9.8),
        rho=1000.0,
        clip_bound=0.5,
        damping=1.0,
        center=(0.5, 0.5, 0.5),
        size=(0.8, 0.8, 0.8),
    )


def _make_state(sim):
    n = int(sim.n_particles)
    x = jnp.asarray(
        _sample_particles(n, center=sim.center, size=sim.size), dtype=jnp.float32
    )
    v = jnp.zeros((n, 3), dtype=jnp.float32)
    C = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    F = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n, 1, 1))
    stress = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    jax.block_until_ready(x)
    return MPMState(x=x, v=v, C=C, F=F), stress


def _ready(result):
    grid_mv, grid_m = result
    grid_mv.block_until_ready()
    grid_m.block_until_ready()


def _prepare_v3(params, state, stress):
    prepared = home_cell_order(params, state, stress)
    prepared.x.block_until_ready()
    prepared.v.block_until_ready()
    prepared.C.block_until_ready()
    prepared.stress.block_until_ready()
    prepared.bucket_bounds.block_until_ready()
    return prepared


def _runner_v1(params, cfg):
    @jax.jit
    def run(x, v, C, stress):
        return cutile_p2g_v1(
            x,
            v,
            C,
            stress,
            params.num_grids,
            params.dt,
            params.vol,
            params.p_mass,
            params.inv_dx,
            params.dx,
            particles_per_tile=cfg.particles_per_tile,
            kernel_hints=cfg.hints(),
        )

    return run


def _runner_v3(params, cfg):
    @jax.jit
    def run(x, v, C, stress, cell_bounds):
        return cutile_p2g_v3(
            x,
            v,
            C,
            stress,
            cell_bounds,
            params.num_grids,
            params.dt,
            params.vol,
            params.p_mass,
            params.inv_dx,
            params.dx,
            particles_per_tile=cfg.particles_per_tile,
            kernel_hints=cfg.hints(),
        )

    return run


def _time_config(cfg, params, inputs, warmups, repeats):
    runner = _runner_v1(params, cfg) if cfg.backend == "v1" else _runner_v3(params, cfg)
    for _ in range(warmups):
        _ready(runner(*inputs))

    start = time.perf_counter()
    for _ in range(repeats):
        _ready(runner(*inputs))
    mean_s = (time.perf_counter() - start) / repeats
    return mean_s


def _configs(args):
    yielded = set()

    def unique(cfg):
        if cfg in yielded:
            return None
        yielded.add(cfg)
        return cfg

    for backend in args.backends.split(","):
        backend = backend.strip()
        if not backend:
            continue
        if backend not in {"v1", "v3"}:
            raise ValueError(f"Unknown backend {backend!r}; expected v1 or v3.")
        for particles_per_tile in _parse_ints(args.particle_tiles):
            for num_ctas in _parse_optional_ints(args.num_ctas):
                for occupancy in _parse_optional_ints(args.occupancy):
                    cfg = TuneConfig(
                        backend=backend,
                        particles_per_tile=particles_per_tile,
                        num_ctas=num_ctas,
                        occupancy=occupancy,
                    )
                    if unique(cfg) is not None:
                        yield cfg

        default_tile = (
            V1_PARTICLES_PER_TILE if backend == "v1" else V3_PARTICLES_PER_TILE
        )
        cfg = TuneConfig(
            backend=backend,
            particles_per_tile=default_tile,
            num_ctas=None,
            occupancy=None,
        )
        if unique(cfg) is not None:
            yield cfg


def _inputs_for_backend(backend, params, state, stress, prepared_v3):
    if backend == "v1":
        return state.x, state.v, state.C, stress
    return (
        prepared_v3.x,
        prepared_v3.v,
        prepared_v3.C,
        prepared_v3.stress,
        prepared_v3.bucket_bounds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="v1,v3")
    parser.add_argument("--num-grids", type=int, default=50)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--dt", type=float, default=5.0e-5)
    parser.add_argument("--particle-tiles", default="4,8,16,32")
    parser.add_argument("--num-ctas", default="auto")
    parser.add_argument("--occupancy", default="auto")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cutile_tune/results.json")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sim = _sim_config(args)
    params = MPMParams(sim)
    logger.info(
        "Preparing %d particles on a %d^3 grid", sim.n_particles, sim.num_grids
    )
    state, stress = _make_state(sim)
    prepared_v3 = None
    if "v3" in {item.strip() for item in args.backends.split(",")}:
        logger.info("Preparing home-cell sorted v3 inputs")
        prepared_v3 = _prepare_v3(params, state, stress)

    results = []
    for cfg in _configs(args):
        default = (
            V1_PARTICLES_PER_TILE if cfg.backend == "v1" else V3_PARTICLES_PER_TILE
        )
        inputs = _inputs_for_backend(cfg.backend, params, state, stress, prepared_v3)
        label = {
            "backend": f"cutile_{cfg.backend}",
            "particles_per_tile": cfg.particles_per_tile,
            "hints": cfg.hints() or "auto",
            "default": cfg.particles_per_tile == default and not cfg.hints(),
        }
        try:
            mean_s = _time_config(cfg, params, inputs, args.warmups, args.repeats)
            record = {
                **label,
                "mean_ms": mean_s * 1000,
                "p2g_particles_per_sec": sim.n_particles / mean_s,
                "status": "ok",
            }
            logger.info(
                "%s tile=%d hints=%s %.3f ms (%.3e particles/s)%s",
                label["backend"],
                cfg.particles_per_tile,
                label["hints"],
                record["mean_ms"],
                record["p2g_particles_per_sec"],
                " default" if label["default"] else "",
            )
        except Exception as exc:
            record = {**label, "status": "failed", "error": str(exc)}
            logger.exception(
                "%s tile=%d hints=%s failed",
                label["backend"],
                cfg.particles_per_tile,
                label["hints"],
            )
        results.append(record)

    results.sort(key=lambda row: row.get("mean_ms", float("inf")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_grids": sim.num_grids,
        "n_particles": sim.n_particles,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if results and results[0]["status"] == "ok":
        best = results[0]
        logger.info(
            "Best: %s tile=%d hints=%s %.3f ms",
            best["backend"],
            best["particles_per_tile"],
            best["hints"],
            best["mean_ms"],
        )
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
