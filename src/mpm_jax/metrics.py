from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunConfig:
    kernel: str
    material_elasticity: str
    n_particles: int
    num_grids: int
    num_frames: int
    steps_per_frame: int
    render_enabled: bool

    @classmethod
    def from_hydra(cls, cfg: Any, choices: dict[str, str]) -> RunConfig:
        sim = cfg.sim
        render_cfg = cfg.get("render", {})
        return cls(
            kernel=str(choices.get("backend", "jax")),
            material_elasticity=str(choices.get("material", "unknown")),
            n_particles=int(sim.n_particles),
            num_grids=int(sim.num_grids),
            num_frames=int(sim.num_frames),
            steps_per_frame=int(sim.steps_per_frame),
            render_enabled=bool(render_cfg.get("enabled", True)),
        )

    @property
    def total_steps(self) -> int:
        return self.num_frames * self.steps_per_frame


@dataclass(frozen=True)
class RunMetrics:
    config: RunConfig
    elapsed_s: float
    render_path: str | None = None

    @property
    def total_steps(self) -> int:
        return self.config.total_steps

    @property
    def steps_per_sec(self) -> float:
        return self.total_steps / self.elapsed_s

    @property
    def ms_per_step(self) -> float:
        return self.elapsed_s / self.total_steps * 1000

    @property
    def ms_per_frame(self) -> float:
        return self.elapsed_s / self.config.num_frames * 1000

    def to_record(self) -> dict[str, Any]:
        return {
            "kernel": self.config.kernel,
            "material_elasticity": self.config.material_elasticity,
            "n_particles": self.config.n_particles,
            "num_grids": self.config.num_grids,
            "num_frames": self.config.num_frames,
            "steps_per_frame": self.config.steps_per_frame,
            "render_enabled": self.config.render_enabled,
            "total_steps": int(self.total_steps),
            "elapsed_s": float(self.elapsed_s),
            "ms_per_step": float(self.ms_per_step),
            "steps_per_sec": float(self.steps_per_sec),
            "render_path": self.render_path,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_record(), indent=2))


def metrics_dataframe(result_files: Iterable[str | Path]):
    import pandas as pd

    records = [json.loads(Path(path).read_text()) for path in result_files]
    return pd.DataFrame.from_records(records)
