from dataclasses import dataclass, field
from typing import Callable

from mpm_jax.solver import MPMSolver, WarpGraphSolver
from mpm_jax.stepping.jax_frames import build_jax_frame, build_jax_v1_5_frame
from mpm_jax.stepping.cuda_frames import (
    build_cuda_v1_frame, build_cuda_v2_frame, build_cuda_v3_frame, build_cuda_v4_frame,
)
from mpm_jax.stepping.warp_frames import (
    build_warp_v1_frame, build_warp_v2_tile_frame, build_warp_v3_frame,
)
from mpm_jax.stepping.warp_graph_frame import build_warp_graph


@dataclass(frozen=True)
class KernelSpec:
    solver_cls: type
    build_frame: Callable
    defaults: dict = field(default_factory=dict)


KERNELS = {
    "jax":                    KernelSpec(MPMSolver, build_jax_frame),
    "jax_v1_5":               KernelSpec(MPMSolver, build_jax_v1_5_frame),
    "cuda_v1_inline":         KernelSpec(MPMSolver, build_cuda_v1_frame),
    "cuda_v2_inline":         KernelSpec(MPMSolver, build_cuda_v2_frame, {"loop_kind": "fori"}),
    "cuda_v3_inline":         KernelSpec(MPMSolver, build_cuda_v3_frame, {"loop_kind": "fori", "cuda_graph": False}),
    "cuda_v4_inline":         KernelSpec(MPMSolver, build_cuda_v4_frame),
    "warp_v1_inline":         KernelSpec(MPMSolver, build_warp_v1_frame),
    "warp_v2_tile":           KernelSpec(MPMSolver, build_warp_v2_tile_frame),
    "warp_v3_supercell_tile": KernelSpec(MPMSolver, build_warp_v3_frame),
    "warp_bonus_graph":       KernelSpec(WarpGraphSolver, build_warp_graph),
    "warp_bonus_v2_graph":    KernelSpec(WarpGraphSolver, build_warp_graph, {"indexed_sort": True}),
}

REMOVED_KERNELS = {
    "cuda_v1": "Use cuda_v1_inline (scatter-only variant removed).",
    "cuda_v2": "Use cuda_v2_inline.",
    "cuda_v4": "Use cuda_v4_inline.",
    "cuda_fused": "Deprecated; use an inline kernel and profile=jax.",
    "cuda_v2_fori_inline": "Use kernel=cuda_v2_inline with loop_kind=fori.",
    "cuda_v3_fori_inline": "Use kernel=cuda_v3_inline with loop_kind=fori.",
    "cuda_v6_inline": "Use kernel=cuda_v3_inline with cuda_graph=true.",
}
