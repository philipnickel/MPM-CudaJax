from mpm_jax.registry import KERNELS, REMOVED_KERNELS, KernelSpec


def test_every_kernel_has_a_spec():
    expected = {
        "jax", "jax_v1_5",
        "cuda_v1_inline", "cuda_v2_inline", "cuda_v3_inline", "cuda_v4_inline",
        "warp_v1_inline", "warp_v2_tile", "warp_v3_supercell_tile",
        "warp_bonus_graph", "warp_bonus_v2_graph",
    }
    assert set(KERNELS) == expected
    for spec in KERNELS.values():
        assert isinstance(spec, KernelSpec)
        assert spec.solver_cls is not None
        assert callable(spec.build_frame)


def test_removed_kernels_listed():
    for name in ("cuda_v1", "cuda_v2", "cuda_v4", "cuda_fused",
                 "cuda_v2_fori_inline", "cuda_v3_fori_inline", "cuda_v6_inline"):
        assert name in REMOVED_KERNELS
