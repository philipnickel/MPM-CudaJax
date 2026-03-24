"""Pluggable P2G implementations."""
from mpm_jax.p2g.jax import make_jax_p2g


def get_p2g_fn(cfg, runtime=None):
    """Return a P2G function based on config."""
    kernel_name = cfg.kernel.name
    if kernel_name == "jax":
        return make_jax_p2g(cfg)
    elif kernel_name == "cuda_naive":
        from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
        return make_cuda_naive_p2g(cfg, runtime)
    elif kernel_name == "cuda_warp":
        from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
        return make_cuda_warp_p2g(cfg, runtime)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
