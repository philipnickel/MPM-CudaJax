"""Test if warp kernel compiles."""
from mpm_jax.cuda.runtime import CudaRuntime

r = CudaRuntime()
print("Compiling warp kernel...")
k = r.compile_kernel("p2g_scatter_warp.cu", "p2g_scatter_warp")
print("OK:", k)
