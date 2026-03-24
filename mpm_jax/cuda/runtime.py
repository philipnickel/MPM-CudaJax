"""CUDA runtime: kernel compilation and launch via cuda.core."""
from pathlib import Path

KERNELS_DIR = Path(__file__).parent / "kernels"


def get_ptr(arr) -> int:
    """Extract raw GPU device pointer from a JAX array."""
    return arr.__cuda_array_interface__['data'][0]


class CudaRuntime:
    """Manages cuda.core device, stream, and kernel compilation cache."""

    def __init__(self):
        try:
            from cuda.core import Device
        except ImportError:
            raise ImportError(
                "cuda-core is required for CUDA kernels. "
                "Install with: pip install cuda-core"
            )

        self.dev = Device()
        self.dev.set_current()
        self.stream = self.dev.create_stream()
        self._cache = {}

    def compile_kernel(self, source_path: str, kernel_name: str):
        """Compile a .cu file and extract a kernel by name.

        Args:
            source_path: Path to .cu file, relative to kernels/ directory
            kernel_name: Name of the __global__ function to extract

        Returns:
            Compiled kernel handle
        """
        if kernel_name in self._cache:
            return self._cache[kernel_name]

        from cuda.core import Program, ProgramOptions

        cu_path = KERNELS_DIR / source_path
        if not cu_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {cu_path}")

        code = cu_path.read_text()
        try:
            prog = Program(code, code_type="c++", options=ProgramOptions(
                std="c++17",
                arch=f"sm_{self.dev.arch}",
            ))
            mod = prog.compile("cubin")
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile {cu_path}: {e}"
            ) from e

        kernel = mod.get_kernel(kernel_name)
        self._cache[kernel_name] = kernel
        return kernel

    def launch(self, kernel, grid, block, *args):
        """Launch a kernel on the runtime's stream and synchronize.

        Args:
            kernel: Compiled kernel handle
            grid: Grid dimensions (int or tuple)
            block: Block dimensions (int or tuple)
            *args: Kernel arguments (ints for pointers, scalars for values)
        """
        from cuda.core import LaunchConfig, launch

        config = LaunchConfig(grid=grid, block=block)
        launch(self.stream, config, kernel, *args)
        self.stream.sync()
