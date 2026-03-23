# Build CUDA FFI kernels for JAX integration.
#
# Usage:
#   make                    # build all kernels
#   make clean              # remove .so files
#   make FFI_INC=<path>     # override XLA FFI include path
#
# Requires: nvcc, jax (for include path)

NVCC       := nvcc
ARCH       := sm_90
NVCC_FLAGS := -arch=$(ARCH) -O3 --use_fast_math -std=c++17 -Xcompiler -fPIC -shared

# XLA FFI headers — auto-detect from jax install
FFI_INC ?= $(shell python -c "import jax; print(jax.extend.ffi.include_dir())" 2>/dev/null)

ifeq ($(FFI_INC),)
$(error Cannot find JAX FFI include directory. Set FFI_INC manually or ensure jax is installed.)
endif

INCLUDES := -I$(FFI_INC)

# Targets
TARGETS := libp2g_scatter.so libp2g_fused.so

.PHONY: all clean

all: $(TARGETS)

libp2g_scatter.so: p2g_scatter.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

libp2g_fused.so: p2g_fused.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGETS)
