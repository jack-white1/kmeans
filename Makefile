# Compiler
NVCC = nvcc

# Architecture
ARCH = -arch=sm_89

# Flags for NVCC to link cublas
CUBLAS_FLAGS = -lcublas -lcurand

# Source files
SRCS = single_kernel_k_means.cu multi_kernel_k_means.cu cublas_check.cu

# Extract executable names from the source files
EXES = $(SRCS:.cu=)

# Default target
all: $(EXES)

# Compile each source file to an executable
$(EXES): %: %.cu
	$(NVCC) $(ARCH) -o $@ $< $(CUBLAS_FLAGS)

# Clean target
clean:
	rm -f $(EXES)

# Phony targets
.PHONY: all clean