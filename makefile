PYTHON ?= python3
PYTHON_CONFIG ?= python3-config

NUMPY_INCLUDE := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)
EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)

CXX ?= g++
CXXFLAGS_CPU := -std=c++17 -O3 -fPIC -Wall -Wextra -pedantic -fopenmp
CXXFLAGS_GPU_HOST := -std=c++17 -O3 -fPIC -Wall -Wextra -pedantic
INCLUDES := $(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CPU_LDFLAGS := -bundle -undefined dynamic_lookup -fopenmp
else
	CPU_LDFLAGS := -shared -fopenmp $(PYTHON_LDFLAGS)
endif

NVCC ?= nvcc
NVCC_PATH := $(shell command -v $(NVCC) 2>/dev/null)
ifeq ($(strip $(NVCC_PATH)),)
ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
NVCC := /usr/local/cuda/bin/nvcc
NVCC_PATH := $(NVCC)
else ifneq ($(wildcard /usr/local/cuda-13.2/bin/nvcc),)
NVCC := /usr/local/cuda-13.2/bin/nvcc
NVCC_PATH := $(NVCC)
endif
endif
NVCCFLAGS := -std=c++17 -O3 -ccbin $(CXX) \
	-Xcompiler -fPIC \
	-Xcompiler -Wall \
	-Xcompiler -Wextra \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_80,code=sm_80

TARGET_ACCEL := accel$(EXT_SUFFIX)
TARGET_SIM := simulator$(EXT_SUFFIX)
TARGET_ACCEL_GPU := accel_gpu$(EXT_SUFFIX)
TARGET_SIM_GPU := simulator_gpu$(EXT_SUFFIX)

GPU_CUDA_OBJECT := accel_gpu.cuda.o
GPU_HOST_OBJECT := simulator_gpu.host.o

CPU_TARGETS := $(TARGET_ACCEL) $(TARGET_SIM)
GPU_TARGETS := $(TARGET_ACCEL_GPU) $(TARGET_SIM_GPU)

ifeq ($(strip $(NVCC_PATH)),)
	HAVE_NVCC := 0
else
	HAVE_NVCC := 1
endif

.PHONY: all cpu gpu cuda_warn clean test

all: cpu
ifeq ($(HAVE_NVCC),1)
all: gpu
else
all: cuda_warn
endif

cpu: $(CPU_TARGETS)

ifeq ($(HAVE_NVCC),1)
gpu: $(GPU_TARGETS)
else
gpu: cuda_warn
endif

cuda_warn:
	@echo "WARNING: nvcc not found; skipping GPU targets. Set NVCC=/path/to/nvcc or add CUDA's bin directory to PATH."

$(TARGET_ACCEL): accel.cpp
	$(CXX) $(CXXFLAGS_CPU) $(INCLUDES) $(CPU_LDFLAGS) -o $@ $<

$(TARGET_SIM): simulator.cpp
	$(CXX) $(CXXFLAGS_CPU) $(INCLUDES) $(CPU_LDFLAGS) -o $@ $<

$(GPU_CUDA_OBJECT): accel_gpu.cu accel_gpu_api.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc -o $@ $<

$(GPU_HOST_OBJECT): simulator_gpu.cpp accel_gpu_api.h
	$(CXX) $(CXXFLAGS_GPU_HOST) $(INCLUDES) -c -o $@ $<

$(TARGET_ACCEL_GPU): $(GPU_CUDA_OBJECT)
	$(NVCC) $(NVCCFLAGS) --shared -o $@ $^ $(PYTHON_LDFLAGS) -lcudart

$(TARGET_SIM_GPU): $(GPU_HOST_OBJECT) $(GPU_CUDA_OBJECT)
	$(NVCC) $(NVCCFLAGS) --shared -o $@ $^ $(PYTHON_LDFLAGS) -lcudart

test: cpu $(if $(filter 1,$(HAVE_NVCC)),gpu,)
	$(PYTHON) test_accel.py

clean:
	rm -f $(TARGET_ACCEL) $(TARGET_SIM) $(TARGET_ACCEL_GPU) $(TARGET_SIM_GPU)
	rm -f $(GPU_CUDA_OBJECT) $(GPU_HOST_OBJECT)
	rm -rf __pycache__
	rm -f *.pyc
