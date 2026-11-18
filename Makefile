# bobr-framework-cuda Makefile
# Machine-specific settings: copy config.mk.example → config.mk
-include config.mk
include config.mk.example

ROOT_DIR   := $(abspath .)
SRC_DIR    := src
TEST_DIR   := tests
BUILD_DIR  := build
OUT_DIR    := out
PY_BUILD_DIR := build/python
PYTHON_DIR := python

NVCC         := $(CUDA_HOME)/bin/nvcc
CUDA_LIB_DIR ?= $(CUDA_HOME)/lib64
CXX          := g++

# Host + device flags
CXXFLAGS   := -std=c++17 -Isrc -I$(CUDA_HOME)/include -MMD -MP
NVCCFLAGS  := -std=c++17 -Isrc \
              -arch=$(GPU_ARCH) -lineinfo \
              -I$(CUDA_HOME)/include \
              -MMD -MP \
              -DBOBR_SRC_DIR=\"$(abspath $(SRC_DIR))\" \
              -DNVRTC_ARCH_FLAG=\"--gpu-architecture=$(GPU_ARCH)\"
ifdef ALLOC_DEBUG
        NVCCFLAGS += -DBOBR_ALLOC_DEBUG
endif
# rpath: the extension finds the toolkit's libraries without LD_LIBRARY_PATH
LDFLAGS    := -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) \
              -lcublas -lcurand -lcuda -lcudart -lnvrtc

export PATH := $(CUDA_HOME)/bin:$(PATH)

# ── C++ test binary sources ──
CPP_SRCS := $(SRC_DIR)/shape_data.cpp \
            $(SRC_DIR)/utils.cpp \
            $(SRC_DIR)/nodes.cpp \
            $(SRC_DIR)/evaluate.cpp \
            $(SRC_DIR)/graph_viz.cpp \
			$(SRC_DIR)/train_step.cpp

CU_SRCS  := $(SRC_DIR)/primitives.cu \
            $(SRC_DIR)/kernels.cu \
            $(SRC_DIR)/allocator.cu \
            $(SRC_DIR)/optim.cu \
            $(SRC_DIR)/stream.cu

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
ALL_OBJS := $(CPP_OBJS) $(CU_OBJS)

# ── Python extension sources ──
PY_CPP_SRCS := $(SRC_DIR)/nodes.cpp \
               $(SRC_DIR)/utils.cpp \
               $(SRC_DIR)/graph_viz.cpp \
               $(SRC_DIR)/evaluate.cpp \
               $(SRC_DIR)/shape_data.cpp \
			   $(SRC_DIR)/train_step.cpp

PY_CU_SRCS  := $(SRC_DIR)/allocator.cu \
               $(SRC_DIR)/module.cu \
               $(SRC_DIR)/kernels.cu \
               $(SRC_DIR)/optim.cu \
               $(SRC_DIR)/primitives.cu \
               $(SRC_DIR)/fusion_reactor.cu \
               $(SRC_DIR)/stream.cu

PY_CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(PY_BUILD_DIR)/%.o,$(PY_CPP_SRCS))
PY_CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(PY_BUILD_DIR)/%.o,$(PY_CU_SRCS))
PY_ALL_OBJS := $(PY_CPP_OBJS) $(PY_CU_OBJS)

PYBIND_FLAGS    := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
# sysconfig works inside a venv, where python3-config does not exist
PYTHON_SUFFIX   := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
PY_MODULE       := $(PY_BUILD_DIR)/bobrito$(PYTHON_SUFFIX)
PY_MODULE_LINK  := $(PYTHON_DIR)/bobrito$(PYTHON_SUFFIX)

# Auto-generated header dependencies
DEP_FILES := $(CPP_OBJS:.o=.d) $(CU_OBJS:.o=.d) \
             $(PY_CPP_OBJS:.o=.d) $(PY_CU_OBJS:.o=.d)
-include $(DEP_FILES)

# ══════════════════════════════════════════════════════════════
$(BUILD_DIR) $(PY_BUILD_DIR) $(OUT_DIR):
	mkdir -p $@

# ── Native test objects ──
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ── Python extension objects ──
$(PY_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(PY_BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(PY_BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(PY_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $< -o $@

$(PY_MODULE): $(PYTHON_DIR)/bindings.cu $(PY_ALL_OBJS) | $(PY_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC $(PYBIND_FLAGS) $^ -o $@ $(LDFLAGS)

# ══════════════════════════════════════════════════════════════
# Targets
# ══════════════════════════════════════════════════════════════

python: $(PY_MODULE)
	@cp -f $(PY_MODULE) $(PY_MODULE_LINK)
	@echo "→ $(PY_MODULE_LINK)"

$(BUILD_DIR)/general_test: $(TEST_DIR)/general_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/memory_leak_test: $(TEST_DIR)/memory_leak_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/mnist_test: $(TEST_DIR)/mnist_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/new_test: $(TEST_DIR)/new_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

general_test:      $(BUILD_DIR)/general_test
memory_leak_test:  $(BUILD_DIR)/memory_leak_test
mnist_test:        $(BUILD_DIR)/mnist_test
new_test:          $(BUILD_DIR)/new_test

run_gen_test:      general_test; ./$(BUILD_DIR)/general_test
run_mem_leak_test: memory_leak_test; ./$(BUILD_DIR)/memory_leak_test
run_mnist_test:    mnist_test; ./$(BUILD_DIR)/mnist_test
run_new_test:      new_test; ./$(BUILD_DIR)/new_test

# Print a resolved variable, e.g. make -s print-CUDA_HOME (used by scripts/lib/cuda-env.sh)
print-%:
	@echo '$($*)'

clean:
	rm -rf $(BUILD_DIR) $(OUT_DIR)
	rm -f $(PY_MODULE_LINK)

# ══════════════════════════════════════════════════════════════
.PHONY: python clean general_test memory_leak_test mnist_test new_test \
        run_gen_test run_mem_leak_test run_mnist_test run_new_test
