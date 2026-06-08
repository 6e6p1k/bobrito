# ── Compilers ──
NVCC      := nvcc
CXX       := g++

# ── Flags ──
# -std=c++17  : use C++17 features (structured bindings, shared_ptr, etc.)
# -lcublas    : link against cuBLAS library (for matmul)
# -Isrc       : look for headers in src/ directory
CXXFLAGS  := -std=c++17 -Isrc
NVCCFLAGS := -std=c++17 -Isrc -I/usr/include
LDFLAGS   := -lcublas -lcurand -lcuda -lcudart -lnvrtc

# ── Directories ──
SRC_DIR   := src
TEST_DIR  := tests
BUILD_DIR := build
OUT_DIR   := out

CPP_SRCS := $(SRC_DIR)/shape_data.cpp \
			$(SRC_DIR)/utils.cpp \
			$(SRC_DIR)/nodes.cpp \
			$(SRC_DIR)/evaluate.cpp \
			$(SRC_DIR)/graph_viz.cpp

CU_SRCS  := $(SRC_DIR)/primitives.cu \
			$(SRC_DIR)/kernels.cu \
			$(SRC_DIR)/allocator.cu \
			$(SRC_DIR)/optim.cu

# Convert source paths to object paths:
#   $(patsubst pattern,replacement,text)
#   src/node.cpp → build/node.o
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))

# All object files combined — needed to link any executable
ALL_OBJS := $(CPP_OBJS) $(CU_OBJS)

# ══════════════════════════════════════════════════════════════
# Directory creation
# ══════════════════════════════════════════════════════════════

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ══════════════════════════════════════════════════════════════
# Pattern rules — how to compile each file type
# ══════════════════════════════════════════════════════════════

# Rule: any .cpp → .o  (uses g++)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule: any .cu → .o  (uses nvcc)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ══════════════════════════════════════════════════════════════
# Executables
# ══════════════════════════════════════════════════════════════

$(BUILD_DIR)/new_test: $(TEST_DIR)/new_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/general_test: $(TEST_DIR)/general_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/memory_leak_test: $(TEST_DIR)/memory_leak_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/mnist_test: $(TEST_DIR)/mnist_test.cpp $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# ══════════════════════════════════════════════════════════════
# Convenience targets
# ══════════════════════════════════════════════════════════════

new_test:			$(BUILD_DIR)/new_test
general_test:		$(BUILD_DIR)/general_test
memory_leak_test:	$(BUILD_DIR)/memory_leak_test
mnist_test:			$(BUILD_DIR)/mnist_test

run_new_test: new_test
	./$(BUILD_DIR)/new_test

run_gen_test: general_test
	./$(BUILD_DIR)/general_test

run_mem_leak_test: memory_leak_test
	./$(BUILD_DIR)/memory_leak_test

run_mnist_test: mnist_test
	./$(BUILD_DIR)/mnist_test

# ══════════════════════════════════════════════════════════════
# Python
# ══════════════════════════════════════════════════════════════

# ── Python-specific objects (compiled with -fPIC) ──
PY_BUILD_DIR 	:= build/python
PYTHON_DIR		:= python

$(PY_BUILD_DIR):
	mkdir -p $(PY_BUILD_DIR)

# Python needs these sources (no tensor.cu/bobr.cu — those have main())
PY_CPP_SRCS :=  $(SRC_DIR)/nodes.cpp \
                $(SRC_DIR)/utils.cpp \
				$(SRC_DIR)/graph_viz.cpp \
                $(SRC_DIR)/evaluate.cpp \
				$(SRC_DIR)/shape_data.cpp
                
PY_CU_SRCS  := $(SRC_DIR)/allocator.cu \
               $(SRC_DIR)/module.cu \
               $(SRC_DIR)/kernels.cu \
			   $(SRC_DIR)/optim.cu \
			   $(SRC_DIR)/primitives.cu \
			   $(SRC_DIR)/fusion_reactor.cu

PY_CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(PY_BUILD_DIR)/%.o,$(PY_CPP_SRCS))
PY_CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(PY_BUILD_DIR)/%.o,$(PY_CU_SRCS))
PY_ALL_OBJS := $(PY_CPP_OBJS) $(PY_CU_OBJS)

PYBIND_FLAGS := $(shell python3 -m pybind11 --includes)
PYTHON_SUFFIX := $(shell python3-config --extension-suffix)

# Pattern rules for python objects — same as normal but with -fPIC
$(PY_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(PY_BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(PY_BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(PY_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $< -o $@

# Python shared library
$(PY_BUILD_DIR)/bobr$(PYTHON_SUFFIX): $(PYTHON_DIR)/bindings.cu $(PY_ALL_OBJS) | $(PY_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC $(PYBIND_FLAGS) $^ -o $@ $(LDFLAGS)

python: $(PY_BUILD_DIR)/bobr$(PYTHON_SUFFIX)
	rm -f $(PYTHON_DIR)/bobr$(PYTHON_SUFFIX)
	cp $(PY_BUILD_DIR)/bobr$(PYTHON_SUFFIX) $(PYTHON_DIR)/bobr$(PYTHON_SUFFIX)

# ══════════════════════════════════════════════════════════════
# clean & phony
# ══════════════════════════════════════════════════════════════

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(OUT_DIR)

clean_allocator:
	rm -rf $(BUILD_DIR)/allocator.o

# .PHONY tells make these aren't real files
.PHONY: clean new_test general_test \
        run_new_test run_gen_test \
        memory_leak_test run_mem_leak_test \
        mnist_test run_mnist_test \
		python