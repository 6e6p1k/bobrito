from setuptools import setup
from torch.utils.cpp_extension import BuildExtension  # nah don't use this

# use pybind11's cmake or just:
# nvcc -shared -std=c++17 -Xcompiler -fPIC $(python3 -m pybind11 --includes) \
#   bindings.cpp ops.cu allocator.cu kernels.cu utils.cpp node.cpp backward.cpp graph_viz.cpp visualize_graph.cpp \
#   -o bobrito$(python3-config --extension-suffix) -lcublas