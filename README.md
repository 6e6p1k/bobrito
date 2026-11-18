# Bobrito

A small deep learning framework in C++ and CUDA with Python bindings. Operations build a lazy static graph, and nothing runs until a result is asked for. The backward pass is built as more nodes of the same graph, so a whole training step (forward, backward and the optimizer update) can be captured into one CUDA Graph and replayed with a single launch.

This is step 4 of [backprop-to-cuda](https://github.com/6e6p1k/backprop-to-cuda), which holds the three earlier versions: autograd over NumPy, a C++ engine on the CPU, and an eager CUDA prototype.

## Example

```python
import bobrito as bobr

model = bobr.Sequential()
model.add(bobr.LinearRelu(784, 128))
model.add(bobr.LinearRelu(128, 64))
model.add(bobr.Linear(64, 10))

x = bobr.from_numpy(x_batch)            # float32 arrays; these buffers are refilled every step
y = bobr.from_numpy(y_onehot_batch)
logits = model.forward(x)               # builds graph nodes, runs nothing
loss = bobr.cross_entropy(logits, y)

g = bobr.make_model(loss)
g.build_backward()                      # adds the backward nodes to the same graph
opt = bobr.AdamW(params=model.parameters(), lr=1e-3, weight_decay=0.0)

step = g.train_step(opt)                # first call runs eagerly, the second captures a CUDA Graph, later calls replay it
for xb, yb in batches:                  # CuPy or torch arrays already on the GPU
    x.data_from_cuda(xb)
    y.data_from_cuda(yb)
    step()
```

`python/bench/mnist.py` is the full version of this.

## What is inside

| Files | Contents |
|---|---|
| `src/nodes`, `src/primitives`, `src/ops.h` | Graph nodes and the operations: elementwise arithmetic with broadcasting, `relu`, `exp`, `log`, `sqrt`, `matmul`, `sum` / `min` / `max` (over one dimension or the whole tensor), `reshape`, `contiguous`, `linear`, `softmax`, `cross_entropy` |
| `src/shape_data` | Shapes and strides, checked when a node is created |
| `src/evaluate` | Topological order, forward pass, building the backward nodes, forward plus backward in one pass |
| `src/kernels` | CUDA kernels for strided and broadcast elementwise operations and the fused backward kernels |
| `src/primitives.cu` | Batched matrix multiply through cuBLAS (TF32 tensor-op math mode), reductions through CUB, random initialisers through cuRAND |
| `src/allocator` | Device memory from a CUDA memory pool with `cudaMallocAsync` and `cudaFreeAsync`, ordered on the framework's stream |
| `src/stream`, `src/train_step` | The single stream everything runs on, CUDA Graph capture and replay, and a train step that recaptures when a buffer moves |
| `src/module`, `src/optim` | `Linear`, `LinearRelu`, `Sequential`; AdamW |
| `src/fusion`, `src/fusion_reactor` | Kernel fusion compiled at run time with NVRTC. In progress and not used by evaluation yet |
| `src/graph_viz` | Graph export to Graphviz |
| `python/bindings.cu` | The pybind11 module, `bobrito` |

## Build

Needs Linux (WSL2 works) with an NVIDIA GPU and driver, `python3`, `make`, `g++`, and Graphviz for graph export. Everything else, including PyTorch for the tests and the CUDA 13.2 toolkit, is installed into a `.venv` inside the repository.

```bash
./bobr install          # .venv, CUDA toolkit, config.mk (set GPU_ARCH there if your GPU is not sm_120)
./bobr doctor
./bobr build python
./bobr test smoke
```

All commands are in [docs/CLI.md](docs/CLI.md); the build pipeline and the dev panel are in [docs/DEV.md](docs/DEV.md).

## Tests

- `./bobr test smoke` runs `python/tests/test_gradients.py`: every operation's forward and backward results against PyTorch, and against numerical gradients from finite differences.
- `./bobr test full` adds the op-by-op, shape and advanced-op suites and the two C++ suites in `tests/`.

## Benchmark

`./bobr bench mnist` (`python/bench/mnist.py`) trains a 784-128-64-10 MLP on MNIST with AdamW, batch 64, 5 epochs, same architecture and precision for every framework (fp32 with TF32 matmuls). "graph" rows capture the whole training step into one CUDA Graph. Data is on the GPU before timing starts, every timed region has a warm-up, and there is one synchronisation per epoch. On an RTX 5070 (sm_120), CUDA 13.2, PyTorch 2.15:

| framework | test acc | train time | ms/iter |
|---|---|---|---|
| bobr | 96.67% | 0.91s | 0.195 |
| bobr graph | 97.20% | 0.39s | 0.084 |
| torch eager | 96.86% | 2.05s | 0.437 |
| torch full graph | 96.41% | 0.63s | 0.134 |
| torch compile | 96.92% | 2.39s | 0.509 |
| torch compile (reduce-overhead) | 96.85% | 2.14s | 0.457 |

bobr graph is 5.2x faster than torch eager and 1.6x faster than torch's own CUDA Graph capture.

This is a small model, where a step is mostly the overhead of launching kernels rather than compute. Nothing larger has been measured yet.

## Limits

- float32 only, one GPU, one stream.
- The graph is static: shapes are fixed when it is built, so every batch has the same size.
- Layers are `Linear` and `LinearRelu`; the only optimizer is AdamW.
- Fusion is not wired in.
- The scripts at the repository root (`debug_*.py`, `test_*.py`) are older debugging scripts, and broken old tests are kept in `tests/legacy/` and `python/tests/legacy/`. Neither is part of the test suites.
