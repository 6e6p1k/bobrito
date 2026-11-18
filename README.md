# bobr-framework-cuda

CUDA autograd engine with a static lazy graph. See [docs/CLI.md](docs/CLI.md) and [docs/DEV.md](docs/DEV.md).

## Quick start

```bash
./bobr install                  # .venv with torch and the CUDA 13 toolkit; creates config.mk
                                # (set GPU_ARCH in config.mk if your GPU is not sm_120)
./bobr configure                # compile_commands.json for clangd
./bobr build python
./bobr doctor
./bobr test smoke
./bobr bench mnist
./bobr ui                       # optional web dev panel
```

## Benchmarks
`./bobr bench mnist` (`python/bench/mnist.py`) trains a 784-128-64-10 MLP on MNIST with AdamW, batch 64, 5 epochs, same architecture and precision for every framework (fp32 with TF32 matmuls). "graph" rows capture the whole training step into one CUDA Graph. On an RTX 5070 (sm_120):

| framework | test acc | train time | ms/iter |
|---|---|---|---|
| bobr | 96.67% | 0.91s | 0.195 |
| bobr graph | 97.20% | 0.39s | 0.084 |
| torch eager | 96.86% | 2.05s | 0.437 |
| torch full graph | 96.41% | 0.63s | 0.134 |
| torch compile | 96.92% | 2.39s | 0.509 |
| torch compile (reduce-overhead) | 96.85% | 2.14s | 0.457 |

bobr graph is 5.2x faster than torch eager and 1.6x faster than torch's own CUDA Graph capture.
