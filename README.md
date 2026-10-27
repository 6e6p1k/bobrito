# bobr-framework-cuda

CUDA autograd engine with a static lazy graph. See [docs/CLI.md](docs/CLI.md) and [docs/DEV.md](docs/DEV.md).

## Quick start

```bash
./bobr install                  # .venv with torch and the CUDA 13 toolkit; creates config.mk
                                # (set GPU_ARCH in config.mk if your GPU is not sm_120)
./bobr configure                # compile_commands.json for Cursor/clangd
./bobr build python
./bobr doctor
./bobr test smoke
./bobr ui                       # optional web dev panel
```

## Benchmarks
 trains a 784-128-64-10 MLP on MNIST with AdamW, batch 64, 50 epochs, matched architecture and precision (fp32, no TF32 for either framework). On RTX Blackwell (sm_120):

bobr: 11.6s, 0.247 ms/iter, 96.9% test acc
torch eager: 19.8s, 0.422 ms/iter, 97.6% test acc
torch compile: 24.8s, 0.529 ms/iter, 97.7% test acc

1.71x faster than torch eager, 2.14x faster than torch compile.