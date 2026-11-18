# bobr CLI

All dev workflows go through `./bobr`. The web dev panel calls these same commands.

## Quick start

```bash
git clone https://github.com/6e6p1k/bobrito
cd bobrito
./bobr install                   # .venv: torch (CUDA 13.2), pybind11, CUDA 13.2 toolkit; creates config.mk
./bobr doctor
./bobr build python
./bobr test smoke
./bobr configure                 # compile_commands.json for clangd
./bobr ui                        # optional web panel
```

Everything lives in the repo-local `.venv`; nothing is installed system-wide and no sudo is needed. The only system requirements are the NVIDIA driver, `python3`, `make` and `g++`, plus **Graphviz** (the `dot` binary) for graph export: `sudo apt install graphviz`. Without it, `./bobr test smoke` fails with `sh: dot: not found` / `Graph export failed.`

## Commands

| Command | Description |
|---------|-------------|
| `./bobr install` | Create `.venv`, pip install `requirements-dev.txt`, run `install-cuda`, create `config.mk`, dev-panel npm install |
| `./bobr install-cuda` | CUDA 13.2 toolkit (nvcc, cuBLAS, cuRAND, NVRTC, CUB) from NVIDIA's pip wheels into `.venv`, assembled as `.venv/cuda` |
| `./bobr install-cuda --check` | Verify the toolkit make will use (headers and libraries) |
| `./bobr configure` | Generate `compile_commands.json` (for clangd) |
| `./bobr configure-cuda` | Use the `.venv` toolkit (drops CUDA overrides from `config.mk`) |
| `./bobr configure-cuda --system` | Use a system toolkit (`/usr/local/cuda`), written into `config.mk` |
| `./bobr doctor` | Health check: venv, torch, toolkit, GPU_ARCH, import bobrito, IDE db, stale builds |
| `./bobr doctor --json` | Machine-readable output |
| `./bobr build python` | Build + copy pybind11 extension to `python/` |
| `./bobr build primitives` | Recompile one `.o` (then rebuild python to link) |
| `./bobr build clean` | `make clean` |
| `./bobr test smoke` | Gradient checks against torch (`test_gradients.py`) |
| `./bobr test full` | Smoke + ops vs torch, shapes, advanced ops + C++ general_test and new_test |
| `./bobr run-script <path>` | Run a script with the venv Python and `python/` on the path |
| `./bobr ui` | Web dev panel at http://localhost:1420 |

## Per-machine config

`config.mk` (gitignored) overrides `config.mk.example`, which defaults everything to the venv:

- `GPU_ARCH` — e.g. `sm_86`, `sm_89`, `sm_120` (RTX 50xx)
- `VENV` — `$(CURDIR)/.venv`
- `PYTHON` — `$(VENV)/bin/python`
- `CUDA_HOME` — `$(VENV)/cuda`; set it to `/usr/local/cuda` for a system toolkit (`./bobr configure-cuda --system`)
- `CUDA_LIB_DIR` — `$(CUDA_HOME)/lib64`; also the extension's rpath, so `import bobrito` needs no `LD_LIBRARY_PATH`

Scripts never parse `config.mk`; they ask make, so they always agree with the build:

```bash
make -s print-CUDA_HOME
make -s print-PYTHON
```

Ubuntu's apt `nvidia-cuda-toolkit` (12.0) cannot target sm_120; an `nvcc` on PATH is ignored unless `CUDA_HOME` points at it.

## IDE (clangd)

1. Use clangd as the C++/CUDA language server (Zed uses it by default; in VS Code/Cursor install the **clangd** extension, MS C/C++ IntelliSense is disabled in `.vscode/settings.json`)
2. `./bobr configure`
3. Restart language server: *clangd: Restart language server*

Red squiggles after clone almost always mean step 2 was skipped.

## Agents

Use `./bobr` subcommands directly. Do not depend on the web UI.

```bash
./bobr build python && ./bobr test smoke
```
