# Development guide

## Build pipeline

```
src/*.cu, src/*.cpp, src/*.h, src/*.cuh
        ↓  make (with -MMD header deps)
build/python/*.o
        ↓  nvcc -shared + pybind
build/python/bobrito.cpython-*.so
        ↓  cp
python/bobrito.cpython-*.so   ← import bobrito loads this
```

**Common mistake:** editing headers or `.cu` files but only rebuilding one `.o` without relinking `python/`.  
Always run `./bobr build python` after logic changes. Use `./bobr doctor` to catch stale objects.

## Toolchain

Python, torch and the CUDA 13.2 toolkit all live in the repo-local `.venv` (`./bobr install`); make resolves the paths from `config.mk` over `config.mk.example`. See [CLI.md](CLI.md).

## Official tests

| Suite | Command | Includes |
|-------|---------|----------|
| smoke | `./bobr test smoke` | gradients vs torch (`test_gradients.py`) |
| full | `./bobr test full` | smoke + ops vs torch + shapes + advanced (not yet ported to the Graph API) + general_test |

Legacy / debug scripts live at repo root (`debug_*.py`) or `python/tests/mini_tests*.py` — not in official suites.

## Dev panel

```bash
./bobr ui
```

Open http://localhost:1420 (WSL forwards to Windows Chrome automatically).

Buttons call `./bobr` — same as CLI. See `dev-panel/manifest.json`.

## Remote GPU (Mac → Linux pod)

1. Clone repo on the GPU machine
2. Cursor Remote SSH into the pod
3. `./bobr install && ./bobr configure && ./bobr build python`
4. `./bobr ui` — use forwarded port in browser on Mac

CUDA commands never run on the Mac; only the IDE and browser are local.
