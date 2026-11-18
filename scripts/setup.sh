#!/usr/bin/env bash
# ./bobr install — everything into the repo-local .venv:
#   1. create .venv (falls back to --without-pip when python3-venv/ensurepip is missing)
#   2. pip install -r requirements-dev.txt (torch for CUDA 13.2, pybind11, numpy, compiledb)
#   3. ./bobr install-cuda (CUDA 13.2 toolkit from pip wheels, assembled as .venv/cuda)
#   4. config.mk from config.mk.example if missing; dev-panel npm install if the panel is present
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

echo "== bobr setup =="
echo "repo: $ROOT"

bobr_ensure_config "$ROOT"

VENV_DIR="$(bobr_venv "$ROOT")"
PY="$(bobr_python "$ROOT")"
BASE_PY="${BASE_PYTHON:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "→ creating ${VENV_DIR} with $("$BASE_PY" --version 2>&1)"
  if ! "$BASE_PY" -m venv "$VENV_DIR" 2>/dev/null; then
    # Debian/Ubuntu without python3-venv: no ensurepip. Make the venv without pip and seed pip
    # into it with the system pip (needs pip >= 22.3 for --python).
    rm -rf "$VENV_DIR"
    "$BASE_PY" -m venv --without-pip "$VENV_DIR"
    if ! "$BASE_PY" -m pip --python "${VENV_DIR}/bin/python" install -q pip; then
      echo "✗ could not install pip into ${VENV_DIR}." >&2
      echo "  Install python3-venv (sudo apt install python3-venv) and rerun ./bobr install." >&2
      exit 1
    fi
  fi
else
  echo "→ using existing ${VENV_DIR}"
fi

if [[ "$PY" != "${VENV_DIR}/"* ]]; then
  echo "⚠ PYTHON=${PY} (config.mk) is outside ${VENV_DIR}; the build and tests will not use the venv." >&2
fi

echo "→ pip install -r requirements-dev.txt (torch, pybind11, numpy, compiledb)"
"${VENV_DIR}/bin/python" -m pip install -q --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -q -r requirements-dev.txt

echo "→ CUDA toolkit into ${VENV_DIR}"
bash "$ROOT/scripts/install-cuda-deps.sh"

if [[ -f dev-panel/package.json && ! -d dev-panel/node_modules ]]; then
  if command -v npm &>/dev/null; then
    echo "→ npm install (dev-panel)"
    (cd dev-panel && npm install)
  else
    echo "⚠ npm not found; skipping the dev panel (./bobr ui needs it)"
  fi
fi

echo ""
"${VENV_DIR}/bin/python" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
echo ""
echo "Done. Next:"
echo "  ./bobr doctor"
echo "  ./bobr build python"
echo "  ./bobr test smoke"
echo "  ./bobr configure       # compile_commands.json for clangd"
echo "  ./bobr ui              # optional dev panel"
