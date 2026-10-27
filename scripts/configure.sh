#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

echo "== bobr configure (compile_commands.json for clangd) =="

bobr_ensure_config "$ROOT"
VENV_DIR="$(bobr_venv "$ROOT")"
PY="$(bobr_python "$ROOT")"

if command -v bear &>/dev/null; then
  echo "Using bear..."
  bear -- make -B python 2>/dev/null || bear -- make python
else
  if [[ ! -x "${VENV_DIR}/bin/compiledb" ]]; then
    echo "Installing compiledb into ${VENV_DIR}..."
    "$PY" -m pip install -q compiledb
  fi
  echo "Using compiledb..."
  "${VENV_DIR}/bin/compiledb" -n make python
fi

if [[ -f compile_commands.json ]]; then
  echo "✓ compile_commands.json written ($(wc -l < compile_commands.json) lines)"
  echo "  Restart clangd in Cursor: Command Palette → 'clangd: Restart language server'"
else
  echo "✗ compile_commands.json not created" >&2
  exit 1
fi
