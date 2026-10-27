#!/usr/bin/env bash
# ./bobr configure-cuda           use the .venv toolkit (default): drops CUDA overrides from config.mk
# ./bobr configure-cuda --system  use a system toolkit: writes CUDA_HOME / CUDA_LIB_DIR into config.mk
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

echo "== bobr configure-cuda =="
bobr_ensure_config "$ROOT"

if [[ "${1:-}" == "--system" ]]; then
  if ! home="$(bobr_detect_system_cuda_home)"; then
    echo "No system CUDA toolkit found (/usr/local/cuda*, or nvcc on PATH)." >&2
    echo "Use the venv toolkit instead: ./bobr install-cuda" >&2
    exit 1
  fi
  lib="$(bobr_detect_cuda_lib_dir "$home")" || lib="${home}/lib64"
  bobr_config_set "$ROOT" CUDA_HOME "$home"
  bobr_config_set "$ROOT" CUDA_LIB_DIR "$lib"
else
  bobr_config_set "$ROOT" CUDA_HOME ""
  bobr_config_set "$ROOT" CUDA_LIB_DIR ""
fi

home="$(bobr_cuda_home "$ROOT")"
echo "✓ make now uses:"
echo "    CUDA_HOME=${home}"
echo "    CUDA_LIB_DIR=$(bobr_cuda_lib_dir "$ROOT")"
echo "    nvcc=${home}/bin/nvcc"
if [[ ! -x "${home}/bin/nvcc" ]]; then
  echo "⚠ that nvcc does not exist yet — run ./bobr install-cuda"
fi
