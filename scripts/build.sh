#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

TARGET="${1:-python}"

bobr_ensure_config "$ROOT"

if ! bobr_cuda_preflight "$ROOT"; then
  echo "" >&2
  exit 1
fi

gpu_arch="$(bobr_gpu_arch "$ROOT")"
if ! bobr_check_gpu_arch "$ROOT" "$gpu_arch"; then
  nvcc="$(bobr_nvcc "$ROOT")"
  compute_cap="$(bobr_nvidia_smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r' || true)"
  echo "GPU_ARCH=${gpu_arch} is not supported by ${nvcc} ($("$nvcc" --version 2>/dev/null | grep release | sed 's/^ *//'))" >&2
  if [[ -n "$compute_cap" ]]; then
    echo "  GPU compute capability: ${compute_cap} (try GPU_ARCH ?= $(bobr_suggest_gpu_arch "$compute_cap") in config.mk)" >&2
  fi
  if [[ -n "$BOBR_ARCH_PROBE_ERR" ]]; then
    echo "  nvcc probe output (the failure may not be the arch itself):" >&2
    bobr_arch_probe_error >&2
  fi
  echo "  Run ./bobr doctor for full diagnostics." >&2
  exit 1
fi

case "$TARGET" in
  python)
    make python
    ;;
  clean)
    make clean
    ;;
  allocator|primitives|kernels|evaluate|module|optim|fusion_reactor|nodes|utils|shape_data|graph_viz)
    # Single translation unit; the python module still needs relinking
    make "build/python/${TARGET}.o"
    echo ""
    echo "Note: .o rebuilt. For Python to pick up changes, also run: ./bobr build python"
    ;;
  *)
    # Pass through to make (e.g. build/python/allocator.o)
    make "$TARGET"
    ;;
esac
