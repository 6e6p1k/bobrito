#!/usr/bin/env bash
# Install the CUDA toolkit bobrito builds against into the repo-local .venv, from NVIDIA's
# pip wheels (CUDA 13.2: nvcc, cuBLAS, cuRAND, NVRTC, CUB/CCCL). No sudo, works for sm_120.
# Then assemble $(VENV)/cuda with the usual bin/ include/ lib64/ layout that the Makefile expects.
#
# Usage:
#   ./scripts/install-cuda-deps.sh          # install into .venv (run ./bobr install first)
#   ./scripts/install-cuda-deps.sh --check  # verify only, no install
#
# Using a system toolkit instead (/usr/local/cuda): ./bobr configure-cuda --system
set -euo pipefail

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

check_libs() {
  local ok=0 home lib nvcc
  home="$(bobr_cuda_home "$ROOT")"
  lib="$(bobr_cuda_lib_dir "$ROOT")"
  nvcc="${home}/bin/nvcc"
  echo "CUDA_HOME=${home}"
  echo "CUDA_LIB_DIR=${lib}"

  if [[ -x "$nvcc" ]]; then
    echo "✓ nvcc ($("$nvcc" --version | grep release | sed 's/^ *//'))"
  else
    echo "✗ nvcc — ${nvcc} missing"
    ok=1
  fi

  local h
  for h in cuda_runtime.h cublas_v2.h curand.h nvrtc.h cuda.h; do
    if [[ -f "${home}/include/${h}" ]]; then
      echo "✓ ${h}"
    else
      echo "✗ ${h} — not under ${home}/include"
      ok=1
    fi
  done

  # CUB ships with CCCL: include/cub (CUDA 12) or include/cccl/cub (CUDA 13, found by nvcc itself)
  if [[ -f "${home}/include/cub/cub.cuh" || -f "${home}/include/cccl/cub/cub.cuh" ]]; then
    echo "✓ cub/cub.cuh"
  else
    echo "✗ cub/cub.cuh — not under ${home}/include or ${home}/include/cccl"
    ok=1
  fi

  local l
  for l in libcudart libcublas libcurand libnvrtc libcuda; do
    if compgen -G "${lib}/${l}.so" >/dev/null; then
      echo "✓ ${l}.so"
    else
      echo "✗ ${l}.so — not in ${lib}"
      ok=1
    fi
  done
  return $ok
}

# $(VENV)/cuda: bin, include, nvvm point into the wheel; lib64 holds every runtime library
# plus the unversioned names the linker wants (-lcublas) and the driver's libcuda.so.
assemble_cuda_home() {
  local py="$1" venv="$2"
  local site cu dest f b base
  site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  cu="${site}/nvidia/cu13"
  if [[ ! -x "${cu}/bin/nvcc" ]]; then
    echo "✗ ${cu}/bin/nvcc missing — pip install -r requirements-cuda.txt failed?" >&2
    return 1
  fi

  dest="${venv}/cuda"
  rm -rf "$dest"
  mkdir -p "${dest}/lib64"
  for f in bin include nvvm; do
    ln -s "${cu}/${f}" "${dest}/${f}"
  done
  for f in "${cu}"/lib/*.so*; do
    ln -s "$f" "${dest}/lib64/"
  done
  for f in "${dest}"/lib64/lib*.so.*; do
    b="$(basename "$f")"
    base="${b%%.so.*}.so"
    [[ -e "${dest}/lib64/${base}" ]] || ln -s "$(readlink "$f")" "${dest}/lib64/${base}"
  done

  # The driver library comes from the system (WSL: /usr/lib/wsl/lib). Only the unversioned name
  # is linked here, so at run time libcuda.so.1 still resolves to the driver's own copy.
  for f in /usr/lib/wsl/lib/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
           /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so /usr/lib64/libcuda.so.1; do
    if [[ -e "$f" ]]; then
      ln -sf "$f" "${dest}/lib64/libcuda.so"
      break
    fi
  done
  echo "✓ assembled ${dest} → ${cu}"
}

echo "== bobr CUDA toolkit (.venv) =="
echo "repo: $ROOT"
echo ""

if $CHECK_ONLY; then
  check_libs
  exit $?
fi

PY="$(bobr_python "$ROOT")"
VENV_DIR="$(bobr_venv "$ROOT")"
if [[ ! -x "$PY" ]]; then
  echo "✗ ${PY} not found — run ./bobr install first (it creates the venv and calls this script)" >&2
  exit 1
fi
if [[ "$PY" != "${VENV_DIR}/"* ]]; then
  echo "PYTHON=${PY} is not inside VENV=${VENV_DIR}; refusing to pip install into it." >&2
  echo "Use a system toolkit with ./bobr configure-cuda --system, or reset PYTHON in config.mk." >&2
  exit 1
fi

echo "→ pip install -r requirements-cuda.txt"
"$PY" -m pip install -q -r requirements-cuda.txt
assemble_cuda_home "$PY" "$VENV_DIR"

echo ""
echo "== Verifying =="
check_libs

echo ""
echo "Done. Next: ./bobr build python && ./bobr test smoke"
