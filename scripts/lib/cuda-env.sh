#!/usr/bin/env bash
# Shared environment for bobr scripts.
#
# make is the single source of truth: config.mk (if present) overrides config.mk.example, and
# every script asks make for the resolved value instead of parsing config.mk itself.
# By default everything points into the repo-local .venv (see config.mk.example).

# Print a variable as make resolves it, e.g. bobr_var "$ROOT" CUDA_HOME
bobr_var() {
  local root="${1:?root}" name="${2:?name}"
  make -s --no-print-directory -C "$root" "print-${name}" 2>/dev/null
}

bobr_python()       { bobr_var "${1:?root}" PYTHON; }
bobr_cuda_home()    { bobr_var "${1:?root}" CUDA_HOME; }
bobr_cuda_lib_dir() { bobr_var "${1:?root}" CUDA_LIB_DIR; }
bobr_gpu_arch()     { bobr_var "${1:?root}" GPU_ARCH; }
bobr_venv()         { bobr_var "${1:?root}" VENV; }
bobr_nvcc()         { echo "$(bobr_cuda_home "${1:?root}")/bin/nvcc"; }

# Make sure config.mk exists (it is optional for make, but scripts and the panel expect it)
bobr_ensure_config() {
  local root="${1:?root}"
  if [[ ! -f "$root/config.mk" ]]; then
    cp "$root/config.mk.example" "$root/config.mk"
    echo "→ created config.mk from config.mk.example (set GPU_ARCH if your GPU is not sm_120)"
  fi
}

# Print a system CUDA toolkit root (not the venv one), or return 1.
bobr_detect_system_cuda_home() {
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    echo /usr/local/cuda
    return 0
  fi
  local d
  for d in /usr/local/cuda-*; do
    if [[ -x "$d/bin/nvcc" ]]; then
      echo "$d"
      return 0
    fi
  done
  if command -v nvcc &>/dev/null; then
    local real home
    real="$(readlink -f "$(command -v nvcc)")"
    home="$(dirname "$(dirname "$real")")"
    # Ubuntu apt: nvcc lives under /usr/lib/nvidia-cuda-toolkit, headers in /usr/include
    [[ "$home" == */nvidia-cuda-toolkit ]] && home=/usr
    echo "$home"
    return 0
  fi
  return 1
}

# Print the directory under a toolkit root that holds libcudart, or return 1.
bobr_detect_cuda_lib_dir() {
  local home="${1:-}" d
  for d in "${home}/lib64" "${home}/lib" /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
    if compgen -G "${d}/libcudart.so*" >/dev/null; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

# 0 = the toolkit make will use is there, 1 = not (prints how to fix)
bobr_cuda_preflight() {
  local root="${1:?root}"
  local home lib nvcc
  home="$(bobr_cuda_home "$root")"
  lib="$(bobr_cuda_lib_dir "$root")"
  nvcc="${home}/bin/nvcc"

  if [[ ! -x "$nvcc" ]]; then
    echo "CUDA toolkit not found where make looks for it."
    echo ""
    echo "  CUDA_HOME=${home}"
    echo "  missing:   ${nvcc}"
    echo ""
    echo "  Fix:"
    echo "    ./bobr install-cuda              # CUDA 13 into .venv (default, no sudo)"
    echo "    ./bobr configure-cuda --system   # or use a system toolkit (/usr/local/cuda)"
    return 1
  fi

  if ! compgen -G "${lib}/libcudart.so*" >/dev/null; then
    echo "CUDA libraries not found."
    echo ""
    echo "  CUDA_LIB_DIR=${lib} has no libcudart"
    echo ""
    echo "  Fix: ./bobr install-cuda (venv toolkit) or ./bobr configure-cuda --system"
    return 1
  fi
  return 0
}

# 0 = the toolkit's nvcc accepts GPU_ARCH, 1 = not
bobr_check_gpu_arch() {
  local root="${1:?root}" arch="${2:?arch}"
  local nvcc tmp
  nvcc="$(bobr_nvcc "$root")"
  [[ -x "$nvcc" ]] || return 1
  tmp="$(mktemp /tmp/bobr-nvcc-arch-XXXX.cu)"
  echo "// arch probe" > "$tmp"
  if "$nvcc" -std=c++17 -arch="$arch" -c "$tmp" -o "${tmp}.o" 2>/dev/null; then
    rm -f "$tmp" "${tmp}.o"
    return 0
  fi
  rm -f "$tmp" "${tmp}.o"
  return 1
}

bobr_suggest_gpu_arch() {
  local cap="${1:-}"
  if [[ -n "$cap" ]]; then
    echo "sm_${cap//./}"
    return 0
  fi
  return 1
}

# nvidia-smi is not on PATH under WSL; it lives in /usr/lib/wsl/lib
bobr_nvidia_smi() {
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi "$@"
  elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    /usr/lib/wsl/lib/nvidia-smi "$@"
  else
    return 127
  fi
}

# Set or clear one `NAME ?= value` line in config.mk
bobr_config_set() {
  local root="${1:?root}" name="${2:?name}" value="${3:-}"
  local mk="$root/config.mk"
  bobr_ensure_config "$root" >/dev/null
  sed -i "/^${name}[[:space:]]*[?:]\{0,1\}=/d" "$mk"
  if [[ -n "$value" ]]; then
    echo "${name} ?= ${value}" >> "$mk"
  fi
}
