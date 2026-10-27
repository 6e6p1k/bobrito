#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

JSON=false
[[ "${1:-}" == "--json" ]] && JSON=true

status=0
lines=()

record() {
  local level="$1" name="$2" msg="$3"
  lines+=("$level|$name|$msg")
  if [[ "$level" == "fail" ]]; then
    status=1
  fi
}

if [[ -f config.mk ]]; then
  record ok config_mk "config.mk present"
else
  record warn config_mk "missing — make uses config.mk.example defaults (./bobr install creates it)"
fi

# What make resolves (config.mk over config.mk.example) is what the build and tests use
VENV_DIR="$(bobr_venv "$ROOT")"
PYTHON="$(bobr_python "$ROOT")"
cuda_home="$(bobr_cuda_home "$ROOT")"
cuda_lib="$(bobr_cuda_lib_dir "$ROOT")"
nvcc="${cuda_home}/bin/nvcc"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  record ok venv "${VENV_DIR}"
else
  record fail venv "${VENV_DIR} missing — run: ./bobr install"
fi

if [[ -x "$PYTHON" ]] && "$PYTHON" -c "import pybind11" 2>/dev/null; then
  where="venv"
  [[ "$PYTHON" == "${VENV_DIR}/"* ]] || where="NOT the venv"
  record ok python "$("$PYTHON" --version 2>&1) at ${PYTHON} (${where})"
else
  record fail python "${PYTHON} missing or without pybind11 — run: ./bobr install"
fi

if [[ -x "$PYTHON" ]] && torch_info="$("$PYTHON" -c "import torch; print(torch.__version__, 'CUDA', torch.version.cuda)" 2>/dev/null)"; then
  record ok torch "$torch_info"
else
  record fail torch "torch not importable with ${PYTHON} — run: ./bobr install"
fi

if [[ -x "$nvcc" ]]; then
  ver="$("$nvcc" --version 2>/dev/null | grep release | head -1 | sed 's/^ *//' || true)"
  where="venv"
  [[ "$cuda_home" == "${VENV_DIR}/"* ]] || where="system toolkit"
  record ok cuda_home "CUDA_HOME=${cuda_home} (${where}; ${ver})"
else
  record fail cuda_home "CUDA_HOME=${cuda_home}, ${nvcc} missing — run: ./bobr install-cuda"
fi

if compgen -G "${cuda_lib}/libcudart.so*" >/dev/null; then
  record ok cuda_libs "CUDA_LIB_DIR=${cuda_lib}"
else
  record fail cuda_libs "libcudart not found in CUDA_LIB_DIR=${cuda_lib} — run: ./bobr install-cuda"
fi

# Informational: an nvcc on PATH is never used by the build unless CUDA_HOME points at it
if command -v nvcc &>/dev/null && [[ "$(readlink -f "$(command -v nvcc)")" != "$(readlink -f "$nvcc" 2>/dev/null)" ]]; then
  path_ver="$(nvcc --version 2>/dev/null | grep release | head -1 | sed 's/^ *//' || true)"
  record ok nvcc_path "PATH has $(command -v nvcc) (${path_ver}); not used, make uses CUDA_HOME"
fi

gpu_arch="$(bobr_gpu_arch "$ROOT")"
if [[ -x "$nvcc" ]]; then
  if bobr_check_gpu_arch "$ROOT" "$gpu_arch"; then
    record ok gpu_arch "GPU_ARCH=${gpu_arch} supported by ${nvcc}"
  else
    compute_cap="$(bobr_nvidia_smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r' || true)"
    hint="GPU_ARCH=${gpu_arch} not supported by ${nvcc}"
    if [[ -n "$compute_cap" ]]; then
      hint="${hint} — GPU compute ${compute_cap}, try GPU_ARCH ?= $(bobr_suggest_gpu_arch "$compute_cap") in config.mk"
    fi
    record fail gpu_arch "$hint"
  fi
fi

if gpu="$(bobr_nvidia_smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"; then
  record ok gpu "${gpu:-unknown}"
else
  record warn gpu "nvidia-smi unavailable"
fi

if [[ -x "$PYTHON" ]] && "$PYTHON" -c "import sys; sys.path.insert(0,'python'); import bobrito" 2>/dev/null; then
  so="$(ls python/bobrito*.so 2>/dev/null | head -1 || echo '?')"
  record ok bobr_import "import bobrito OK ($so)"
else
  record fail bobr_import "run: ./bobr build python"
fi

if [[ -f compile_commands.json ]]; then
  stale=""
  if find compile_commands.json -mmin +1440 2>/dev/null | grep -q .; then
    stale=" (stale — ./bobr configure)"
  fi
  record ok compile_db "present$stale"
else
  record warn compile_db "missing — run: ./bobr configure (only needed for clangd)"
fi

if [[ -f src/primitives.cu && -f build/python/primitives.o ]]; then
  if [[ src/primitives.cu -nt build/python/primitives.o ]]; then
    record warn build_stale "primitives.cu newer than .o — ./bobr build python"
  else
    record ok build_stale "primitives.o up to date"
  fi
fi

if $JSON; then
  echo "["
  n=${#lines[@]}
  for i in "${!lines[@]}"; do
    IFS='|' read -r level name msg <<< "${lines[$i]}"
    ok="true"
    [[ "$level" == "fail" ]] && ok="false" || true
    msg_json=$(printf '%s' "$msg" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    printf '  {"id":%s,"level":%s,"message":%s,"ok":%s}' \
      "$(printf '%s' "$name" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
      "$(printf '%s' "$level" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
      "$msg_json" "$ok"
    [[ $i -lt $((n - 1)) ]] && echo ","
  done
  echo "]"
  exit $status
fi

echo "== bobr doctor =="
for entry in "${lines[@]}"; do
  IFS='|' read -r level name msg <<< "$entry"
  case "$level" in
    fail) echo "✗ $name: $msg" ;;
    warn) echo "⚠ $name: $msg" ;;
    *)    echo "✓ $name: $msg" ;;
  esac
done
echo ""
[[ $status -eq 0 ]] && echo "Critical checks passed." || echo "Fix failures above."
exit $status
