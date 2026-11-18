#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

PYTHON="${PYTHON:-$(bobr_python "$ROOT")}"
SUITE="${1:-smoke}"

export PYTHONPATH="${ROOT}/python:${ROOT}:${PYTHONPATH:-}"

run_py() {
  echo "── python $*"
  "$PYTHON" "$@"
}

run_make() {
  echo "── make $*"
  make "$@"
}

failed=0

case "$SUITE" in
  smoke)
    # Gradient checks against torch, written against the current graph API
    run_py python/tests/test_gradients.py || failed=1
    ;;
  full)
    run_py python/tests/test_gradients.py || failed=1
    # Op-by-op checks vs torch (forward + backward), shape/reduction checks, and the C++ suite
    run_py python/tests/test_ops_vs_torch.py || failed=1
    run_py python/tests/test_shapes.py || failed=1
    run_py python/tests/test_advanced_ops.py || failed=1
    run_make run_gen_test || failed=1
    ;;
  *)
    echo "Unknown suite: $SUITE (use smoke or full)" >&2
    exit 1
    ;;
esac

if [[ $failed -eq 0 ]]; then
  echo ""
  echo "✓ test $SUITE passed"
else
  echo ""
  echo "✗ test $SUITE had failures" >&2
  exit 1
fi
