#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=lib/cuda-env.sh
source "$ROOT/scripts/lib/cuda-env.sh"

PYTHON="${PYTHON:-$(bobr_python "$ROOT")}"
SUITE="${1:-mnist}"
shift || true

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
  mnist)
    # MNIST training: bobr eager/graph vs torch eager/graph/compile
    run_py python/bench/mnist.py "$@" || failed=1
    ;;
  # something_else_for_later)
  #   run_py python/bench/something_new.py || failed=1
  #   ;;
  *)
    echo "Unknown suite: $SUITE (use mnist)" >&2
    exit 1
    ;;
esac

if [[ $failed -eq 0 ]]; then
  echo ""
  echo "✓ benched $SUITE successfully"
else
  echo ""
  echo "✗ encountered failures while benching $SUITE" >&2
  exit 1
fi
