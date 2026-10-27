#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/dev-panel"

if [[ ! -d node_modules ]]; then
  npm install
fi

echo "bobr dev panel → http://localhost:1420"
echo "Press Ctrl+C to stop."
npm run panel:dev
