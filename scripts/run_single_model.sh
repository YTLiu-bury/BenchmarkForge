#!/usr/bin/env bash
set -Eeuo pipefail

if (($# < 1)); then
  echo "Usage: $0 SUITE/MODEL [BACKEND] [DEVICE]" >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$ROOT/config/env.local.sh" 2>/dev/null || source "$ROOT/config/env.example.sh"
# shellcheck disable=SC1090
source "$ROOT/scripts/activate_runtime.sh"

MODEL="$1"
BACKEND="${2:-triton_experimental}"
DEVICE="${3:-0}"
PYTHON="$BENCHMARK_PYTHON"
RUN_ID="single-$(echo "$MODEL" | tr '/ ' '__')-$BACKEND-$(date +%Y%m%d-%H%M%S)"

exec "$PYTHON" "$ROOT/benchctl.py" run \
  --set daily-full \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --devices "$DEVICE" \
  --run-id "$RUN_ID" \
  --continue-on-error \
  "${@:4}"
