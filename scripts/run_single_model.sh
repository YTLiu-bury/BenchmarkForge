#!/usr/bin/env bash
set -Eeuo pipefail

if (($# < 1)); then
  echo "Usage: $0 SUITE/MODEL [BACKEND] [DEVICE]" >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Select the runtime profile without changing the historical default.
ENV_FILE="${BENCHMARK_ENV_FILE:-$ROOT/config/env.local.sh}"
if [[ "$ENV_FILE" != /* ]]; then
  ENV_FILE="$ROOT/$ENV_FILE"
fi
if [[ ! -f "$ENV_FILE" ]]; then
  if [[ "$ENV_FILE" == "$ROOT/config/env.local.sh" ]]; then
    ENV_FILE="$ROOT/config/env.example.sh"
  else
    echo "runtime environment file not found: $ENV_FILE" >&2
    exit 2
  fi
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
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
