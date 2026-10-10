#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTAINER="${BENCHMARK_CONTAINER:-ascendforge_ci}"
CONTAINER_WORKSPACE="${BENCHMARK_CONTAINER_WORKSPACE:-$ROOT}"
CONTAINER_ENV="${BENCHMARK_CONTAINER_ENV:-${BENCHMARK_ENV:-}}"
CONTAINER_CANN_HOME="${BENCHMARK_CONTAINER_CANN_HOME:-${CANN_HOME:-/usr/local/Ascend/cann-9.2.0}}"

# If this script is already running in the benchmark container, continue locally.
if [[ -f /.dockerenv ]]; then
  exec bash "$ROOT/scripts/run_daily.sh" "$@"
fi

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
  echo "benchmark container is not running: $CONTAINER" >&2
  exit 1
fi

exec docker exec \
  --env "BENCHMARK_ENV=$CONTAINER_ENV" \
  --env "CANN_HOME=$CONTAINER_CANN_HOME" \
  --env "BENCHMARK_CONTAINER_WORKSPACE=$CONTAINER_WORKSPACE" \
  --env "DATE=${DATE:-}" \
  --env "SET_NAME=${SET_NAME:-}" \
  --env "RUN_ID=${RUN_ID:-}" \
  --env "DVM_DEVICES=${DVM_DEVICES:-}" \
  --env "TRITON_DEVICES=${TRITON_DEVICES:-}" \
  --env "DYNAMIC_MODES=${DYNAMIC_MODES:-}" \
  --env "AUTO_PUBLISH=${AUTO_PUBLISH:-}" \
  --env "PUBLISH_WEBSITE_ROOT=${PUBLISH_WEBSITE_ROOT:-}" \
  --env "PUBLIC_WEBSITE_ROOT=${PUBLIC_WEBSITE_ROOT:-}" \
  "$CONTAINER" \
  bash -lc 'cd "$BENCHMARK_CONTAINER_WORKSPACE" && exec bash ./scripts/run_daily.sh "$@"' \
  bash "$@"
