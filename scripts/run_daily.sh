#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# On the 173 host, re-enter the known-good container before loading runtime
# variables. Inside the container this branch is skipped.
if [[ ! -f /.dockerenv && "${BENCHMARK_CONTAINER_REEXEC:-1}" == "1" ]]; then
  exec bash "$ROOT/scripts/run_daily_container.sh" "$@"
fi

# shellcheck disable=SC1091
source "$ROOT/config/env.local.sh" 2>/dev/null || source "$ROOT/config/env.example.sh"
# shellcheck disable=SC1090
source "$ROOT/scripts/activate_runtime.sh"

PYTHON="$BENCHMARK_PYTHON"
DATE="${DATE:-$(date +%Y%m%d)}"
DVM_DEVICES="${DVM_DEVICES:-4,5}"
TRITON_DEVICES="${TRITON_DEVICES:-1,2}"
SET_NAME="${SET_NAME:-daily-full}"
DYNAMIC_MODES="${DYNAMIC_MODES:-off,on}"
DYNAMIC_LABEL="${DYNAMIC_MODES//,/-}"
RUN_ID="${RUN_ID:-${DATE}-${SET_NAME}-dvm-triton-experimental-dynamic-${DYNAMIC_LABEL}}"
AUTO_PUBLISH="${AUTO_PUBLISH:-1}"
PUBLISH_WEBSITE_ROOT="${PUBLISH_WEBSITE_ROOT:-}"
PUBLIC_WEBSITE_ROOT="${PUBLIC_WEBSITE_ROOT:-}"

dynamic_args=()
IFS=',' read -r -a dynamic_values <<< "$DYNAMIC_MODES"
for dynamic_mode in "${dynamic_values[@]}"; do
  dynamic_args+=(--dynamic "$dynamic_mode")
done

"$PYTHON" "$ROOT/benchctl.py" run \
  --set "$SET_NAME" \
  --date "$DATE" \
  --run-id "$RUN_ID" \
  --backend dvm \
  --backend triton_experimental \
  --devices "dvm=$DVM_DEVICES" \
  --devices "triton_experimental=$TRITON_DEVICES" \
  "${dynamic_args[@]}" \
  --profile on \
  --continue-on-error \
  "$@"

if [[ "$AUTO_PUBLISH" == "1" ]]; then
  publish_args=(publish --run-id "$RUN_ID")
  if [[ -n "$PUBLISH_WEBSITE_ROOT" ]]; then
    publish_args+=(--website-root "$PUBLISH_WEBSITE_ROOT")
  fi
  "$PYTHON" "$ROOT/benchctl.py" "${publish_args[@]}"

  public_args=(
    publish
    --run-id "$RUN_ID"
    --model-set community-dashboard
    --publish-root "$ROOT/publish/community-dashboard"
  )
  if [[ -n "$PUBLIC_WEBSITE_ROOT" ]]; then
    public_args+=(--website-root "$PUBLIC_WEBSITE_ROOT")
  fi
  "$PYTHON" "$ROOT/benchctl.py" "${public_args[@]}"
fi
