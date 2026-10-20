#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# On the 173 host, re-enter the known-good container before loading runtime
# variables. Inside the container this branch is skipped.
if [[ ! -f /.dockerenv && "${BENCHMARK_CONTAINER_REEXEC:-1}" == "1" ]]; then
  exec bash "$ROOT/scripts/run_daily_container.sh" "$@"
fi

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

PYTHON="$BENCHMARK_PYTHON"
DATE="${DATE:-$(date +%Y%m%d)}"
DVM_DEVICES="${DVM_DEVICES:-4,5}"
TRITON_DEVICES="${TRITON_DEVICES:-1,2}"
SET_NAME="${SET_NAME:-daily-40}"
DYNAMIC_MODES="${DYNAMIC_MODES:-off}"
DYNAMIC_LABEL="${DYNAMIC_MODES//,/-}"
RUN_ID="${RUN_ID:-${DATE}-${SET_NAME}-dvm-triton-experimental-dynamic-${DYNAMIC_LABEL}}"
AUTO_PUBLISH="${AUTO_PUBLISH:-1}"
VALIDATE_OUTPUTS="${VALIDATE_OUTPUTS:-1}"
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

if [[ "$VALIDATE_OUTPUTS" == "1" ]]; then
  "$PYTHON" "$ROOT/scripts/validate_run_outputs.py" \
    --root "$ROOT" --run-id "$RUN_ID" --stage run
fi

if [[ "$AUTO_PUBLISH" == "1" ]]; then
  publish_args=(publish --run-id "$RUN_ID")
  if [[ -n "$PUBLISH_WEBSITE_ROOT" ]]; then
    publish_args+=(--website-root "$PUBLISH_WEBSITE_ROOT")
  fi
  "$PYTHON" "$ROOT/benchctl.py" "${publish_args[@]}"
  if [[ "$VALIDATE_OUTPUTS" == "1" ]]; then
    "$PYTHON" "$ROOT/scripts/validate_run_outputs.py" \
      --root "$ROOT" --run-id "$RUN_ID" --stage publish
  fi

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
  if [[ "$VALIDATE_OUTPUTS" == "1" ]]; then
    "$PYTHON" "$ROOT/scripts/validate_run_outputs.py" \
      --root "$ROOT" --publish-root "$ROOT/publish/community-dashboard" \
      --run-id "$RUN_ID" --stage publish
  fi
fi
