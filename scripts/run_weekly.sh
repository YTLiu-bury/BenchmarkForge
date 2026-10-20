#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATE="${DATE:-$(date +%Y%m%d)}"
SET_NAME="${SET_NAME:-weekly-full}"
DYNAMIC_MODES="${DYNAMIC_MODES:-off}"
DYNAMIC_LABEL="${DYNAMIC_MODES//,/-}"
RUN_ID="${RUN_ID:-${DATE}-${SET_NAME}-dvm-triton-experimental-dynamic-${DYNAMIC_LABEL}}"

export DATE SET_NAME DYNAMIC_MODES RUN_ID
exec bash "$ROOT/scripts/run_daily.sh" "$@"
