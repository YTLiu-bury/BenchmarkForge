#!/usr/bin/env bash

# Sourced by launchers: remove inherited toolkit paths, then activate exactly
# one conda environment and one CANN installation. Driver paths are preserved.

clean_cann_path_var() {
  local name="$1"
  local value="${!name-}"
  local part
  local kept=()
  IFS=':' read -r -a parts <<< "$value"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    case "$part" in
      /usr/local/Ascend/cann-*|/usr/local/Ascend/ascend-toolkit/*|/home/*/cann/cann-*)
        continue
        ;;
    esac
    kept+=("$part")
  done
  local joined=""
  if ((${#kept[@]})); then
    joined="$(IFS=:; echo "${kept[*]}")"
  fi
  printf -v "$name" '%s' "$joined"
  export "$name"
}

RUNTIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

BENCHMARK_ENV="${BENCHMARK_ENV:-}"
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.2.0}"
export CANN_HOME
HF_HOME="${HF_HOME:-$RUNTIME_ROOT/cache/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RUNTIME_ROOT/cache/pip}"
TORCH_HOME="${TORCH_HOME:-$RUNTIME_ROOT/cache/torch}"
export HF_HOME PIP_CACHE_DIR TORCH_HOME
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME"
if [[ -n "$BENCHMARK_ENV" ]]; then
  if [[ ! -f "$BENCHMARK_ENV/bin/activate" ]]; then
    echo "missing benchmark environment: $BENCHMARK_ENV/bin/activate" >&2
    return 1
  fi
  set +u
  source "$BENCHMARK_ENV/bin/activate"
  set -u
fi
if [[ ! -f "$CANN_HOME/set_env.sh" ]]; then
  echo "missing CANN environment: $CANN_HOME/set_env.sh" >&2
  return 1
fi

for variable in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do
  clean_cann_path_var "$variable"
done
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME TOOLCHAIN_HOME TBE_IMPL_PATH

# CANN 9.2.0's set_env.sh may return non-zero on a container without the host
# install metadata file even after exporting the usable runtime paths. Disable
# errexit while sourcing it so that a metadata probe cannot abort the pipeline.
set +u
set +e
source "$CANN_HOME/set_env.sh"
cann_rc=$?
set -e
set -u
if (( cann_rc != 0 )); then
  echo "warning: CANN set_env.sh returned $cann_rc; continuing with exported runtime paths" >&2
fi

# Normalize toolkit paths when CANN was copied from another host.
if [[ "$CANN_HOME" != "/usr/local/Ascend/cann-9.2.0" ]]; then
  # The copied CANN package may retain the source host path in set_env.sh.
  # Remove those stale entries and rebuild the runtime paths from CANN_HOME.
  for variable in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do
    clean_cann_path_var "$variable"
  done
  export ASCEND_HOME_PATH="$CANN_HOME"
  export ASCEND_TOOLKIT_HOME="$CANN_HOME"
  export ASCEND_OPP_PATH="$CANN_HOME/opp"
  export ASCEND_AICPU_PATH="$CANN_HOME"

  prepend_existing_paths() {
    local variable="$1"
    shift
    local path
    local current="${!variable-}"
    for path in "$@"; do
      [[ -d "$path" ]] || continue
      current="$path${current:+:$current}"
    done
    printf -v "$variable" '%s' "$current"
    export "$variable"
  }

  prepend_existing_paths PATH \
    "$CANN_HOME/bin" \
    "$CANN_HOME/compiler/bin" \
    "$CANN_HOME/tools/profiler/bin" \
    "$CANN_HOME/tools/ascend_system_advisor/asys" \
    "$CANN_HOME/tools/show_kernel_debug_data" \
    "$CANN_HOME/tools/msobjdump" \
    "$CANN_HOME/aarch64-linux/bin"
  # prepend_existing_paths adds each item to the front, so list these in
  # reverse priority: toolkit libraries first, driver libraries last.
  prepend_existing_paths LD_LIBRARY_PATH \
    "/usr/local/Ascend/driver/lib64/driver" \
    "/usr/local/Ascend/driver/lib64/common" \
    "/usr/local/Ascend/driver/lib64" \
    "$CANN_HOME/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" \
    "$CANN_HOME/opp/lib64" \
    "$CANN_HOME/runtime/lib64" \
    "$CANN_HOME/fwkacllib/lib64" \
    "$CANN_HOME/aarch64-linux/devlib" \
    "$CANN_HOME/aarch64-linux/lib64/plugin/nnengine" \
    "$CANN_HOME/aarch64-linux/lib64/plugin/opskernel" \
    "$CANN_HOME/aarch64-linux/lib64"

else
  echo "using container-mounted CANN environment: $CANN_HOME" >&2
fi

if [[ -n "$BENCHMARK_ENV" ]]; then
  export BENCHMARK_PYTHON="$BENCHMARK_ENV/bin/python"
else
  BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-$(command -v python3 || command -v python || true)}"
  if [[ -z "$BENCHMARK_PYTHON" ]]; then
    echo "python executable not found; set BENCHMARK_ENV or BENCHMARK_PYTHON" >&2
    return 1
  fi
  export BENCHMARK_PYTHON
fi

for variable in PATH LD_LIBRARY_PATH PYTHONPATH; do
  value="${!variable-}"
  if [[ "$value" == *"cann-9.0.0"* ]]; then
    echo "mixed CANN environment detected in $variable" >&2
    return 1
  fi
done
