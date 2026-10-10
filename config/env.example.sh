#!/usr/bin/env bash

# Copy this file to config/env.local.sh and adjust only the paths that differ
# on your machine. env.local.sh is ignored by Git.
BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

export TORCHBENCH_DIR="${TORCHBENCH_DIR:-$BENCHMARK_ROOT/third_party/benchmarks/torchbench}"
export TORCHBENCH_DATA_PATH="${TORCHBENCH_DATA_PATH:-$BENCHMARK_ROOT/data/torchbenchmark/.data}"
export CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.2.0}"
export BENCHMARK_RUNS_DIR="${BENCHMARK_RUNS_DIR:-$BENCHMARK_ROOT/runs}"
export HF_HOME="${HF_HOME:-$BENCHMARK_ROOT/cache/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BENCHMARK_ROOT/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$BENCHMARK_ROOT/cache/torch}"

# Leave empty when the current shell is already inside the desired conda env.
export BENCHMARK_ENV="${BENCHMARK_ENV:-}"

# Used only by scripts/run_daily_container.sh when launching from the host.
export BENCHMARK_CONTAINER="${BENCHMARK_CONTAINER:-ascendforge_ci}"
export BENCHMARK_CONTAINER_WORKSPACE="${BENCHMARK_CONTAINER_WORKSPACE:-$BENCHMARK_ROOT}"
export BENCHMARK_CONTAINER_ENV="${BENCHMARK_CONTAINER_ENV:-$BENCHMARK_ENV}"
export BENCHMARK_CONTAINER_CANN_HOME="${BENCHMARK_CONTAINER_CANN_HOME:-$CANN_HOME}"
