# NPU Benchmark Pipeline

This workspace is the migration sandbox for the Ascend PyTorch benchmark pipeline. It reuses the existing
`benchmarks/torchbench` runners and does not patch the active Python environment.

## Design boundaries

- `benchctl.py` is the only user-facing entry point.
- `config/models.yaml` is generated once from the current BenchBoard configs and becomes the local catalog.
- `model_sets.yaml` defines views (`daily-full`, `community-dashboard`, `smoke`) without duplicating model policy.
- `daily-full` is the union of internal daily models and all community-dashboard models.
- Every run writes an immutable `plan.json`, environment `run.json`, per-case logs/results and aggregate CSV/TSV.
- One worker owns one NPU. Models assigned to the same NPU run serially.
- `rerun` only selects cases that have not passed.

## Bootstrap catalog

Run inside `ascendforge_ci`:

```bash
cd "$BENCHMARK_ROOT"
"$BENCHMARK_PYTHON" tools/import_legacy_config.py \
  --daily /path/to/legacy/auto_board/run.yaml \
  --community config/seeds/community_106.yaml \
  --output config/models.yaml
```

`config/models.yaml` is the versioned catalog after this one-time import. Daily execution does not read the legacy
BenchBoard YAML files.

## Inspect without running

```bash
python benchctl.py list --set daily-full
python benchctl.py list --set community-dashboard
python benchctl.py doctor --set smoke

python benchctl.py plan \
  --set smoke \
  --backend triton_experimental \
  --devices 7 \
  --run-id smoke-plan
```

## Run

```bash
bash scripts/run_single_model.sh torchbench/alexnet triton_experimental 7

# Minimal accuracy smoke without profiler output
python benchctl.py run --set smoke --model torchbench/alexnet \
  --backend triton_experimental --devices 7 --iterations 1 --profile off

DATE=20260917 DVM_DEVICES=4,5 TRITON_DEVICES=1,2 \
  bash scripts/run_daily.sh
```

The daily command runs both backends from the same resolved model catalog. It refuses to assign one physical NPU to
two backends in the same run. By default every model runs with both dynamic shapes disabled and enabled. Cases on one
NPU remain serial, while different NPUs run concurrently. Set `DYNAMIC_MODES=off` or `DYNAMIC_MODES=on` to select only
one mode. After execution it publishes isolated dashboard data under `publish/website` by default. Set
`AUTO_PUBLISH=0` to skip that stage.

## Resume and export

```bash
python benchctl.py status --run-id <run-id>
python benchctl.py rerun --run-id <run-id>
python benchctl.py export --run-id <run-id>
```

## Publish dashboard data

The dashboard does not read `results.csv` directly. `publish` reuses BenchBoard's existing
`process_raw_data.py` and `aggregate_backend_daily_data.py`, so E2E and OP metrics keep the same schema and
calculation rules as the current website.

```bash
# Build dashboard data in this isolated workspace first.
python benchctl.py publish --run-id <run-id>

# After verification, write the daily payloads into the BenchBoard website tree.
python benchctl.py publish --run-id <run-id> \
  --website-root /path/to/website

# Reuse the same run but expose only the public community-aligned model set.
python benchctl.py publish --run-id <run-id> \
  --model-set community-dashboard \
  --publish-root publish/community-dashboard \
  --website-root /path/to/public/website
```

To publish there automatically after a daily run:

```bash
PUBLISH_WEBSITE_ROOT=/path/to/website \
  DATE=20260917 DVM_DEVICES=4,5 TRITON_DEVICES=1,2 \
  bash scripts/run_daily.sh
```

The isolated output is:

```text
publish/web_data/<date>/<run>.json
publish/website/dvm/<date>.json
publish/website/dvm/manifest.json
publish/website/triton_experimental/<date>.json
publish/website/triton_experimental/manifest.json
```

Static-shape runs keep extension `default`; dynamic-shape runs use extension `dynamic`, so the dashboard can switch
or compare them without merging their measurements.

Publishing requires a run executed with profiler enabled to populate `eager_OP_avg_time`,
`compile_OP_avg_time`, and `OP_speed_up_rate`. Runs with `--profile off` still publish accuracy and E2E data when
the benchmark log contains the corresponding summaries.

## Output contract

```text
runs/<run-id>/
├── plan.json
├── run.json
├── status.tsv
├── results.csv
├── summary.json
├── case-results/
├── logs/
├── raw/
├── profile/
├── cache/
└── compile-debug/
```

`plan.json` is the exact resolved command matrix. Results never infer the intended model list from whatever logs happen
to exist.
