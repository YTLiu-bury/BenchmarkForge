from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import dashboard_aggregate, dashboard_processor, error_publish
from .planner import sanitize


SUITE_TO_RUN_TYPE = {
    "torchbench": "bench",
    "huggingface": "huggingface",
    "timm": "timm",
}

PROFILE_KEEP_FILES = (
    "step_trace_time.csv",
    "op_statistic.csv",
    "kernel_details.csv",
    "trace_view.json",
)

DASHBOARD_ASSETS = (
    "benchmark_dashboard.html",
    "dashboard-ex.html",
    "index.html",
    "board.json",
    "full_model.txt",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def infer_package_tag(run_metadata: dict[str, Any]) -> str:
    packages = run_metadata.get("packages")
    version = str(packages.get("torch_npu", "") if isinstance(packages, dict) else "")
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError("cannot infer package tag from torch_npu version; pass --package-tag")
    return "".join(match.groups())


def runtime_board_metadata(run_metadata: dict[str, Any]) -> dict[str, str]:
    packages = run_metadata.get("packages")
    packages = packages if isinstance(packages, dict) else {}
    paths = run_metadata.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    torch_npu_version = str(packages.get("torch_npu", ""))
    commit_match = re.search(r"\+git([0-9a-fA-F]+)", torch_npu_version)
    cann_home = Path(str(paths.get("cann_home", ""))).name
    cann_match = re.match(r"cann-(.+)", cann_home)
    return {
        "commit": commit_match.group(1) if commit_match else "",
        "torch_npu_version": torch_npu_version,
        "torch_version": str(packages.get("torch", "")),
        "torchvision_version": str(packages.get("torchvision", "")),
        "cann_version": cann_match.group(1) if cann_match else cann_home,
        "runner_repo_commit": str(run_metadata.get("runner_repo_commit", "")),
        "pytorch_benchmark_commit": str(run_metadata.get("pytorch_benchmark_commit", "")),
    }


def enrich_dashboard_metadata(
    website_root: Path,
    backend: str,
    run_date: str,
    metadata: dict[str, str],
) -> None:
    daily_path = website_root / backend / f"{run_date}.json"
    payload = load_json(daily_path)
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    by_run_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        run.update(metadata)
        by_run_id[str(run.get("run_id", ""))] = run
    daily_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = website_root / backend / "manifest.json"
    manifest = load_json(manifest_path)
    dates = manifest.get("dates") if isinstance(manifest.get("dates"), list) else []
    for date_entry in dates:
        if not isinstance(date_entry, dict) or str(date_entry.get("date", "")) != run_date:
            continue
        summaries = date_entry.get("runs") if isinstance(date_entry.get("runs"), list) else []
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            run = by_run_id.get(str(summary.get("run_id", "")))
            if run is not None:
                summary.update(metadata)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project_daily_payload(payload: dict[str, Any], model_names: set[str]) -> dict[str, Any]:
    """Filter one historical daily payload to a dashboard model projection."""
    projected = deepcopy(payload)
    projected_runs: list[dict[str, Any]] = []
    for run in projected.get("runs", []):
        if not isinstance(run, dict):
            continue
        models = run.get("models")
        if not isinstance(models, dict):
            continue
        run["models"] = {name: value for name, value in models.items() if name in model_names}
        run["model_count"] = len(run["models"])
        if run["models"]:
            projected_runs.append(run)
    projected["runs"] = projected_runs
    projected["run_count"] = len(projected_runs)
    return projected


def seed_dashboard_history(
    history_root: Path,
    website_root: Path,
    backends: list[str],
    model_names: set[str] | None,
) -> list[str]:
    """Seed a standalone website root from existing daily JSON files."""
    if history_root.resolve() == website_root.resolve():
        return []
    seeded: list[str] = []
    website_root.mkdir(parents=True, exist_ok=True)
    for name in DASHBOARD_ASSETS:
        source = history_root / name
        target = website_root / name
        if not source.is_file():
            continue
        if not target.is_file() or source.stat().st_size != target.stat().st_size:
            shutil.copy2(source, target)
        seeded.append(str(target))
    for backend in backends:
        source_dir = history_root / backend
        target_dir = website_root / backend
        if not source_dir.is_dir():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.glob("*.json")):
            if source.name == "manifest.json":
                continue
            target = target_dir / source.name
            if model_names is None:
                shutil.copy2(source, target)
            else:
                payload = project_daily_payload(load_json(source), model_names)
                target.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            seeded.append(str(target))
    return seeded


def reset_stage(stage_root: Path, run_dir: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_stage = stage_root.resolve()
    try:
        resolved_stage.relative_to(resolved_run)
    except ValueError as error:
        raise ValueError(f"refusing to reset stage outside run directory: {resolved_stage}") from error
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)


def stage_profile_files(profile_source: Path, profile_target: Path) -> set[str]:
    """Flatten one case's nested profiler output into the dashboard layout."""
    staged_phases: set[str] = set()
    if not profile_source.is_dir():
        return staged_phases

    for phase in ("eager", "compile"):
        phase_target = profile_target / phase
        for file_name in PROFILE_KEEP_FILES:
            candidates = []
            for candidate in profile_source.rglob(file_name):
                try:
                    relative_parts = candidate.relative_to(profile_source).parts
                except ValueError:
                    continue
                if phase in relative_parts and candidate.is_file() and candidate.stat().st_size > 0:
                    candidates.append(candidate)
            if not candidates:
                continue
            source = max(
                candidates,
                key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
            )
            phase_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, phase_target / file_name)
            if file_name == "step_trace_time.csv":
                staged_phases.add(phase)
    return staged_phases


def stage_case(case: dict[str, Any], backend_dir: Path) -> tuple[bool, set[str]]:
    log_path = Path(str(case.get("log", "")))
    if not log_path.is_file():
        return False, set()
    model_key = sanitize(str(case.get("model_id") or case.get("model") or case.get("case_id")))
    model_dir = backend_dir / model_key
    log_dir = model_dir / "board_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_path, log_dir / f"{case['case_id']}.log")

    profile_source = Path(str(case.get("profile", "")))
    profile_target = model_dir / "profile"
    staged_phases = stage_profile_files(profile_source, profile_target)
    return True, staged_phases


def canonicalize_run_models(
    run: dict[str, Any], model_aliases: dict[str, str]
) -> dict[str, Any]:
    """Restore catalog model names for logs that fail before printing a model name."""
    models = run.get("models")
    if not isinstance(models, dict):
        return run

    canonical_models: dict[str, Any] = {}
    for raw_name, raw_entry in models.items():
        canonical_name = model_aliases.get(str(raw_name), str(raw_name))
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else raw_entry
        if isinstance(entry, dict):
            entry["model_name"] = canonical_name
        existing = canonical_models.get(canonical_name)
        if isinstance(existing, dict) and isinstance(entry, dict):
            for key, value in entry.items():
                if existing.get(key) in (None, "", "missing") and value not in (None, ""):
                    existing[key] = value
        else:
            canonical_models[canonical_name] = entry
    run["models"] = dict(sorted(canonical_models.items()))
    return run


def extension_for_dynamic(base_extension: str, dynamic: str) -> str:
    if dynamic != "on":
        return base_extension
    if not base_extension or base_extension == "default":
        return "dynamic"
    tokens = base_extension.split("+")
    return base_extension if "dynamic" in tokens else f"{base_extension}+dynamic"


def aggregate_backend(
    web_data_root: Path,
    website_root: Path,
    run_date: str,
    backend: str,
) -> None:
    dashboard_aggregate.write_backend_payload(
        web_data_dir=web_data_root,
        output_root=website_root,
        logical_date=run_date,
        backend=backend,
        include_dates=[run_date],
    )


def publish_board_data(
    plan: dict[str, Any],
    run_dir: Path,
    *,
    publish_root: Path,
    website_root: Path,
    package_tag: str | None,
    extension: str,
    selected_backends: list[str] | None,
    selected_model_ids: set[str] | None,
    projection: str,
    history_root: Path | None,
) -> dict[str, Any]:
    run_metadata = load_json(run_dir / "run.json")
    run_date = str(plan.get("run_date") or "").strip()
    if not re.fullmatch(r"\d{8}", run_date):
        raise ValueError(f"run has invalid date: {run_date!r}")
    package_tag = package_tag or infer_package_tag(run_metadata)
    web_data_root = publish_root / "web_data"
    stage_root = run_dir / "board-stage"
    board_root = stage_root / "board_file"
    reset_stage(stage_root, run_dir)
    dashboard_processor.configure_paths(board_root, web_data_root)
    board_metadata = runtime_board_metadata(run_metadata)

    available_backends = [str(item) for item in plan.get("backends", [])]
    backends = selected_backends or available_backends
    unknown = sorted(set(backends) - set(available_backends))
    if unknown:
        raise ValueError(f"backend not present in run: {', '.join(unknown)}")

    selected_model_names: set[str] | None = None
    if selected_model_ids is not None:
        selected_model_names = {
            str(case.get("model", ""))
            for case in plan.get("cases", [])
            if str(case.get("model_id", "")) in selected_model_ids
        }
    seeded_history = (
        seed_dashboard_history(history_root, website_root, backends, selected_model_names)
        if history_root is not None
        else []
    )

    generated_runs: list[str] = []
    staged_cases = 0
    profile_counts = {"complete": 0, "partial": 0, "missing": 0}
    for backend in backends:
        dynamic_modes = [str(item) for item in plan.get("dynamic_modes", ["config"])]
        for dynamic in dynamic_modes:
            run_extension = extension_for_dynamic(extension, dynamic)
            for suite, run_type in SUITE_TO_RUN_TYPE.items():
                cases = [
                    case
                    for case in plan.get("cases", [])
                    if case.get("backend") == backend
                    and case.get("suite") == suite
                    and str(case.get("dynamic", "config")) == dynamic
                    and (
                        selected_model_ids is None
                        or str(case.get("model_id", "")) in selected_model_ids
                    )
                ]
                if not cases:
                    continue
                layout = dashboard_processor.RawDataLayout.create(
                    run_type, backend, run_date, package_tag, run_extension
                )
                model_aliases: dict[str, str] = {}
                for case in cases:
                    staged, profile_phases = stage_case(case, layout.backend_dir)
                    if staged:
                        alias = sanitize(
                            str(case.get("model_id") or case.get("model") or case.get("case_id"))
                        )
                        model_aliases[alias] = str(case.get("model") or alias)
                        staged_cases += 1
                        if profile_phases == {"eager", "compile"}:
                            profile_counts["complete"] += 1
                        elif profile_phases:
                            profile_counts["partial"] += 1
                        else:
                            profile_counts["missing"] += 1
                if not layout.iter_model_dirs():
                    continue
                generated_run = dashboard_processor.RawDataProcessor(layout).process()
                generated_run = canonicalize_run_models(generated_run, model_aliases)
                layout.output_file.write_text(
                    json.dumps(generated_run, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                generated_runs.append(str(layout.output_file))
        aggregate_backend(web_data_root, website_root, run_date, backend)
        enrich_dashboard_metadata(website_root, backend, run_date, board_metadata)

    error_publish.publish(run_dir, website_root, run_date)

    result = {
        "run_id": plan.get("run_id", ""),
        "date": run_date,
        "package_tag": package_tag,
        "extension": extension,
        "projection": projection,
        "projected_model_count": len(
            {
                str(case.get("model_id", ""))
                for case in plan.get("cases", [])
                if selected_model_ids is None
                or str(case.get("model_id", "")) in selected_model_ids
            }
        ),
        "backends": backends,
        "metadata": board_metadata,
        "staged_cases": staged_cases,
        "profile_counts": profile_counts,
        "generated_runs": generated_runs,
        "history_root": str(history_root) if history_root is not None else "",
        "seeded_history": seeded_history,
        "web_data_root": str(web_data_root),
        "website_root": str(website_root),
        "daily_payloads": [str(website_root / backend / f"{run_date}.json") for backend in backends],
        "manifests": [str(website_root / backend / "manifest.json") for backend in backends],
    }
    report_name = "board-publish.json" if projection == "all" else f"board-publish-{sanitize(projection)}.json"
    output = run_dir / report_name
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
