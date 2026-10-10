#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEB_DATA_DIR = PIPELINE_ROOT / "publish" / "web_data"
FALLBACK_WEB_DATA_DIR = DEFAULT_WEB_DATA_DIR
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "publish" / "website"
COMMIT_URL_PATH = PIPELINE_ROOT / "config" / "commit_url.json"
CANN_URL_PATH = PIPELINE_ROOT / "config" / "cann_url.json"
PYTORCH_PR_BASE_URL = "https://gitcode.com/Ascend/pytorch/pull"
WATCH_BACKENDS = {"dvm", "triton_experimental"}


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_date(value: str) -> str:
    date = str(value or "").strip()
    if date.lower() == "today":
        return datetime.now().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date must be today or YYYYMMDD: {value}")
    return date


def normalize_backend(value: str) -> str:
    backend = str(value or "").strip()
    if backend not in WATCH_BACKENDS:
        allowed = ", ".join(sorted(WATCH_BACKENDS))
        raise ValueError(f"backend must be one of: {allowed}")
    return backend


def normalize_mode_label(value: Any) -> str:
    return str(value or "").strip().replace("+aclgraph", "")


def normalize_cann_version(raw_name: str) -> str:
    text = str(raw_name or "").strip()
    return text[4:] if text.lower().startswith("cann") else text


def resolve_cann_info(run_date: str, cann_info: dict[str, Any]) -> dict[str, str]:
    date = str(run_date or "").strip()
    if not date or not isinstance(cann_info, dict):
        return {}

    candidates = [
        str(item).strip()
        for item in cann_info.keys()
        if str(item).strip() and str(item).strip() <= date
    ]
    if not candidates:
        return {}

    selected_date = max(candidates)
    versions = cann_info.get(selected_date)
    if not isinstance(versions, dict) or not versions:
        return {}

    cann_keys = [
        str(item)
        for item in versions.keys()
        if str(item).lower().startswith("cann") and isinstance(versions.get(item), dict)
    ]
    if not cann_keys:
        return {}

    cann_name = sorted(cann_keys)[-1]
    payload = versions.get(cann_name, {})
    download_url = str(payload.get("download_url") or "").strip() if isinstance(payload, dict) else ""
    version = str(payload.get("version") or "").strip() if isinstance(payload, dict) else ""
    innerversion = str(payload.get("innerversion") or "").strip() if isinstance(payload, dict) else ""
    return {
        "cann_date": selected_date,
        "cann_name": str(cann_name),
        "cann_version": version or normalize_cann_version(str(cann_name)),
        "cann_innerversion": innerversion,
        "cann_download_url": download_url,
        "akg": str(versions.get("akg") or "").strip(),
        "mfusion": str(versions.get("mfusion") or "").strip(),
    }


def resolve_commit_info(run_date: str, pta_version: str, commit_info: dict[str, Any]) -> dict[str, str]:
    date = str(run_date or "").strip()
    version = str(pta_version or "").strip()
    if not date or not version or not isinstance(commit_info, dict):
        return {"pr_id": "", "commit": "", "pr_url": ""}

    day_info = commit_info.get(date)
    info = day_info.get(version) if isinstance(day_info, dict) else {}
    if not isinstance(info, dict):
        info = {}

    pr_id = str(info.get("pr") or info.get("pr_id") or "").strip()
    commit = str(info.get("commit") or "").strip()
    pr_url = str(info.get("pr_url") or "").strip()
    if not pr_url and pr_id:
        pr_url = f"{PYTORCH_PR_BASE_URL}/{pr_id}"
    return {"pr_id": pr_id, "commit": commit, "pr_url": pr_url}


def rewrite_run_id(run_id: Any, logical_date: str) -> str:
    text = str(run_id or "").strip()
    if re.match(r"^\d{8}_", text):
        return f"{logical_date}_{text[9:]}"
    return text


def normalize_run(
    run: dict[str, Any],
    logical_date: str,
    source_path: Path,
    commit_info: dict[str, Any],
    cann_info: dict[str, Any],
) -> dict[str, Any]:
    source_date = str(run.get("date") or source_path.parent.name or logical_date).strip()
    normalized = copy.deepcopy(run)
    pta_version = str(normalized.get("pta_version") or "").strip()
    normalized["date"] = logical_date
    normalized["logical_date"] = logical_date
    normalized["source_date"] = source_date
    normalized["mode"] = normalize_mode_label(normalized.get("mode"))
    normalized["mode_label"] = normalize_mode_label(normalized.get("mode_label")) or normalized["mode"]
    normalized["extension"] = str(normalized.get("extension") or "default")
    normalized["run_id"] = rewrite_run_id(normalized.get("run_id"), logical_date)
    normalized["source_file"] = str(source_path)
    normalized.update(resolve_commit_info(source_date, pta_version, commit_info))
    normalized.update(resolve_cann_info(source_date, cann_info))

    models = normalized.get("models") if isinstance(normalized.get("models"), dict) else {}
    normalized["models"] = models
    for entry in models.values():
        if not isinstance(entry, dict):
            continue
        entry["date"] = str(entry.get("date") or source_date or logical_date)
        entry["logical_date"] = logical_date
        entry["source_date"] = source_date
        entry["mode"] = normalize_mode_label(entry.get("mode") or normalized["mode"])
        entry["extension"] = str(entry.get("extension") or normalized["extension"])
    normalized["model_count"] = len(models)
    return normalized


def is_missing_model(entry: Any) -> bool:
    return not isinstance(entry, dict) or entry.get("status") == "missing"


def should_replace_model(existing: Any, incoming: Any) -> bool:
    existing_missing = is_missing_model(existing)
    incoming_missing = is_missing_model(incoming)
    if incoming_missing and not existing_missing:
        return False
    return True


def merge_duplicate_run(existing: dict[str, Any], incoming: dict[str, Any], source_path: Path) -> dict[str, Any]:
    merged = copy.deepcopy(existing)
    existing_source = str(existing.get("source_file") or "").strip()
    incoming_source = str(incoming.get("source_file") or source_path).strip()
    existing_source_date = str(existing.get("source_date") or "").strip()
    incoming_source_date = str(incoming.get("source_date") or "").strip()
    existing_models = merged.get("models") if isinstance(merged.get("models"), dict) else {}
    incoming_models = incoming.get("models") if isinstance(incoming.get("models"), dict) else {}

    merged.update({key: value for key, value in incoming.items() if key != "models"})
    for model_name, incoming_entry in incoming_models.items():
        existing_entry = existing_models.get(model_name)
        if model_name not in existing_models or should_replace_model(existing_entry, incoming_entry):
            existing_models[model_name] = incoming_entry
    merged["models"] = existing_models
    merged["model_count"] = len(existing_models)

    source_files = merged.get("source_files")
    if not isinstance(source_files, list):
        source_files = []
    for source in [existing_source, incoming_source]:
        if source and source not in source_files:
            source_files.append(source)
    merged["source_files"] = source_files

    source_dates = merged.get("source_dates")
    if not isinstance(source_dates, list):
        source_dates = []
    for date in [existing_source_date, incoming_source_date]:
        if date and date not in source_dates:
            source_dates.append(date)
    merged["source_dates"] = source_dates
    return merged


def is_backend_run(run: dict[str, Any], path: Path, backend: str) -> bool:
    mode = normalize_mode_label(run.get("mode_label") or run.get("mode"))
    return mode == backend or backend in path.name


def collect_runs(
    web_data_dir: Path,
    logical_date: str,
    include_dates: list[str],
    backend: str,
    commit_info: dict[str, Any],
    cann_info: dict[str, Any],
) -> list[dict[str, Any]]:
    runs_by_key: dict[str, dict[str, Any]] = {}
    for physical_date in include_dates:
        date_dir = web_data_dir / physical_date
        if not date_dir.is_dir():
            print(f"[WARN] skip missing date dir: {date_dir}")
            continue

        for path in sorted(date_dir.glob("*.json")):
            if path.name.endswith("_board_data.json"):
                continue
            data = load_json(path, {})
            if not isinstance(data, dict):
                print(f"[WARN] skip non-object json: {path}")
                continue
            if not is_backend_run(data, path, backend):
                continue
            run = normalize_run(data, logical_date, path, commit_info, cann_info)
            key = "|".join(
                [
                    str(run.get("run_id") or ""),
                    str(run.get("type") or ""),
                    str(run.get("mode") or ""),
                    str(run.get("extension") or ""),
                    str(run.get("pta_version") or ""),
                ]
            )
            if key in runs_by_key:
                print(f"[INFO] merge duplicate logical run with later source: {path}")
                runs_by_key[key] = merge_duplicate_run(runs_by_key[key], run, path)
            else:
                runs_by_key[key] = run
    return list(runs_by_key.values())


def build_daily_payload(web_data_dir: Path, logical_date: str, include_dates: list[str], backend: str) -> dict[str, Any]:
    commit_info = load_json(COMMIT_URL_PATH, {})
    cann_info = load_json(CANN_URL_PATH, {})
    runs = collect_runs(web_data_dir, logical_date, include_dates, backend, commit_info, cann_info)
    return {
        "date": logical_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dates": include_dates,
        "backend": backend,
        "runs": runs,
    }


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id", ""),
        "date": run.get("date", ""),
        "type": run.get("type", ""),
        "mode": run.get("mode", ""),
        "mode_label": run.get("mode_label", ""),
        "extension": run.get("extension", "default"),
        "pta_version": run.get("pta_version", ""),
        "pr_id": run.get("pr_id", ""),
        "commit": run.get("commit", ""),
        "pr_url": run.get("pr_url", ""),
        "cann_version": run.get("cann_version", ""),
        "cann_innerversion": run.get("cann_innerversion", ""),
        "hardware": run.get("hardware", ""),
        "model_count": run.get("model_count", len(run.get("models", {}) if isinstance(run.get("models"), dict) else {})),
        "logical_date": run.get("logical_date", run.get("date", "")),
        "source_date": run.get("source_date", ""),
        "source_dates": run.get("source_dates", []),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def rebuild_manifest(output_dir: Path, backend: str) -> dict[str, Any]:
    dates: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = load_json(path, {})
        if not isinstance(payload, dict):
            continue
        date = str(payload.get("date") or path.stem).strip()
        runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
        dates.append(
            {
                "date": date,
                "path": path.name,
                "run_count": len(runs),
                "runs": [run_summary(run) for run in runs if isinstance(run, dict)],
            }
        )

    dates.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "dates": dates,
    }


def write_backend_payload(
    web_data_dir: Path,
    output_root: Path,
    logical_date: str,
    backend: str,
    include_dates: list[str] | None = None,
) -> tuple[Path, Path]:
    """Build one backend's daily payload and manifest without invoking scripts."""
    logical_date = resolve_date(logical_date)
    backend = normalize_backend(backend)
    dates = [resolve_date(item) for item in (include_dates or [logical_date])]
    payload = build_daily_payload(web_data_dir, logical_date, dates, backend)
    output_dir = output_root / backend
    daily_path = output_dir / f"{logical_date}.json"
    manifest_path = output_dir / "manifest.json"
    write_json(daily_path, payload)
    write_json(manifest_path, rebuild_manifest(output_dir, backend))
    return daily_path, manifest_path


def parse_args() -> argparse.Namespace:
    default_data_dir = DEFAULT_WEB_DATA_DIR if DEFAULT_WEB_DATA_DIR.exists() else FALLBACK_WEB_DATA_DIR
    parser = argparse.ArgumentParser(description="Build lazy-loaded backend board data.")
    parser.add_argument("--backend", required=True, help="backend to aggregate: dvm or triton_experimental")
    parser.add_argument("--web-data-dir", default=str(default_data_dir), help="raw web_data root, usually board_file/web_data")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="website root containing backend subdirs")
    parser.add_argument("--output-dir", help="explicit backend output dir; defaults to <output-root>/<backend>")
    parser.add_argument("--logical-date", required=True, help="date shown in the board, today or YYYYMMDD")
    parser.add_argument(
        "--include-date",
        action="append",
        dest="include_dates",
        help="physical web_data date dir to include; can be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = normalize_backend(args.backend)
    logical_date = resolve_date(args.logical_date)
    include_dates = [resolve_date(item) for item in (args.include_dates or [logical_date])]
    web_data_dir = Path(args.web_data_dir).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = Path(args.output_root).expanduser().resolve() / backend

    payload = build_daily_payload(web_data_dir, logical_date, include_dates, backend)
    daily_path = output_dir / f"{logical_date}.json"
    write_json(daily_path, payload)

    manifest = rebuild_manifest(output_dir, backend)
    write_json(output_dir / "manifest.json", manifest)

    print(
        f"generated {daily_path} with {len(payload['runs'])} {backend} runs "
        f"from {', '.join(include_dates)}"
    )
    print(f"updated {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
