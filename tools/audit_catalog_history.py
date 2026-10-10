#!/usr/bin/env python3
"""Audit catalog coverage and timeout evidence in historical BenchBoard data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchpipe.catalog import Catalog  # noqa: E402


SUITE_ALIASES = {
    "bench": "torchbench",
    "benchmark": "torchbench",
    "torchbench": "torchbench",
    "huggingface": "huggingface",
    "timm": "timm",
}
BACKENDS = ("dvm", "triton_experimental")
TIMEOUT_PATTERNS = (
    re.compile(r"\[timeout\]", re.IGNORECASE),
    re.compile(r"timed out", re.IGNORECASE),
    re.compile(r"timeout(?:expired| after)", re.IGNORECASE),
    re.compile(r"(?:exit|return)(?: code)?\s*[:=]?\s*124\b", re.IGNORECASE),
    re.compile(r"terminated by signal\s+(?:9|15)\b", re.IGNORECASE),
)
KILL_PATTERNS = (
    re.compile(r"\bkilled\b", re.IGNORECASE),
    re.compile(r"(?:exit|return)(?: code)?\s*[:=]?\s*137\b", re.IGNORECASE),
    re.compile(r"out of memory|\boom\b", re.IGNORECASE),
)


def normalize_suite(value: str) -> str:
    return SUITE_ALIASES.get(value.strip().lower(), value.strip().lower())


def has_value(value: Any) -> bool:
    return value not in (None, "", "null")


@dataclass
class History:
    dates: set[str] = field(default_factory=set)
    backends: set[str] = field(default_factory=set)
    records: int = 0
    eager_dates: set[str] = field(default_factory=set)
    compile_dates: set[str] = field(default_factory=set)
    pass_dates: set[str] = field(default_factory=set)
    timeout_dates: set[str] = field(default_factory=set)
    latest: dict[str, dict[str, Any]] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "dates",
            "backends",
            "eager_dates",
            "compile_dates",
            "pass_dates",
            "timeout_dates",
        ):
            data[key] = sorted(data[key])
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default="daily-full")
    parser.add_argument("--min-date", default="20260901")
    parser.add_argument(
        "--web-data",
        type=Path,
        default=Path("board_file/web_data"),
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("board_file/raw_data"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "audits" / "catalog_history")
    return parser.parse_args()


def iter_json_files(root: Path, min_date: str):
    for date_dir in sorted(root.iterdir() if root.is_dir() else []):
        if not date_dir.is_dir() or not date_dir.name.isdigit() or date_dir.name < min_date:
            continue
        yield from sorted(date_dir.glob("*.json"))


def load_history(catalog_models, web_data: Path, min_date: str):
    by_key = {(model.suite, model.name.casefold()): model.model_id for model in catalog_models}
    history = {model.model_id: History() for model in catalog_models}
    unmatched = Counter()
    files = 0
    for path in iter_json_files(web_data, min_date):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        files += 1
        suite = normalize_suite(str(payload.get("type", "")))
        backend = str(payload.get("mode", ""))
        date = str(payload.get("date", path.parent.name))
        models = payload.get("models", {})
        if not isinstance(models, dict):
            continue
        for raw_name, record in models.items():
            if not isinstance(record, dict):
                continue
            name = str(record.get("model_name") or raw_name)
            model_id = by_key.get((suite, name.casefold()))
            if model_id is None:
                unmatched[f"{suite}/{name}"] += 1
                continue
            item = history[model_id]
            item.records += 1
            item.dates.add(date)
            if backend:
                item.backends.add(backend)
            if has_value(record.get("eager_E2E_avg_time")):
                item.eager_dates.add(date)
            if has_value(record.get("compile_E2E_avg_time")):
                item.compile_dates.add(date)
            accuracy = str(record.get("accuracy", "")).lower()
            status = str(record.get("status", "")).lower()
            if (accuracy and "pass" in accuracy) or (
                not accuracy and status in {"ok", "pass", "passed", "success"}
            ):
                item.pass_dates.add(date)
            if "timeout" in accuracy or "timeout" in status:
                item.timeout_dates.add(date)
            latest = item.latest.get(backend)
            if backend in BACKENDS and (latest is None or date >= str(latest.get("date", ""))):
                item.latest[backend] = {
                    "date": date,
                    "status": record.get("status"),
                    "accuracy": record.get("accuracy"),
                    "eager_E2E_avg_time": record.get("eager_E2E_avg_time"),
                    "compile_E2E_avg_time": record.get("compile_E2E_avg_time"),
                }
    return history, unmatched, files


def scan_log_evidence(raw_data: Path, min_date: str, selected_names: set[str]):
    events: list[dict[str, str]] = []
    files_scanned = 0
    for date_dir in sorted(raw_data.iterdir() if raw_data.is_dir() else []):
        if not date_dir.is_dir() or not date_dir.name.isdigit() or date_dir.name < min_date:
            continue
        for path in date_dir.glob("**/board_log/*.log"):
            if path.parent.parent.name.casefold() not in selected_names:
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            matches = []
            kind = ""
            for line in text.splitlines():
                if any(pattern.search(line) for pattern in TIMEOUT_PATTERNS):
                    matches.append(line.strip()[:400])
                    kind = "timeout"
                elif any(pattern.search(line) for pattern in KILL_PATTERNS):
                    matches.append(line.strip()[:400])
                    kind = kind or "killed_or_oom"
                if len(matches) >= 3:
                    break
            if matches:
                parts = path.parts
                backend = next((item for item in BACKENDS if item in parts), "unknown")
                events.append(
                    {
                        "date": date_dir.name,
                        "model": path.parent.parent.name,
                        "backend": backend,
                        "kind": kind,
                        "path": str(path),
                        "evidence": " | ".join(matches),
                    }
                )
    return events, files_scanned


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    catalog = Catalog(ROOT / "config")
    models = catalog.select(args.model_set)
    history, unmatched, json_files = load_history(models, args.web_data, args.min_date)
    events, log_files = scan_log_evidence(
        args.raw_data,
        args.min_date,
        {model.name.casefold() for model in models},
    )
    event_names = {event["model"].casefold() for event in events}
    for model in models:
        if model.name.casefold() in event_names:
            for event in events:
                if event["model"].casefold() == model.name.casefold() and event["kind"] == "timeout":
                    history[model.model_id].timeout_dates.add(event["date"])

    def ids(predicate):
        return [model.model_id for model in models if predicate(history[model.model_id])]

    seen = ids(lambda item: item.records > 0)
    eager = ids(lambda item: bool(item.eager_dates))
    compiled = ids(lambda item: bool(item.compile_dates))
    passed = ids(lambda item: bool(item.pass_dates))
    timeout = ids(lambda item: bool(item.timeout_dates))
    latest_backend: dict[str, Any] = {}
    for backend in BACKENDS:
        latest_backend[backend] = {
            "present": ids(lambda item, b=backend: b in item.latest),
            "eager": ids(
                lambda item, b=backend: b in item.latest
                and has_value(item.latest[b].get("eager_E2E_avg_time"))
            ),
            "compile": ids(
                lambda item, b=backend: b in item.latest
                and has_value(item.latest[b].get("compile_E2E_avg_time"))
            ),
        }
    return {
        "model_set": args.model_set,
        "min_date": args.min_date,
        "summary": {
            "catalog_models": len(models),
            "json_files_scanned": json_files,
            "log_files_scanned": log_files,
            "ever_seen": len(seen),
            "ever_eager": len(eager),
            "ever_compile": len(compiled),
            "ever_pass": len(passed),
            "timeout_models": len(timeout),
            "timeout_log_events": sum(event["kind"] == "timeout" for event in events),
            "kill_or_oom_log_events": sum(event["kind"] == "killed_or_oom" for event in events),
        },
        "never_seen": sorted(set(model.model_id for model in models) - set(seen)),
        "never_eager": sorted(set(model.model_id for model in models) - set(eager)),
        "never_compile": sorted(set(model.model_id for model in models) - set(compiled)),
        "timeout_models": timeout,
        "latest_backend": latest_backend,
        "log_events": events,
        "models": {model.model_id: history[model.model_id].serializable() for model in models},
        "unmatched_top": unmatched.most_common(50),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Daily-full 历史数据与超时审计",
        "",
        f"- 模型集合：`{report['model_set']}`",
        f"- 起始日期：`{report['min_date']}`",
        f"- Catalog 模型数：{summary['catalog_models']}",
        f"- 历史出现过：{summary['ever_seen']}",
        f"- 产出过 eager 时间：{summary['ever_eager']}",
        f"- 产出过 compile 时间：{summary['ever_compile']}",
        f"- 曾通过：{summary['ever_pass']}",
        f"- 有 timeout 证据的模型：{summary['timeout_models']}",
        f"- timeout 日志事件：{summary['timeout_log_events']}",
        f"- kill/OOM 日志事件：{summary['kill_or_oom_log_events']}",
        "",
    ]
    for title, key in (
        ("从未出现在历史 JSON", "never_seen"),
        ("从未产出 eager 时间", "never_eager"),
        ("从未产出 compile 时间", "never_compile"),
        ("存在 timeout 证据", "timeout_models"),
    ):
        lines.extend([f"## {title}", ""])
        values = report[key]
        lines.extend([f"- `{value}`" for value in values] or ["- 无"])
        lines.append("")
    lines.extend(["## 最新记录覆盖", ""])
    for backend, values in report["latest_backend"].items():
        lines.append(
            f"- `{backend}`：记录 {len(values['present'])}，eager {len(values['eager'])}，compile {len(values['compile'])}"
        )
    lines.extend(["", "## 超时与 kill/OOM 日志证据", ""])
    for event in report["log_events"]:
        lines.append(
            f"- `{event['date']}` `{event['backend']}` `{event['model']}` `{event['kind']}`：{event['evidence']}"
        )
    if not report["log_events"]:
        lines.append("- 无")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    md_path = output.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
