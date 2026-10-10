#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"traceback",
        r"exception",
        r"error",
        r"failed",
        r"fail",
        r"out of memory",
        r"timeout",
        r"timed? out",
    )
]

CATEGORY_PATTERNS = {
    "oom": (r"out of memory", r"outofmemory", r"memory allocation.*failed"),
    "timeout": (r"timeout", r"timed? out", r"time limit"),
    "missing_data": (r"no such file or directory", r"file not found", r"missing.*(data|file|asset)"),
    "accuracy": (r"fail_accuracy", r"allclose\s*=\s*false", r"accuracy.*(fail|mismatch|diff)", r"precision.*(fail|mismatch)"),
    "dependency": (r"modulenotfounderror", r"importerror", r"cannot import name", r"no module named"),
    "compile_error": (r"inductor", r"lowering", r"codegen", r"compile", r"notimplementederror", r"device .*not supported"),
    "network": (r"hugging ?face", r"download", r"connection", r"proxy", r"dns", r"max retries exceeded", r"http[s]?://"),
}


def sanitize_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?i)\b[A-Za-z]:\\[^\s\"']+", "<path>", text)
    text = re.sub(r"(?<![\w:])/(?:home|root|tmp|workspace|mnt|opt|usr|var|data)/[^\s\"']+", "<path>", text)
    text = re.sub(r"(?i)\b(?:l\d{6,}|nlucci)\b", "<user>", text)
    return text


def classify(text: str, status: str) -> str:
    if "accuracy" in status.lower() or "fail_accuracy" in text.lower():
        return "accuracy"
    order = ("oom", "timeout", "missing_data", "dependency", "compile_error", "network")
    for category in order:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in CATEGORY_PATTERNS[category]):
            return category
    return "runtime"


def excerpt(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    indexes = [index for index, line in enumerate(lines) if any(pattern.search(line) for pattern in ERROR_PATTERNS)]
    selected: set[int] = set()
    for index in indexes[-8:]:
        selected.update(range(max(0, index - 1), min(len(lines), index + 2)))
    value = "\n".join(lines[index] for index in sorted(selected)) if selected else "\n".join(lines[-20:])
    return sanitize_text(value)[:4000].strip()


def title(text: str, category: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r"(?:Error|Exception|Failed|Failure):", line, re.IGNORECASE):
            return sanitize_text(line)[:180]
    return {
        "network": "网络或模型下载失败",
        "missing_data": "数据或本地资源缺失",
        "oom": "内存不足（OOM）",
        "timeout": "模型运行超时",
        "accuracy": "精度校验失败",
        "compile_error": "编译失败",
        "dependency": "依赖缺失或版本不匹配",
    }.get(category, "模型运行失败")


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def package_tag(run: dict[str, Any]) -> str:
    value = str((run.get("packages") or {}).get("torch_npu") or "")
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    return "".join(match.groups()) if match else ""


def pta_version(tag: str) -> str:
    match = re.match(r"(\d)(\d)(\d+)$", tag)
    return f"{match.group(1)}.{match.group(2)}.{match.group(3)}" if match else tag


def run_id(date: str, tag: str, case: dict[str, Any]) -> str:
    backend = str(case.get("backend") or "")
    extension = "dynamic" if str(case.get("dynamic") or "off") == "on" else "default"
    suite = {"torchbench": "bench", "huggingface": "huggingface", "timm": "timm"}.get(str(case.get("suite") or ""), str(case.get("suite") or ""))
    return f"{date}_{pta_version(tag)}_{backend}_{extension}_{suite}"


def build_items(run_dir: Path, date: str) -> list[dict[str, Any]]:
    run = load_json(run_dir / "run.json", {})
    tag = package_tag(run)
    items: list[dict[str, Any]] = []
    for case_path in sorted((run_dir / "case-results").glob("*.json")):
        case = load_json(case_path, {})
        status = str(case.get("status") or "unknown")
        accuracy = str((case.get("metrics") or {}).get("accuracy") or "")
        if status in {"pass", "pass_accuracy", "ok"} and "fail" not in accuracy.lower():
            continue
        log_path = Path(str(case.get("log") or ""))
        text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
        category = classify(text, status)
        item = {
            "date": date,
            "hardware": "npu",
            "type": {"torchbench": "bench", "huggingface": "huggingface", "timm": "timm"}.get(str(case.get("suite") or ""), str(case.get("suite") or "")),
            "run_id": run_id(date, tag, case),
            "pta_version": pta_version(tag),
            "mode_label": str(case.get("backend") or ""),
            "extension": "dynamic" if str(case.get("dynamic") or "off") == "on" else "default",
            "model": str(case.get("model") or case.get("model_id") or case.get("case_id") or ""),
            "status": status,
            "error_category": category,
            "error_title": title(text, category),
            "error_md5": hashlib.md5(sanitize_text(text).encode("utf-8", errors="ignore")).hexdigest(),
            "error_excerpt": excerpt(text),
            "log_size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
        }
        items.append(item)
    return items


def publish(run_dir: Path, website_root: Path, date: str) -> None:
    errors_dir = website_root / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    items = build_items(run_dir, date)
    (errors_dir / f"{date}.json").write_text(
        json.dumps({"date": date, "generated_at": datetime.now(timezone.utc).isoformat(), "items": items, "item_count": len(items)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_items: list[dict[str, Any]] = []
    dates: dict[str, dict[str, Any]] = {}
    for path in sorted(errors_dir.glob("*.json")):
        payload = load_json(path, {})
        current = payload.get("items") if isinstance(payload, dict) else []
        current = [item for item in current if isinstance(item, dict)]
        date_value = str(payload.get("date") or path.stem)
        detail_file = f"errors/{path.name}"
        dates[date_value] = {"date": date_value, "item_count": len(current), "detail_file": detail_file}
        for item in current:
            all_items.append({key: value for key, value in item.items() if key not in {"log_path", "source_path", "log_excerpt"}} | {"detail_file": detail_file})
    all_items.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("run_id") or ""), str(item.get("model") or "")), reverse=True)
    (website_root / "errors_index.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "schema_version": "1.0", "dates": dict(sorted(dates.items(), reverse=True)), "items": all_items, "item_count": len(all_items)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"published {len(items)} errors to {website_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("website_root", type=Path)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    publish(args.run_dir.resolve(), args.website_root.resolve(), args.date)


if __name__ == "__main__":
    main()
