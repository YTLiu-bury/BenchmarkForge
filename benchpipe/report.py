from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .executor import PASS_STATUSES, result_path
from .planner import write_json


BASE_COLUMNS = [
    "case_id",
    "model_id",
    "suite",
    "model",
    "backend",
    "dynamic",
    "device",
    "status",
    "return_code",
    "attempt",
    "duration_seconds",
    "e2e_status",
    "profile_status",
    "log",
    "output",
]


def collect_results(plan: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in plan["cases"]:
        path = result_path(run_dir, case["case_id"])
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            results.append(
                {
                    **{key: case.get(key, "") for key in BASE_COLUMNS},
                    "status": "not_run",
                    "metrics": {},
                }
            )
    return results


def export_results(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    results = collect_results(plan, run_dir)
    metric_keys = sorted({key for item in results for key in item.get("metrics", {})})
    columns = BASE_COLUMNS + metric_keys
    csv_path = run_dir / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = {key: item.get(key, "") for key in BASE_COLUMNS}
            row.update(item.get("metrics", {}))
            writer.writerow(row)
    status_path = run_dir / "status.tsv"
    with status_path.open("w", encoding="utf-8") as handle:
        handle.write("case_id\tsuite\tmodel\tbackend\tdynamic\tdevice\tstatus\tlog\n")
        for item in results:
            handle.write(
                "\t".join(
                    str(item.get(key, ""))
                    for key in (
                        "case_id",
                        "suite",
                        "model",
                        "backend",
                        "dynamic",
                        "device",
                        "status",
                        "log",
                    )
                )
                + "\n"
            )
    counts = Counter(str(item.get("status", "unknown")) for item in results)
    summary = {
        "run_id": plan["run_id"],
        "model_count": plan["model_count"],
        "case_count": plan["case_count"],
        "completed_cases": sum(status != "not_run" for status in counts.elements()),
        "passed_cases": sum(count for status, count in counts.items() if status in PASS_STATUSES),
        "status_counts": dict(sorted(counts.items())),
        "results_csv": str(csv_path),
        "status_tsv": str(status_path),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
