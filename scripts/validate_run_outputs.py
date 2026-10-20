#!/usr/bin/env python3
"""Fail fast when a benchmark run or its dashboard export loses mode metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_MODES = {"train", "inference"}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plan_expectations(plan: dict[str, Any]) -> tuple[set[tuple[str, str, str]], list[str]]:
    expected: set[tuple[str, str, str]] = set()
    errors: list[str] = []
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        return expected, ["plan has no cases"]

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"case[{index}] is not an object")
            continue
        backend = str(case.get("backend") or "").strip()
        suite = str(case.get("suite") or "").strip()
        mode = str(case.get("mode") or "").strip().lower()
        if not backend or not suite:
            errors.append(f"case[{index}] is missing backend or suite")
        if mode not in VALID_MODES:
            errors.append(
                f"case[{index}] {suite}/{case.get('model')}: "
                f"invalid execution mode {mode!r}; expected train or inference"
            )
        else:
            expected.add((backend, suite, mode))
    return expected, errors


def validate_plan(root: Path, run_id: str) -> tuple[dict[str, Any], list[str]]:
    plan_path = root / "runs" / run_id / "plan.json"
    if not plan_path.is_file():
        return {}, [f"missing plan: {plan_path}"]
    plan = load_json(plan_path)
    if not isinstance(plan, dict):
        return {}, [f"invalid plan object: {plan_path}"]
    expected, errors = plan_expectations(plan)
    if not expected:
        errors.append("plan has no train/inference cases")
    declared = {str(mode).strip().lower() for mode in plan.get("modes", [])}
    case_modes = {mode for _, _, mode in expected}
    if declared and declared != case_modes:
        errors.append(
            f"plan modes {sorted(declared)} do not match case modes {sorted(case_modes)}"
        )
    return {"plan": plan, "expected": expected}, errors


def validate_published(publish_root: Path, date: str, expected: set[tuple[str, str, str]]) -> list[str]:
    data_dir = publish_root / "web_data" / date
    errors: list[str] = []
    if not data_dir.is_dir():
        return [f"missing published data directory: {data_dir}"]

    for backend, suite, mode in sorted(expected):
        file_suite = {"torchbench": "bench"}.get(suite, suite)
        candidates = sorted(
            path
            for path in data_dir.glob(f"*_{file_suite}_{mode}.json")
            if f"_{backend}_" in path.name
        )
        if not candidates:
            errors.append(f"missing mode-split JSON for {backend}/{suite}/{mode}")
            continue
        for path in candidates:
            try:
                payload = load_json(path)
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"cannot parse {path.name}: {exc}")
                continue
            if payload.get("execution_mode") != mode:
                errors.append(
                    f"{path.name}: execution_mode={payload.get('execution_mode')!r}, expected {mode!r}"
                )
            models = payload.get("models")
            if isinstance(models, dict):
                for name, entry in models.items():
                    entry_mode = entry.get("execution_mode") if isinstance(entry, dict) else None
                    if entry_mode != mode:
                        errors.append(
                            f"{path.name}: model {name} has execution_mode={entry_mode!r}"
                        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("run", "publish"), required=True)
    parser.add_argument("--publish-root", type=Path)
    args = parser.parse_args()

    result, errors = validate_plan(args.root, args.run_id)
    if args.stage == "publish" and not errors:
        plan = result["plan"]
        publish_root = args.publish_root or (args.root / "publish")
        errors.extend(validate_published(publish_root, str(plan.get("run_date")), result["expected"]))

    if errors:
        print("RUN_VALIDATION=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    plan = result["plan"]
    print(
        f"RUN_VALIDATION=PASS stage={args.stage} run_id={args.run_id} "
        f"models={plan.get('model_count')} cases={plan.get('case_count')} "
        f"modes={sorted({mode for _, _, mode in result['expected']})}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
