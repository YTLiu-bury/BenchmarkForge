#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


SUITES = {
    "Bench": ("torchbench", "torchbench.py"),
    "HuggingFace": ("huggingface", "huggingface.py"),
    "Timm": ("timm", "timm_models.py"),
}


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def merge_source(target: dict[str, Any], source: dict[str, Any], tag: str) -> None:
    suites = target.setdefault("suites", {})
    for legacy_name, (suite_name, runner) in SUITES.items():
        section = source.get(legacy_name, {})
        if not isinstance(section, dict):
            continue
        suite = suites.setdefault(
            suite_name,
            {
                "runner": runner,
                "defaults": {
                    "args": section.get("args", {}),
                    "flags": section.get("flags", []),
                },
                "models": {},
            },
        )
        overrides = section.get("overrides", {})
        overrides = overrides if isinstance(overrides, dict) else {}
        for raw_name in section.get("models", []):
            name = str(raw_name).strip()
            if not name:
                continue
            model = suite["models"].setdefault(name, {"tags": [], "policy": {}})
            if tag not in model["tags"]:
                model["tags"].append(tag)
            if name in overrides and isinstance(overrides[name], dict):
                model["policy"].update(overrides[name])


def main() -> int:
    parser = argparse.ArgumentParser(description="Import existing BenchBoard configs into the new catalog")
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--community", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target: dict[str, Any] = {"schema_version": 1, "generated_from": [], "suites": {}}
    merge_source(target, load(args.daily), "daily")
    target["generated_from"].append(str(args.daily))
    if args.community and args.community.is_file():
        merge_source(target, load(args.community), "community")
        target["generated_from"].append(str(args.community))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(target, handle, allow_unicode=True, sort_keys=False, width=120)
    counts = {
        suite: len(config.get("models", {}))
        for suite, config in target["suites"].items()
    }
    print(f"wrote {args.output}: {counts}, total={sum(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
