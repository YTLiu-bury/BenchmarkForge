#!/usr/bin/env python3
"""Compare configured models with the model names recognized by each runner."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchpipe.catalog import Catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-set", default="daily-full")
    parser.add_argument("--output", type=Path, default=ROOT / "audits" / "catalog_support.json")
    return parser.parse_args()


def first_column(path: Path, separator: str) -> set[str]:
    if not path.is_file():
        return set()
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            names.add(line.split(separator, 1)[0])
    return names


def mapping_keys(path: Path, variable: str) -> set[str]:
    """Read string keys from a module-level mapping without importing it."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            return set()
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def main() -> int:
    args = parse_args()
    catalog = Catalog(ROOT / "config")
    benchmark_dir = catalog.paths()["benchmark_dir"]
    torchbench_root = benchmark_dir / "benchmark" / "torchbenchmark" / "models"
    enumerated = {
        "torchbench": {
            item.name
            for item in torchbench_root.iterdir()
            if item.is_dir() and (item / "__init__.py").is_file()
        },
        "huggingface": first_column(benchmark_dir / "huggingface_models_list.txt", ","),
        "timm": first_column(benchmark_dir / "timm_models_list.txt", " "),
    }
    extra_hf = mapping_keys(benchmark_dir / "huggingface.py", "EXTRA_MODELS")
    hf_llm = mapping_keys(
        benchmark_dir / "huggingface_llm_models.py", "HF_LLM_MODELS"
    )
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError:
        transformers = None

    def resolvable(model) -> bool | None:
        if model.suite == "torchbench":
            return model.name in enumerated[model.suite]
        if model.suite == "timm":
            return model.name in enumerated[model.suite]
        if model.name in extra_hf or model.name in hf_llm:
            return True
        if transformers is None:
            return None
        return hasattr(transformers, model.name)

    models = catalog.select(args.model_set)
    resolution = {model.model_id: resolvable(model) for model in models}
    unsupported = [model.model_id for model in models if resolution[model.model_id] is False]
    unknown = [model.model_id for model in models if resolution[model.model_id] is None]
    report = {
        "model_set": args.model_set,
        "catalog_models": len(models),
        "enumerated_by_runner": sum(model.name in enumerated[model.suite] for model in models),
        "load_target_resolvable": sum(value is True for value in resolution.values()),
        "unsupported": unsupported,
        "runtime_unknown": unknown,
        "not_enumerated_by_suite": {
            suite: [model.model_id for model in models if model.suite == suite and model.name not in names]
            for suite, names in enumerated.items()
        },
        "runner_inventory": {suite: len(names) for suite, names in enumerated.items()},
        "note": "The pipeline uses --only, so HuggingFace classes can be loadable even when absent from the runner list.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
