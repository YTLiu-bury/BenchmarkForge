from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class CatalogError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CatalogError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise CatalogError(f"expected a mapping in {path}")
    return data


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    suite: str
    name: str
    runner: str
    args: dict[str, Any]
    flags: list[str]
    tags: tuple[str, ...]
    enabled: bool = True
    reason: str = ""


class Catalog:
    def __init__(self, config_root: Path) -> None:
        self.config_root = config_root.resolve()
        self.pipeline = load_yaml(self.config_root / "pipeline.yaml")
        self.model_data = load_yaml(self.config_root / "models.yaml")
        self.set_data = load_yaml(self.config_root / "model_sets.yaml")
        if self.model_data.get("schema_version") != 1:
            raise CatalogError("config/models.yaml has an unsupported schema_version")

    def paths(self) -> dict[str, Path]:
        raw = self.pipeline.get("paths", {})
        if not isinstance(raw, dict):
            raise CatalogError("pipeline.yaml paths must be a mapping")
        root = self.config_root.parent
        values = {
            "benchmark_dir": os.environ.get("TORCHBENCH_DIR") or raw.get("benchmark_dir"),
            "data_root": os.environ.get("TORCHBENCH_DATA_PATH") or raw.get("data_root"),
            "cann_home": os.environ.get("CANN_HOME") or raw.get("cann_home"),
            "hf_home": os.environ.get("HF_HOME") or raw.get("hf_home"),
            "pip_cache": os.environ.get("PIP_CACHE_DIR") or raw.get("pip_cache"),
            "runs_dir": os.environ.get("BENCHMARK_RUNS_DIR") or raw.get("runs_dir", "runs"),
        }
        out: dict[str, Path] = {}
        for key, value in values.items():
            if not value:
                continue
            path = Path(str(value)).expanduser()
            out[key] = path if path.is_absolute() else (root / path).resolve()
        return out

    def defaults(self) -> dict[str, Any]:
        value = self.pipeline.get("defaults", {})
        if not isinstance(value, dict):
            raise CatalogError("pipeline.yaml defaults must be a mapping")
        return dict(value)

    def all_models(self) -> list[ModelSpec]:
        suites = self.model_data.get("suites", {})
        if not isinstance(suites, dict):
            raise CatalogError("models.yaml suites must be a mapping")
        result: list[ModelSpec] = []
        for suite, suite_config in suites.items():
            if not isinstance(suite_config, dict):
                continue
            runner = str(suite_config.get("runner", "")).strip()
            defaults = suite_config.get("defaults", {})
            default_args = defaults.get("args", {}) if isinstance(defaults, dict) else {}
            default_flags = defaults.get("flags", []) if isinstance(defaults, dict) else []
            models = suite_config.get("models", {})
            if not isinstance(models, dict):
                continue
            for name, model_config in models.items():
                model_config = model_config if isinstance(model_config, dict) else {}
                policy = model_config.get("policy", {})
                policy = policy if isinstance(policy, dict) else {}
                args = dict(default_args) if isinstance(default_args, dict) else {}
                if isinstance(policy.get("args"), dict):
                    args.update(policy["args"])
                flags = list(default_flags) if isinstance(default_flags, list) else []
                if isinstance(policy.get("flags"), list):
                    flags = list(policy["flags"])
                tags = tuple(unique(str(item) for item in model_config.get("tags", [])))
                result.append(
                    ModelSpec(
                        model_id=f"{suite}/{name}",
                        suite=str(suite),
                        name=str(name),
                        runner=runner,
                        args=args,
                        flags=[str(item) for item in flags],
                        tags=tags,
                        enabled=bool(model_config.get("enabled", True)),
                        reason=str(model_config.get("reason", "")),
                    )
                )
        return sorted(result, key=lambda item: (item.suite, item.name.lower()))

    def set_names(self) -> list[str]:
        sets = self.set_data.get("sets", {})
        return sorted(sets) if isinstance(sets, dict) else []

    def select(
        self,
        set_name: str,
        *,
        suites: set[str] | None = None,
        patterns: list[str] | None = None,
    ) -> list[ModelSpec]:
        sets = self.set_data.get("sets", {})
        if not isinstance(sets, dict) or set_name not in sets:
            raise CatalogError(f"unknown model set {set_name!r}; available: {', '.join(self.set_names())}")
        selector = sets[set_name] or {}
        if not isinstance(selector, dict):
            raise CatalogError(f"model set {set_name!r} must be a mapping")
        explicit = {str(item) for item in selector.get("models", [])}
        include_tags = {str(item) for item in selector.get("include_tags", [])}
        exclude = [str(item) for item in selector.get("exclude", [])]
        selected: list[ModelSpec] = []
        for model in self.all_models():
            include = model.model_id in explicit if explicit else bool(include_tags.intersection(model.tags))
            if not include or not model.enabled:
                continue
            if suites and model.suite not in suites:
                continue
            if exclude and any(fnmatch.fnmatch(model.model_id, item) for item in exclude):
                continue
            if patterns and not any(
                fnmatch.fnmatch(model.model_id, item) or fnmatch.fnmatch(model.name, item)
                for item in patterns
            ):
                continue
            selected.append(model)
        return selected
