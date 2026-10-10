from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import Catalog, ModelSpec


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sanitize(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return text or "unknown"


def serialize_args(args: dict[str, Any], flags: list[str]) -> list[str]:
    output: list[str] = []
    for key, value in args.items():
        flag = str(key) if str(key).startswith("--") else f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            output.append(flag)
        else:
            output.extend([flag, str(value)])
    for item in flags:
        flag = str(item)
        output.append(flag if flag.startswith("--") else f"--{flag}")
    return output


def replace_flag(flags: list[str], name: str, enabled: bool) -> list[str]:
    normalized = name.lstrip("-")
    result = [item for item in flags if item.lstrip("-") != normalized]
    if enabled:
        result.append(normalized)
    return result


def effective_command(
    model: ModelSpec,
    *,
    python: str,
    benchmark_dir: Path,
    backend: str,
    result_file: Path,
    profile_dir: Path,
    iterations: int | None,
    repeat: int | None,
    timeout: int | None,
    aclgraph: str,
    dynamic: str,
    profile: str,
    profiler_level: int,
    profiler_warmup: int,
) -> list[str]:
    args = dict(model.args)
    flags = list(model.flags)
    if iterations is not None:
        args["iterations"] = iterations
    if repeat is not None:
        args["repeat"] = repeat
    if timeout is not None:
        args["timeout"] = timeout
    if aclgraph != "config":
        flags = replace_flag(flags, "disable-aclgraph", aclgraph == "off")
    if dynamic != "config":
        flags = replace_flag(flags, "dynamic-shapes", dynamic == "on")
        if dynamic == "off":
            flags = replace_flag(flags, "dynamic-batch-only", False)
    profiler_arg = "enable-profiler"
    profiler_enabled = profiler_arg in {item.lstrip("-") for item in flags} or profiler_arg in args
    if profile != "config":
        profiler_enabled = profile == "on"
    flags = replace_flag(flags, profiler_arg, False)
    args.pop(profiler_arg, None)
    if profiler_enabled:
        configured_iterations = int(args.get("iterations", 0))
        if configured_iterations <= profiler_warmup:
            raise ValueError(
                f"profiler requires iterations > warmup ({configured_iterations} <= {profiler_warmup})"
            )
        args[profiler_arg] = profiler_level
    command = [python, str(benchmark_dir / model.runner)]
    command.extend(serialize_args(args, flags))
    if "--devices" not in command:
        command.extend(["--devices", "npu"])
    command.extend(["--npu-backend", backend, "--only", model.name])
    command.extend(["--output", str(result_file)])
    if "--enable-profiler" in command:
        command.extend(["--prof-output-path", str(profile_dir)])
    return command


def build_plan(
    catalog: Catalog,
    models: list[ModelSpec],
    *,
    run_id: str,
    run_date: str,
    backends: list[str],
    devices: dict[str, list[int]],
    iterations: int | None,
    repeat: int | None,
    timeout: int | None,
    aclgraph: str,
    dynamic_modes: list[str],
    profile: str,
) -> dict[str, Any]:
    paths = catalog.paths()
    benchmark_dir = paths["benchmark_dir"]
    runs_dir = paths["runs_dir"]
    run_dir = runs_dir / run_id
    scheduling = catalog.pipeline.get("scheduling", {})
    scheduling = scheduling if isinstance(scheduling, dict) else {}
    suite_slots = scheduling.get("suite_device_slots", {})
    suite_slots = suite_slots if isinstance(suite_slots, dict) else {}
    defaults = catalog.defaults()
    profiler_level = int(defaults.get("profiler_level", 1))
    profiler_warmup = int(defaults.get("profiler_warmup", 10))
    profiler_active = int(defaults.get("profiler_active", 10))
    cases: list[dict[str, Any]] = []
    for backend in backends:
        backend_devices = devices[backend]
        for dynamic in dynamic_modes:
            for index, model in enumerate(models):
                case_timeout = int(model.args.get("timeout", timeout or 0)) or None
                slot = int(suite_slots.get(model.suite, index))
                device = backend_devices[min(max(slot, 0), len(backend_devices) - 1)]
                case_token = sanitize(f"{model.suite}-{model.name}-{backend}-dynamic-{dynamic}")
                digest = hashlib.sha1(model.model_id.encode("utf-8")).hexdigest()[:8]
                case_id = f"{case_token}-{digest}"
                result_file = run_dir / "raw" / f"{case_id}.csv"
                profile_dir = run_dir / "profile" / case_id
                command = effective_command(
                    model,
                    python=sys.executable,
                    benchmark_dir=benchmark_dir,
                    backend=backend,
                    result_file=result_file,
                    profile_dir=profile_dir,
                    iterations=iterations,
                    repeat=repeat,
                    timeout=case_timeout,
                    aclgraph=aclgraph,
                    dynamic=dynamic,
                    profile=profile,
                    profiler_level=profiler_level,
                    profiler_warmup=profiler_warmup,
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "model_id": model.model_id,
                        "suite": model.suite,
                        "model": model.name,
                        "backend": backend,
                        "dynamic": dynamic,
                        "device": device,
                        "timeout": case_timeout,
                        "command": command,
                        "log": str(run_dir / "logs" / f"{case_id}.log"),
                        "output": str(result_file),
                        "profile": str(profile_dir),
                    }
                )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_date": run_date,
        "created_at": utc_now(),
        "benchmark_dir": str(benchmark_dir),
        "model_count": len(models),
        "case_count": len(cases),
        "backends": backends,
        "dynamic_modes": dynamic_modes,
        "devices": devices,
        "scheduling": {
            "strategy": "suite_device_slots" if suite_slots else "round_robin",
            "suite_device_slots": suite_slots,
        },
        "profiler": {
            "level": profiler_level,
            "warmup": profiler_warmup,
            "active": profiler_active,
        },
        "cases": cases,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
