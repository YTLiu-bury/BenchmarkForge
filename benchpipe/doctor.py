from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .catalog import Catalog


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def package_version(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "installed"))
    except Exception as error:  # importing torch_npu can surface runtime errors
        return False, f"{type(error).__name__}: {error}"


def run_doctor(catalog: Catalog, set_name: str) -> tuple[dict[str, Any], bool]:
    paths = catalog.paths()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    add("python", True, f"{sys.executable} ({sys.version.split()[0]})")
    benchmark_dir = paths.get("benchmark_dir", Path(""))
    add("benchmark_dir", benchmark_dir.is_dir(), str(benchmark_dir))
    for runner in ("torchbench.py", "huggingface.py", "timm_models.py", "common.py"):
        add(f"runner:{runner}", (benchmark_dir / runner).is_file(), str(benchmark_dir / runner))
    data_root = paths.get("data_root", Path(""))
    add("data_root", data_root.is_dir(), str(data_root), required=False)
    cann_home = paths.get("cann_home", Path(""))
    add("cann_home", (cann_home / "set_env.sh").is_file(), str(cann_home))
    add("npu-smi", shutil.which("npu-smi") is not None, shutil.which("npu-smi") or "not found")
    for package in ("torch", "torch_npu", "torchvision", "yaml", "pandas"):
        ok, version = package_version(package)
        add(f"package:{package}", ok, version, required=package in {"torch", "torch_npu", "yaml"})
    try:
        models = catalog.select(set_name)
        add("model_set", bool(models), f"{set_name}: {len(models)} models")
    except Exception as error:
        add("model_set", False, str(error))
    add("runner_repo_commit", True, git_sha(benchmark_dir) or "not a git checkout", required=False)
    add("pytorch_benchmark_commit", True, git_sha(benchmark_dir / "benchmark") or "not a git checkout", required=False)
    ok = all(item["ok"] for item in checks if item["required"])
    return {"ok": ok, "checks": checks}, ok


def print_doctor(report: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    for item in report["checks"]:
        label = "PASS" if item["ok"] else ("WARN" if not item["required"] else "FAIL")
        print(f"[{label:4}] {item['name']}: {item['detail']}")
