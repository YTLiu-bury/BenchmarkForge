#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from benchpipe.catalog import Catalog, CatalogError
from benchpipe.board_publish import publish_board_data
from benchpipe.doctor import git_sha, print_doctor, run_doctor
from benchpipe.executor import execute_plan
from benchpipe.planner import build_plan, sanitize, write_json
from benchpipe.report import export_results


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_ROOT = ROOT / "config"


def comma_values(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def parse_devices(values: list[str], backends: list[str]) -> dict[str, list[int]]:
    default: list[int] | None = None
    mapping: dict[str, list[int]] = {}
    for value in values:
        if "=" in value:
            backend, raw = value.split("=", 1)
            mapping[backend.strip()] = [int(item) for item in raw.split(",") if item.strip()]
        else:
            default = [int(item) for item in value.split(",") if item.strip()]
    for backend in backends:
        if backend not in mapping:
            if default is None:
                raise ValueError(f"missing devices for backend {backend}; use --devices {backend}=0,1")
            mapping[backend] = list(default)
        if not mapping[backend]:
            raise ValueError(f"empty device list for backend {backend}")
    used: dict[int, str] = {}
    for backend, device_list in mapping.items():
        for device in device_list:
            if device < 0:
                raise ValueError(f"invalid device index {device}")
            if device in used and used[device] != backend:
                raise ValueError(f"device {device} is assigned to both {used[device]} and {backend}")
            used[device] = backend
    return {backend: mapping[backend] for backend in backends}


def add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", default="daily-full", dest="set_name")
    parser.add_argument("--suite", action="append", choices=["torchbench", "huggingface", "timm"])
    parser.add_argument("--model", action="append", help="model id/name glob; may be repeated")


def add_run_options(parser: argparse.ArgumentParser) -> None:
    add_selection(parser)
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--devices", action="append", default=[])
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--run-id")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--aclgraph", choices=["config", "on", "off"], default="config")
    parser.add_argument(
        "--dynamic",
        action="append",
        choices=["config", "on", "off"],
        help="dynamic mode; repeat to build an on/off matrix",
    )
    parser.add_argument("--profile", choices=["config", "on", "off"], default="config")


def selected_models(catalog: Catalog, args: argparse.Namespace):
    suites = set(args.suite or []) or None
    patterns = comma_values(args.model)
    return catalog.select(args.set_name, suites=suites, patterns=patterns or None)


def command_list(catalog: Catalog, args: argparse.Namespace) -> int:
    models = selected_models(catalog, args)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": item.model_id,
                        "suite": item.suite,
                        "name": item.name,
                        "tags": item.tags,
                        "flags": item.flags,
                        "args": item.args,
                    }
                    for item in models
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    counts: dict[str, int] = {}
    for item in models:
        counts[item.suite] = counts.get(item.suite, 0) + 1
        if args.names_only:
            print(item.model_id)
        else:
            print(f"{item.model_id:70} tags={','.join(item.tags)}")
    print(f"TOTAL={len(models)} " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 0


def runtime_metadata(catalog: Catalog, plan: dict[str, Any]) -> dict[str, Any]:
    benchmark_dir = Path(plan["benchmark_dir"])
    versions: dict[str, str] = {}
    for package in ("torch", "torch_npu", "torchvision", "triton"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "installed"))
        except Exception as error:
            versions[package] = f"unavailable: {type(error).__name__}: {error}"
    return {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "created_at": plan["created_at"],
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": versions,
        "runner_repo_commit": git_sha(benchmark_dir),
        "pytorch_benchmark_commit": git_sha(benchmark_dir / "benchmark"),
        "pipeline_commit": git_sha(ROOT),
        "paths": {key: str(value) for key, value in catalog.paths().items()},
    }


def create_plan(catalog: Catalog, args: argparse.Namespace) -> tuple[dict[str, Any], Path, int]:
    models = selected_models(catalog, args)
    if not models:
        raise ValueError("model selection is empty")
    backends = comma_values(args.backend) or ["triton_experimental"]
    devices = parse_devices(args.devices or ["0"], backends)
    defaults = catalog.defaults()
    timeout = args.timeout or int(defaults.get("timeout", 3600))
    iterations = args.iterations if args.iterations is not None else int(defaults.get("iterations", 50))
    repeat = args.repeat if args.repeat is not None else int(defaults.get("repeat", 1))
    run_id = args.run_id or sanitize(f"{args.date}-{args.set_name}-{'-'.join(backends)}")
    run_dir = catalog.paths()["runs_dir"] / run_id
    plan_path = run_dir / "plan.json"
    if plan_path.exists() and not getattr(args, "reuse_plan", False):
        raise ValueError(f"run already exists: {run_dir}; choose --run-id or use rerun")
    plan = build_plan(
        catalog,
        models,
        run_id=run_id,
        run_date=args.date,
        backends=backends,
        devices=devices,
        iterations=iterations,
        repeat=repeat,
        timeout=timeout,
        aclgraph=args.aclgraph,
        dynamic_modes=args.dynamic or ["config"],
        profile=args.profile,
    )
    write_json(plan_path, plan)
    write_json(run_dir / "run.json", runtime_metadata(catalog, plan))
    return plan, run_dir, timeout


def command_plan(catalog: Catalog, args: argparse.Namespace) -> int:
    plan, run_dir, _ = create_plan(catalog, args)
    print(f"run_id={plan['run_id']} models={plan['model_count']} cases={plan['case_count']}")
    print(run_dir / "plan.json")
    return 0


def base_environment(catalog: Catalog) -> dict[str, str]:
    env = os.environ.copy()
    paths = catalog.paths()
    if paths.get("data_root"):
        env["TORCHBENCH_DATA_PATH"] = str(paths["data_root"])
    if paths.get("hf_home"):
        env["HF_HOME"] = str(paths["hf_home"])
        env["HUGGINGFACE_HUB_CACHE"] = str(paths["hf_home"] / "hub")
        env["TRANSFORMERS_CACHE"] = str(paths["hf_home"] / "hub")
    if paths.get("pip_cache"):
        env["PIP_CACHE_DIR"] = str(paths["pip_cache"])
    defaults = catalog.defaults()
    env.setdefault("BENCH_PROFILE_WARMUP", str(defaults.get("profiler_warmup", 5)))
    env.setdefault("BENCH_PROFILE_ACTIVE", str(defaults.get("profiler_active", 10)))
    env.setdefault("TORCH_COMPILE_DEBUG", "1")
    env.setdefault("HCCL_DETERMINISTIC", "true")
    # Keep the legacy spelling while existing environments transition.
    env.setdefault("HCCL_DETERMINSTIC", "true")
    env.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    env.setdefault("CLOSE_MATMUL_K_SHIFT", "1")
    return env


def command_run(catalog: Catalog, args: argparse.Namespace) -> int:
    plan, run_dir, timeout = create_plan(catalog, args)
    results = execute_plan(
        plan,
        run_dir=run_dir,
        base_env=base_environment(catalog),
        timeout=timeout,
        dry_run=args.dry_run,
    )
    summary = export_results(plan, run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = [item for item in results if item["status"] not in {"pass_accuracy", "pass_eager", "success", "dry_run"}]
    return 1 if failed and not args.continue_on_error else 0


def load_run(catalog: Catalog, run_id: str) -> tuple[dict[str, Any], Path]:
    run_dir = catalog.paths()["runs_dir"] / run_id
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise ValueError(f"plan not found: {plan_path}")
    return json.loads(plan_path.read_text(encoding="utf-8")), run_dir


def command_rerun(catalog: Catalog, args: argparse.Namespace) -> int:
    plan, run_dir = load_run(catalog, args.run_id)
    timeout = args.timeout or int(catalog.defaults().get("timeout", 3600))
    execute_plan(
        plan,
        run_dir=run_dir,
        base_env=base_environment(catalog),
        timeout=timeout,
        dry_run=args.dry_run,
        only_failed=True,
    )
    print(json.dumps(export_results(plan, run_dir), ensure_ascii=False, indent=2))
    return 0


def command_export(catalog: Catalog, args: argparse.Namespace) -> int:
    plan, run_dir = load_run(catalog, args.run_id)
    print(json.dumps(export_results(plan, run_dir), ensure_ascii=False, indent=2))
    return 0


def command_publish(catalog: Catalog, args: argparse.Namespace) -> int:
    plan, run_dir = load_run(catalog, args.run_id)
    selected_model_ids: set[str] | None = None
    if args.model_set:
        selected_model_ids = {model.model_id for model in catalog.select(args.model_set)}
        planned_model_ids = {str(case.get("model_id", "")) for case in plan.get("cases", [])}
        missing = sorted(selected_model_ids - planned_model_ids)
        if missing:
            raise ValueError(
                f"run does not contain {len(missing)} models from set {args.model_set}: "
                + ", ".join(missing[:10])
            )
    default_publish_root = ROOT / "publish"
    if args.model_set:
        default_publish_root = default_publish_root / args.model_set
    publish_root = (args.publish_root or default_publish_root).expanduser().resolve()
    publishing = catalog.pipeline.get("publishing", {})
    publishing = publishing if isinstance(publishing, dict) else {}
    website_key = "community_website_root" if args.model_set else "internal_website_root"
    configured_website_root = publishing.get(website_key)
    default_website_root = Path(str(configured_website_root)) if configured_website_root else publish_root / "website"
    website_root = (args.website_root or default_website_root).expanduser().resolve()
    configured_history_root = publishing.get("history_website_root")
    default_history_root = Path(str(configured_history_root)) if configured_history_root else None
    history_root = (args.history_root or default_history_root)
    history_root = history_root.expanduser().resolve() if history_root is not None else None
    result = publish_board_data(
        plan,
        run_dir,
        publish_root=publish_root,
        website_root=website_root,
        package_tag=args.package_tag,
        extension=args.extension,
        selected_backends=comma_values(args.backend) or None,
        selected_model_ids=selected_model_ids,
        projection=args.model_set or "all",
        history_root=history_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_status(catalog: Catalog, args: argparse.Namespace) -> int:
    plan, run_dir = load_run(catalog, args.run_id)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        export_results(plan, run_dir)
    print(summary_path.read_text(encoding="utf-8"), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPU benchmark pipeline controller")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="resolve and list a model set")
    add_selection(list_parser)
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument("--names-only", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="check environment without modifying it")
    doctor_parser.add_argument("--set", default="smoke", dest="set_name")
    doctor_parser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="resolve a run into an immutable plan")
    add_run_options(plan_parser)

    run_parser = subparsers.add_parser("run", help="plan and execute benchmark cases")
    add_run_options(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--continue-on-error", action="store_true")

    rerun_parser = subparsers.add_parser("rerun", help="rerun non-passing cases in an existing run")
    rerun_parser.add_argument("--run-id", required=True)
    rerun_parser.add_argument("--timeout", type=int)
    rerun_parser.add_argument("--dry-run", action="store_true")

    for command in ("status", "export"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--run-id", required=True)
    publish_parser = subparsers.add_parser("publish", help="convert a run to dashboard-compatible daily JSON")
    publish_parser.add_argument("--run-id", required=True)
    publish_parser.add_argument("--backend", action="append")
    publish_parser.add_argument("--model-set", help="publish only models from this configured set")
    publish_parser.add_argument("--package-tag", help="legacy package key such as 2130; inferred from torch_npu")
    publish_parser.add_argument("--extension", default="default")
    publish_parser.add_argument("--publish-root", type=Path)
    publish_parser.add_argument("--website-root", type=Path)
    publish_parser.add_argument("--history-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        catalog = Catalog(args.config_root)
        if args.command == "list":
            return command_list(catalog, args)
        if args.command == "doctor":
            report, ok = run_doctor(catalog, args.set_name)
            print_doctor(report, args.json)
            return 0 if ok else 1
        if args.command == "plan":
            return command_plan(catalog, args)
        if args.command == "run":
            return command_run(catalog, args)
        if args.command == "rerun":
            return command_rerun(catalog, args)
        if args.command == "export":
            return command_export(catalog, args)
        if args.command == "publish":
            return command_publish(catalog, args)
        if args.command == "status":
            return command_status(catalog, args)
        raise AssertionError(args.command)
    except (CatalogError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
