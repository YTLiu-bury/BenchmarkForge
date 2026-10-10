from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .planner import write_json


PASS_STATUSES = {"pass_accuracy", "pass_eager", "success"}
PROFILE_KEEP_FILES = {
    "step_trace_time.csv",
    "op_statistic.csv",
    "kernel_details.csv",
}
E2E_SUMMARY_PATTERN = re.compile(
    r"\[(eager|compile)\] summary \[\d+-\d+\] total steps time: [\d.]+ ms, "
    r"avg step time: [\d.]+ ms"
)


def compact_profile(path: Path) -> int:
    """Keep dashboard inputs and remove bulky raw profiler artifacts per case."""
    if not path.is_dir():
        return 0
    removed_bytes = 0
    entries = sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for entry in entries:
        if entry.is_file() and entry.name not in PROFILE_KEEP_FILES:
            try:
                removed_bytes += entry.stat().st_size
                entry.unlink()
            except OSError:
                pass
        elif entry.is_dir():
            try:
                entry.rmdir()
            except OSError:
                pass
    return removed_bytes


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_last_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else {}
    except (OSError, csv.Error, UnicodeError):
        return {}


def profile_status(path: Path) -> str:
    phases = {
        phase
        for phase in ("eager", "compile")
        if any(
            candidate.is_file() and candidate.stat().st_size > 0
            for candidate in path.rglob("step_trace_time.csv")
            if phase in candidate.relative_to(path).parts
        )
    } if path.is_dir() else set()
    if phases == {"eager", "compile"}:
        return "complete"
    if phases:
        return "partial"
    return "missing"


def e2e_status(log_text: str) -> str:
    phases = {match.group(1) for match in E2E_SUMMARY_PATTERN.finditer(log_text)}
    if phases == {"eager", "compile"}:
        return "complete"
    if phases:
        return "partial"
    return "missing"


def classify(return_code: int, timed_out: bool, metrics: dict[str, str], log_text: str) -> str:
    if timed_out:
        return "timeout"
    accuracy = str(metrics.get("accuracy", "")).strip()
    if accuracy:
        return accuracy
    for token in (
        "pass_accuracy",
        "fail_accuracy",
        "eager_two_runs_differ",
        "eager_fail_to_run",
        "fail_to_run",
    ):
        if token in log_text:
            return token
    return "success" if return_code == 0 else f"exit_{return_code}"


def run_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, timeout: int) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[COMMAND] " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.write(f"\n[TIMEOUT] exceeded {timeout} seconds\n")
            return 124, True


def result_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / "case-results" / f"{case_id}.json"


def execute_case(
    case: dict[str, Any],
    *,
    run_dir: Path,
    benchmark_dir: Path,
    base_env: dict[str, str],
    timeout: int,
    dry_run: bool,
) -> dict[str, Any]:
    started = utc_now()
    start_time = time.monotonic()
    device = int(case["device"])
    lock_dir = run_dir.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"npu-{device}.lock"
    case_result_path = result_path(run_dir, case["case_id"])
    previous: dict[str, Any] = {}
    if case_result_path.is_file():
        previous = json.loads(case_result_path.read_text(encoding="utf-8"))
    attempt = int(previous.get("attempt", 0)) + 1
    with lock_file.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = dict(base_env)
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
        env["TORCHINDUCTOR_NPU_BACKEND"] = str(case["backend"])
        env["TORCHINDUCTOR_CACHE_DIR"] = str(run_dir / "cache" / case["case_id"])
        env["TORCH_COMPILE_DEBUG_DIR"] = str(run_dir / "compile-debug" / case["case_id"])
        Path(case["output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(case["profile"]).mkdir(parents=True, exist_ok=True)
        if dry_run:
            return_code, timed_out, status, metrics = 0, False, "dry_run", {}
            Path(case["log"]).parent.mkdir(parents=True, exist_ok=True)
            Path(case["log"]).write_text("[DRY-RUN] " + " ".join(case["command"]) + "\n", encoding="utf-8")
        else:
            log_text = ""
            try:
                case_timeout = int(case.get("timeout") or timeout)
                return_code, timed_out = run_process(
                    list(case["command"]),
                    benchmark_dir,
                    env,
                    Path(case["log"]),
                    case_timeout + 60,
                )
                metrics = load_last_csv_row(Path(case["output"]))
                log_text = Path(case["log"]).read_text(encoding="utf-8", errors="replace")
                status = classify(return_code, timed_out, metrics, log_text)
            except Exception as error:
                return_code, timed_out, metrics, status = 127, False, {}, "runner_error"
                Path(case["log"]).parent.mkdir(parents=True, exist_ok=True)
                with Path(case["log"]).open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[RUNNER_ERROR] {type(error).__name__}: {error}\n")
        requires_profile = "--enable-profiler" in case["command"]
        case_profile_status = profile_status(Path(case["profile"])) if requires_profile else "disabled"
        compacted_profile_bytes = compact_profile(Path(case["profile"])) if requires_profile else 0
        case_e2e_status = e2e_status(log_text) if not dry_run else "dry_run"
    result = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "model_id": case["model_id"],
        "suite": case["suite"],
        "model": case["model"],
        "backend": case["backend"],
        "dynamic": str(case.get("dynamic", "config")),
        "device": device,
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "e2e_status": case_e2e_status,
        "profile_status": case_profile_status,
        "timeout_seconds": int(case.get("timeout") or timeout),
        "attempt": attempt,
        "profile_compacted_bytes": compacted_profile_bytes,
        "started_at": started,
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - start_time, 3),
        "metrics": metrics,
        "log": case["log"],
        "output": case["output"],
    }
    write_json(case_result_path, result)
    return result


def execute_plan(
    plan: dict[str, Any],
    *,
    run_dir: Path,
    base_env: dict[str, str],
    timeout: int,
    dry_run: bool,
    only_failed: bool = False,
) -> list[dict[str, Any]]:
    cases = list(plan["cases"])
    if only_failed:
        filtered = []
        for case in cases:
            path = result_path(run_dir, case["case_id"])
            if not path.is_file():
                filtered.append(case)
                continue
            previous = json.loads(path.read_text(encoding="utf-8"))
            requires_profile = "--enable-profiler" in case.get("command", [])
            metrics_incomplete = previous.get("e2e_status") != "complete" or (
                requires_profile and previous.get("profile_status") != "complete"
            )
            if previous.get("status") not in PASS_STATUSES or metrics_incomplete:
                filtered.append(case)
        cases = filtered
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault((str(case["backend"]), int(case["device"])), []).append(case)

    def worker(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            execute_case(
                item,
                run_dir=run_dir,
                benchmark_dir=Path(plan["benchmark_dir"]),
                base_env=base_env,
                timeout=timeout,
                dry_run=dry_run,
            )
            for item in items
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        futures = [pool.submit(worker, items) for items in groups.values()]
        for future in as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: (item["backend"], item["suite"], item["model"]))
