#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
BOARD_DATA_DIR = PIPELINE_ROOT / "publish" / "board_file"
RAW_DATA_DIR = BOARD_DATA_DIR / "raw_data"
WEB_DATA_DIR = PIPELINE_ROOT / "publish" / "web_data"

ALLOWED_ACTIONS = {"clean", "process", "run"}
ALLOWED_BACKENDS = {"dvm", "mlir", "triton", "akg", "triton_experimental"}
ALLOWED_TYPES= {"bench", "huggingface", "llm", "timm"}
PROFILE_KEEP_FILES = ["step_trace_time.csv", "op_statistic.csv", "kernel_details.csv", "trace_view.json"]
DATE_PATTERN = re.compile(r"^\d{8}$")
PACKAGE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
VERSION_PREFIX_PATTERN = re.compile(r"^v?(\d{3,4})(?:_|$)")
EXTENSION_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

MODEL_PATTERN = re.compile(r"(npu|cuda)\s+train\s+(\S+)")
BENCH_E2E_PATTERN = re.compile(
    r"\[(eager|compile)\] summary \[(\d+)-(\d+)\] total steps time: ([\d.]+) ms, avg step time: ([\d.]+) ms"
)
LLM_E2E_PATTERN = re.compile(
    r"\[(eager|compile)\]\s+total step:\s*\d+\s+total steps time:\s*[\d.+]+\s*ms,\s*avg step time:\s*([\d.]+)\s*ms"
)
OP_COMPILE_PATTERN = re.compile(r"op_compile_time:\s*([\d.]+)\s*ms")



MODEL_NAME_MAP: dict[str, str] = {
    "GLM4-9B":"glm4-9B-chat",
    "GPT-OSS-20B":"gpt-oss-20B",
    "baichuan2-7B-Chat":"baichuan2-7B-Chat",
    "llama3-8B":"llama3-8B",
    "llama3.2-3B":"llama3.2-3B",
    "mamba-codestral-7B":"mamba-codestral-7B",
    "qwen2vl-2B-Instruct":"qwen2vl-2B-Instruct",
    "qwen3-4B":"qwen3-4B",
    "sd3":"sd3",
    "sdxl-base-1.0":"sdxl",
}

_MODEL_NAME_MAP_LOWER = {k.lower(): v for k, v in MODEL_NAME_MAP.items()}


def configure_paths(board_data_dir: Path, web_data_dir: Path) -> None:
    """Point the processor at one pipeline-owned staging tree."""
    global BOARD_DATA_DIR, RAW_DATA_DIR, WEB_DATA_DIR
    BOARD_DATA_DIR = Path(board_data_dir).resolve()
    RAW_DATA_DIR = BOARD_DATA_DIR / "raw_data"
    WEB_DATA_DIR = Path(web_data_dir).resolve()


def normalize_extension(extension: str) -> str:
    raw = str(extension or "").strip()
    if not raw or raw.lower() == "default":
        return "default"
    tokens: list[str] = []
    for item in raw.split("+"):
        token = item.strip().lower()
        if not token or token == "default":
            continue
        if not EXTENSION_TOKEN_PATTERN.fullmatch(token):
            raise ValueError(f"invalid extension token: {item}")
        if token not in tokens:
            tokens.append(token)
    return "+".join(tokens) if tokens else "default"

def normalize_run_type(run_type: str) -> str:
    text = str(run_type or "").strip().lower()
    if text not in ALLOWED_TYPES:
        raise ValueError(f"非法的 run_type: {run_type}")
    return text

def normalize_backend(backend: str) -> str:
    text = str(backend or "").strip().lower()
    if text not in ALLOWED_BACKENDS:
        raise ValueError(f"非法的 backend: {backend}")
    return text

def normalize_date(date: str) -> str:
    text = str(date or "").strip()
    if not DATE_PATTERN.match(text):
        raise ValueError(f"非法的日期: {date}")
    return text

def normalize_version(version: str) -> str:
    text = str(version or "").strip()
    if not text or not PACKAGE_TAG_PATTERN.fullmatch(text):
        raise ValueError(f"非法的 package_tag: {version}")
    return text

def normalize_pta_version(raw_version: str) -> str:
    text = str(raw_version or "").strip()
    match = VERSION_PREFIX_PATTERN.match(text)
    if not match:
        return text
    digits = match.group(1)
    if len(digits) == 4:
        return f"v{digits[0]}.{digits[1:3]}.{digits[3]}"
    return f"v{digits[0]}.{digits[1]}.{digits[2:]}"

def raw_data_suite(run_type:str) -> str:
    return {"bench": "benchmark", "huggingface": "huggingface", "llm": "llm", "timm": "timm"}[run_type]

def normalize_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return raw
    direct = MODEL_NAME_MAP.get(raw)
    if direct:
        return direct
    return _MODEL_NAME_MAP_LOWER.get(raw.lower(), raw)

def relative_path(path: Path | None)->str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(BOARD_DATA_DIR))
    except ValueError:
        return str(path)

def safe_ratio(numerator: float, denominator: float) -> float:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator

@dataclass(frozen=True)
class RawDataLayout:
    run_type: str
    backend: str
    date: str
    version: str
    extension: str

    @classmethod
    def create(cls, run_type: str, backend: str, date: str, version: str, extension: str) -> RawDataLayout:
        return cls(
            run_type=normalize_run_type(run_type),
            backend=normalize_backend(backend),
            date=normalize_date(date),
            version=normalize_version(version),
            extension=normalize_extension(extension),
        )

    @property
    def suite(self) -> str:
        return raw_data_suite(self.run_type)

    @property
    def pta_version(self) -> str:
        return normalize_pta_version(self.version)

    @property
    def mode_label(self) -> str:
        return self.backend

    @property
    def model_type(self) -> str:
        return {"bench": "Benchmark", "huggingface": "HuggingFace", "llm": "LLM", "timm": "Benchmark"}[self.run_type]

    @property
    def backend_dir(self) -> Path:
        return RAW_DATA_DIR / self.date / self.suite / self.version / self.backend / self.extension

    @property
    def output_dir(self) -> Path:
        return WEB_DATA_DIR / self.date

    @property
    def run_id(self) -> str:
        return f"{self.date}_{self.pta_version}_{self.backend}_{self.extension}_{self.run_type}"

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"{self.run_id}.json"

    def iter_model_dirs(self) -> list[Path]:
        if not self.backend_dir.exists():
            return []
        return sorted(path for path in self.backend_dir.iterdir() if path.is_dir())


class RawDataCleaner:
    def __init__(self, layout: RawDataLayout) -> None:
        self.layout = layout

    def clean_backend_dir(self) -> None:
        backend_dir = self.layout.backend_dir
        if not backend_dir.exists() or not backend_dir.is_dir():
            raise FileNotFoundError(f"backend_dir not found: {backend_dir}")

        model_dirs = self.layout.iter_model_dirs()
        if not model_dirs:
            print(f"backend_dir {backend_dir} 下没有模型目录")
            return

        print(f"开始清理目录: {backend_dir}")
        print(f"发现{len(model_dirs)}个 model 目录需要清理")
        for model_dir in model_dirs:
            self._clean_model_dir(model_dir)

    def _safe_rmtree(self, path: Path) -> None:
        path = path.resolve()
        workspace_root = BOARD_DATA_DIR.resolve()
        try:
            path.relative_to(workspace_root)
        except ValueError:
            raise RuntimeError(f"尝试清理非工作区目录: {path}")

        if not path.exists():
            print(f"目录不存在: {path}")
            return

        if path.is_dir():
            shutil.rmtree(path)
            print(f"清理目录: {path}")
        else:
            path.unlink()
            print(f"清理文件: {path}")

    def _safe_copy2(self, src: Path, dst: Path) -> None:
        src = src.resolve()
        dst = dst.resolve()
        workspace_root = BOARD_DATA_DIR.resolve()
        try:
            src.relative_to(workspace_root)
            dst.relative_to(workspace_root)
        except ValueError as exc:
            raise RuntimeError(f"尝试复制非工作区目录: {src} 到 {dst}")

        if not src.exists():
            print(f"源文件不存在: {src}")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"复制文件: {src} 到 {dst}")

    def _delete_children_except(self, dir_path: Path, keep_names: list[str]) -> None:
        if not dir_path.exists() or not dir_path.is_dir():
            return
        for child in dir_path.iterdir():
            if child.name in keep_names:
                continue
            self._safe_rmtree(child)

    def _find_profile_phase_dir(self, profiler_dir: Path, phase: str) -> Path | None:
        direct = profiler_dir / phase
        if direct.is_dir():
            return direct

        nested_matches = sorted(
            path for path in profiler_dir.glob(f"*/{phase}") if path.is_dir()
        )
        if nested_matches:
            return nested_matches[0]
        return None

    def _flatten_profile_files(self, phase_dir: Path, dst_phase_dir: Path | None = None) -> None:
        if not phase_dir.exists() or not phase_dir.is_dir():
            return

        dst_dir = dst_phase_dir or phase_dir
        found_files: list[str] = []
        for file_name in PROFILE_KEEP_FILES:
            matches = sorted(phase_dir.rglob(file_name))
            if not matches:
                print(f"[SKIP] 未找到 {file_name} 文件: {phase_dir}")
                continue

            src = matches[0]
            dst = dst_dir / file_name
            try:
                if src.resolve() != dst.resolve():
                    self._safe_copy2(src, dst)
            except OSError:
                self._safe_copy2(src, dst)
            found_files.append(file_name)

        self._delete_children_except(dst_dir, PROFILE_KEEP_FILES)

    def _clean_model_dir(self, model_dir: Path) -> None:
        board_log_dir = model_dir / "board_log"
        profiler_dir = model_dir / "profile"

        if board_log_dir.exists() and board_log_dir.is_dir():
            pattern = "*.log"
            keep_logs = {path.name for path in board_log_dir.glob(pattern) if path.is_file()}
            if keep_logs:
                self._delete_children_except(board_log_dir, keep_logs)
                print(f"清理 board_log 目录: {board_log_dir}")
            else:
                print(f"[SKIP] 未找到 board_log 日志文件: {board_log_dir}")
        else:
            print(f"[SKIP] board_log 目录不存在: {board_log_dir}")

        if profiler_dir.exists() and profiler_dir.is_dir():
            for phase in ("eager", "compile"):
                phase_dir = self._find_profile_phase_dir(profiler_dir, phase)
                if phase_dir is None:
                    print(f"[SKIP] 未找到 {phase} 阶段目录: {profiler_dir / phase}")
                    continue
                self._flatten_profile_files(phase_dir, profiler_dir / phase)
            self._delete_children_except(profiler_dir, ["eager", "compile"])
        else:
            print(f"[SKIP] profile 目录不存在: {profiler_dir}")

        self._delete_children_except(model_dir, ["board_log", "profile"])

class RawDataProcessor:
    def __init__(self, layout: RawDataLayout) -> None:
        self.layout = layout

    def process(self) -> dict[str, Any]:
        source_dir, log_files = self._collect_log_files()
        model_results: dict[str, dict[str, Any]] = {}

        for log_file in log_files:
            parsed = self._parse_log_file(log_file)
            for entry in parsed.values():
                entry["type"] = self.layout.model_type

            profile_dir = log_file.parent.parent / "profile"
            parsed = self._enrich_models_with_profile(parsed, profile_dir if profile_dir.exists() else None)
            model_results = self._merge_model_results(model_results, parsed)

        model_results = self._normalize_model_result_names(model_results)
        for entry in model_results.values():
            self._refresh_model_metrics(entry)
            entry["hardware"] = "npu"
            entry["date"] = self.layout.date
            entry["mode"] = self.layout.mode_label
            entry["extension"] = self.layout.extension

        run = self._build_run_payload(source_dir, log_files, model_results)
        self.layout.output_dir.mkdir(parents=True, exist_ok=True)
        with self.layout.output_file.open("w", encoding="utf-8") as handle:
            json.dump(run, handle, indent=2, ensure_ascii=False)
        return run

    def _build_run_payload(
        self, source_dir: Path, log_files: list[Path], model_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "run_id": self.layout.run_id,
            "date": self.layout.date,
            "hardware": "npu",
            "type": self.layout.run_type,
            "mode": self.layout.backend,
            "extension": self.layout.extension,
            "pta_version": self.layout.pta_version,
            "log_files": [relative_path(path) for path in log_files],
            "source_dir": relative_path(source_dir),
            "models": dict[str, dict[str, Any]](sorted(model_results.items())),
        }

    def _collect_log_files(self) -> tuple[Path, list[Path]]:
        root_dir = self.layout.backend_dir
        if not root_dir.exists() or not root_dir.is_dir():
            raise FileNotFoundError(f"backend_dir not found: {root_dir}")

        log_files: list[Path] = []
        for model_dir in self.layout.iter_model_dirs():
            log_dir = model_dir / "board_log"
            if not log_dir.is_dir():
                continue
            log_files.extend(sorted(path for path in log_dir.glob("*.log") if path.is_file()))

        if not log_files:
            raise FileNotFoundError(f"未找到任何日志文件: {root_dir}")
        return root_dir, log_files

    def _normalize_model_result_names(
        self, model_results: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        normalized_results: dict[str, dict[str, Any]] = {}
        for raw_name, raw_entry in model_results.items():
            normalized_name = normalize_model_name(raw_entry.get("model_name") or raw_name)
            entry = dict[str, Any](raw_entry)
            entry["model_name"] = normalized_name
            normalized_results = self._merge_model_results(normalized_results, {normalized_name: entry})
        return normalized_results

    def _infer_single_model_name_from_log_path(self, log_path: Path) -> str:
        try:
            return normalize_model_name(log_path.parent.parent.name)
        except Exception:
            return ""

    def _make_empty_model_entry(self, model_name: str) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "type": "",
            "accuracy": "",
            "op_compile_time": None,
            "eager_E2E_avg_time": None,
            "compile_E2E_avg_time": None,
            "E2E_speed_up_rate": None,
            "eager_OP_avg_time": None,
            "compile_OP_avg_time": None,
            "OP_speed_up_rate": None,
            "status": "missing",
        }

    def _build_model_status(self, entry: dict[str, Any]) -> str:
        if entry.get("accuracy") == "fail_accuracy":
            return "accuracy_failed"
        if entry.get("compile_E2E_avg_time") is None and entry.get("eager_E2E_avg_time") is None:
            return "missing"
        if entry.get("compile_E2E_avg_time") is None:
            return "partial"
        return "ok"

    def _parse_log_file(self, log_path: Path) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        current_model: str | None = None

        inferred = self._infer_single_model_name_from_log_path(log_path)
        if inferred:
            current_model = inferred
            results.setdefault(current_model, self._make_empty_model_entry(current_model))

        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()

                model_match = MODEL_PATTERN.search(line)
                if model_match:
                    matched_model = normalize_model_name(model_match.group(2))
                    if (
                        inferred
                        and matched_model != inferred
                        and inferred in results
                        and results[inferred].get("status") == "missing"
                    ):
                        results.pop(inferred, None)
                    current_model = matched_model
                    results.setdefault(current_model, self._make_empty_model_entry(current_model))
                    continue

                if current_model is None:
                    continue

                if self.layout.run_type in {"bench", "huggingface", "timm"}:
                    bench_match = BENCH_E2E_PATTERN.search(line)
                    if bench_match:
                        phase = bench_match.group(1)
                        avg_time = float(bench_match.group(5))

                        if phase == "eager":
                            if results[current_model].get("eager_E2E_avg_time") is None:
                                results[current_model]["eager_E2E_avg_time"] = avg_time
                        elif phase == "compile":
                            if results[current_model].get("compile_E2E_avg_time") is None:
                                results[current_model]["compile_E2E_avg_time"] = avg_time
                        continue
                else:
                    llm_match = LLM_E2E_PATTERN.search(line)
                    if llm_match:
                        phase = llm_match.group(1).lower()
                        avg_time = float(llm_match.group(2))
                        if phase == "eager":
                            if results[current_model].get("eager_E2E_avg_time") is None:
                                results[current_model]["eager_E2E_avg_time"] = avg_time
                        elif phase == "compile":
                            if results[current_model].get("compile_E2E_avg_time") is None:
                                results[current_model]["compile_E2E_avg_time"] = avg_time
                        continue

                if line in {"pass_accuracy", "fail_accuracy"}:
                    results[current_model]["accuracy"] = line
                    continue

                op_compile_match = OP_COMPILE_PATTERN.search(line)
                if op_compile_match:
                    results[current_model]["op_compile_time"] = float(op_compile_match.group(1))

        for entry in results.values():
            entry["E2E_speed_up_rate"] = safe_ratio(
                entry.get("eager_E2E_avg_time"), entry.get("compile_E2E_avg_time")
            )
        return results

    def _locate_step_trace_time_csv(self, profile_dir: Path, phase_dir_name: str) -> Path | None:
        phase_dir = profile_dir / phase_dir_name
        step_trace = phase_dir / "step_trace_time.csv"
        return step_trace if step_trace.exists() else None


    def _calc_op_avg_from_step_trace_time(self, step_trace_csv: Path|None)->float|None:
        if step_trace_csv is None:
            return None
        try:
            df = pd.read_csv(step_trace_csv, sep=",")
        except Exception as exc:
            print(f"读取 step_trace_time.csv 失败: {step_trace_csv}, 错误: {exc}")
            return None

        if "Computing" not in df.columns:
            return None

        series = pd.to_numeric(df["Computing"], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.mean()) / 1000.0


    def _enrich_models_with_profile(
        self, model_results: dict[str, dict[str, Any]], profile_dir: Path|None
    )-> dict[str, dict[str, Any]]:
        if profile_dir is None or not profile_dir.exists():
            return model_results

        eager_step_trace_csv = self._locate_step_trace_time_csv(profile_dir, "eager")
        compile_step_trace_csv = self._locate_step_trace_time_csv(profile_dir, "compile")
        for entry in model_results.values():
            entry["eager_OP_avg_time"] = self._calc_op_avg_from_step_trace_time(eager_step_trace_csv)
            entry["compile_OP_avg_time"] = self._calc_op_avg_from_step_trace_time(compile_step_trace_csv)
            entry["OP_speed_up_rate"] = safe_ratio(
                entry.get("eager_OP_avg_time"), entry.get("compile_OP_avg_time")
            )
        return model_results

    def _merge_model_results(
        self, base: dict[str, dict[str, Any]], incoming: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        merged = dict[str, dict[str, Any]](base)
        for name, inc in incoming.items():
            if name not in merged:
                merged[name] = inc
                continue

            cur = dict[str, Any](merged[name])
            for key, value in inc.items():
                if value is None:
                    continue
                if key == "step_windows" and isinstance(value, dict):
                    step_windows = dict[Any, Any](cur.get("step_windows", {}))
                    step_windows.update(value)
                    cur[key] = step_windows
                else:
                    cur[key] = value
            merged[name] = cur
        return merged

    def _refresh_model_metrics(self, entry: dict[str, Any]) -> None:
        entry["E2E_speed_up_rate"] = safe_ratio(
            entry.get("eager_E2E_avg_time"), entry.get("compile_E2E_avg_time")
        )
        entry["OP_speed_up_rate"] = safe_ratio(
            entry.get("eager_OP_avg_time"), entry.get("compile_OP_avg_time")
        )
        entry["status"] = self._build_model_status(entry)

class RawDataPipeline:
    def __init__(self, layout: RawDataLayout) -> None:
        self.layout = layout
        self.cleaner = RawDataCleaner(layout)
        self.processor = RawDataProcessor(layout)

    def clean(self) -> None:
        self.cleaner.clean_backend_dir()

    def process(self) -> dict[str, Any]:
        return self.processor.process()

    def run(self) -> dict[str, Any]:
        self.clean()
        return self.process()

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统一处理 raw_data 的 clean/process/run 流水线。\n"
            "位置参数顺序：type backend date package_tag"
        )
    )
    parser.add_argument("type")
    parser.add_argument("backend")
    parser.add_argument("date")
    parser.add_argument("version", help="package tag，例如 v271_34228_e9220b31")
    parser.add_argument("--extension", default="default", help="NPU extension，例如 default 或 dynamic+mfusion")
    parser.add_argument("--action", choices=sorted(ALLOWED_ACTIONS), default="run")
    return parser.parse_args(argv)

def main() -> None:
    args = parse_args()
    layout = RawDataLayout.create(args.type, args.backend, args.date, args.version, args.extension)
    pipeline = RawDataPipeline(layout)

    if args.action == "clean":
        pipeline.clean()
    elif args.action == "process":
        pipeline.process()
    elif args.action == "run":
        pipeline.run()
    else:
        raise ValueError(f"非法 action: {args.action}")

    print(f"处理完成: {layout.output_file}")


if __name__ == "__main__":
    main()
