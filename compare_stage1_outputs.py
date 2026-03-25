from __future__ import annotations

import argparse
import os
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd


DEFAULT_NEW_DIR = "outputs/experiments/标题_摘要_window3/stage1"
DEFAULT_OLD_DIR = "data/result/标题+摘要版本+window_size=3"
DEFAULT_REPORT_PATH = "outputs/comparisons/stage1_compare_report_window3.md"
READ_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
KEY_COLUMN_CANDIDATES = ("申请号", "row", "申请年份", "年份", "专利名称")
EXCLUDED_PREFIXES = ("pair_contrib/",)
EXCLUDED_KEYS = {"__stage1_log__"}

STAGE_RULES: list[tuple[str, str]] = [
    ("阶段1 词表与DF", "vocab/"),
    ("阶段1 词表与DF", "df/"),
    ("阶段2 Tokens", "tokens/"),
    ("阶段3 向量化", "vectors/"),
    ("阶段3 向量化", "index/"),
    ("阶段4 向量剪枝", "vectors_filtered/"),
    ("阶段5 BS/FS", "postings/"),
    ("阶段5 BS/FS", "pair_contrib/"),
    ("阶段5 BS/FS", "stats/"),
    ("阶段6 最终结果", "patent_quality_output.csv"),
    ("元数据", "checkpoint.json"),
    ("日志", "__stage1_log__"),
]
STAGE_ORDER = {name: idx for idx, name in enumerate(dict.fromkeys(stage for stage, _ in STAGE_RULES))}


@dataclass
class FileComparison:
    stage: str
    rel_key: str
    new_rel: str | None
    old_rel: str | None
    status: str
    file_type: str
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class ComparisonRun:
    results: list[FileComparison]
    stopped_on: FileComparison | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比两个 stage1 输出目录，并生成 Markdown 报告")
    parser.add_argument("--new-dir", default=DEFAULT_NEW_DIR, help="新 stage1 输出目录")
    parser.add_argument("--old-dir", default=DEFAULT_OLD_DIR, help="旧 stage1 输出目录")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Markdown 报告输出路径")
    parser.add_argument("--sample-size", type=int, default=10, help="数值和文本抽样条数")
    parser.add_argument("--seed", type=int, default=20260325, help="随机抽样种子")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="无交互模式下的并行比较线程数")
    parser.add_argument("--large-csv-threshold-mb", type=int, default=20, help="超过该大小的 CSV 使用抽样比较")
    parser.add_argument("--large-csv-sample-rows", type=int, default=10000, help="大 CSV 随机抽样的行数")
    parser.add_argument("--quiet", action="store_true", help="静默模式，仅输出最终结果")
    parser.add_argument("--no-prompt", action="store_true", help="遇到差异时不询问，直接继续")
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (Path(__file__).resolve().parent / path_obj).resolve()


def to_rel_display(path: Path, anchor: Path) -> str:
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def detect_stage(rel_key: str) -> str:
    for stage, prefix in STAGE_RULES:
        if rel_key == prefix or rel_key.startswith(prefix):
            return stage
    return "其他"


def log(message: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def canonical_key(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root).as_posix()
    if file_path.suffix.lower() == ".log":
        return "__stage1_log__"
    return rel


def list_files(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        key = canonical_key(root, file_path)
        if key in EXCLUDED_KEYS:
            continue
        if key in mapping:
            key = file_path.relative_to(root).as_posix()
        mapping[key] = file_path
    return mapping


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_text_with_fallback(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取文本文件: {path}") from last_error


def load_json(path: Path) -> Any:
    return json.loads(read_text_with_fallback(path))


def canonical_json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def count_csv_rows(path: Path) -> int:
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                total_lines = sum(1 for _ in handle)
            return max(0, total_lines - 1)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def read_csv_sample_fallback(path: Path, row_indices: set[int]) -> pd.DataFrame:
    if not row_indices:
        return pd.DataFrame()
    last_error: Exception | None = None
    selected_rows = sorted(row_indices)
    for encoding in READ_ENCODINGS:
        try:
            chunks: list[pd.DataFrame] = []
            seen_rows = 0
            selected_ptr = 0
            for chunk in pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False, low_memory=False, chunksize=50000):
                local_positions: list[int] = []
                chunk_end = seen_rows + len(chunk)
                while selected_ptr < len(selected_rows) and selected_rows[selected_ptr] < chunk_end:
                    local_positions.append(selected_rows[selected_ptr] - seen_rows)
                    selected_ptr += 1
                if local_positions:
                    chunks.append(chunk.iloc[local_positions].copy())
                seen_rows = chunk_end
                if selected_ptr >= len(selected_rows):
                    break
            if not chunks:
                return pd.DataFrame()
            return pd.concat(chunks, ignore_index=True)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def choose_sort_keys(columns: Sequence[str]) -> list[str]:
    return [column for column in KEY_COLUMN_CANDIDATES if column in columns]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized[column] = series.fillna("").astype(str).str.strip()
    return normalized


def sort_frame_if_possible(frame: pd.DataFrame) -> pd.DataFrame:
    sort_keys = choose_sort_keys(frame.columns.tolist())
    if not sort_keys:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


def is_numeric_like(series: pd.Series) -> bool:
    if series.empty:
        return False
    converted = pd.to_numeric(series, errors="coerce")
    non_empty = series.astype(str).str.strip() != ""
    denominator = int(non_empty.sum())
    if denominator == 0:
        return False
    ratio = float(converted.notna().sum()) / float(denominator)
    return ratio >= 0.95


def safe_pct_diff(new_value: float, old_value: float) -> str:
    diff = abs(new_value - old_value)
    if math.isclose(old_value, 0.0, abs_tol=1e-12):
        if math.isclose(new_value, 0.0, abs_tol=1e-12):
            return "0.000000%"
        return "inf"
    return f"{(diff / abs(old_value)) * 100:.6f}%"


def sample_positions(length: int, sample_size: int, rng: random.Random) -> list[int]:
    if length <= 0:
        return []
    if length <= sample_size:
        return list(range(length))
    anchors = {0, length - 1, length // 2}
    remaining = sample_size - len(anchors)
    if remaining > 0:
        anchors.update(rng.sample(range(length), remaining))
    return sorted(anchors)


def rng_for_key(seed: int, rel_key: str) -> random.Random:
    return random.Random(f"{seed}:{rel_key}")


def flatten_json(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key in sorted(obj):
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(obj[key], new_prefix)
        return
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]"
            yield from flatten_json(item, new_prefix)
        return
    yield prefix or "$", obj


def normalize_meta_json_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        normalized: dict[str, Any] = {}
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower.endswith("_dir") or key_lower.endswith("_path") or key_lower in {"vectors_dir", "artifacts_dir", "log_file"}:
                continue
            normalized[str(key)] = normalize_meta_json_obj(value)
        return normalized
    if isinstance(obj, list):
        return [normalize_meta_json_obj(item) for item in obj]
    return obj


def try_parse_meta_json_scalar(value: Any) -> Any | None:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    try:
        return normalize_meta_json_obj(json.loads(value))
    except Exception:
        return None


def compare_json(new_path: Path, old_path: Path, sample_size: int) -> tuple[str, str, list[str]]:
    new_obj = load_json(new_path)
    old_obj = load_json(old_path)
    if canonical_json_text(new_obj) == canonical_json_text(old_obj):
        return "exact", "JSON 内容完全一致", []

    new_flat = dict(flatten_json(new_obj))
    old_flat = dict(flatten_json(old_obj))
    new_keys = set(new_flat)
    old_keys = set(old_flat)
    only_new = sorted(new_keys - old_keys)
    only_old = sorted(old_keys - new_keys)
    common = sorted(new_keys & old_keys)
    diff_keys = [key for key in common if new_flat[key] != old_flat[key]]

    details = [
        f"叶子节点数量: new={len(new_flat)}, old={len(old_flat)}",
        f"仅新目录存在的键路径数: {len(only_new)}",
        f"仅旧目录存在的键路径数: {len(only_old)}",
        f"共同键路径中取值不同的数量: {len(diff_keys)}",
    ]
    for key in only_new[:sample_size]:
        details.append(f"仅新目录键: {key} -> {new_flat[key]!r}")
    for key in only_old[:sample_size]:
        details.append(f"仅旧目录键: {key} -> {old_flat[key]!r}")
    for key in diff_keys[:sample_size]:
        details.append(f"取值不同: {key} | new={new_flat[key]!r} | old={old_flat[key]!r}")
    return "different", "JSON 内容存在差异", details


def compare_text_like(new_path: Path, old_path: Path, sample_size: int, rng: random.Random) -> tuple[str, str, list[str]]:
    new_text = read_text_with_fallback(new_path)
    old_text = read_text_with_fallback(old_path)
    if new_text == old_text:
        return "exact", "文本内容完全一致", []

    new_lines = new_text.splitlines()
    old_lines = old_text.splitlines()
    details = [
        f"行数: new={len(new_lines)}, old={len(old_lines)}",
        f"sha256(new)={hashlib.sha256(new_text.encode('utf-8')).hexdigest()}",
        f"sha256(old)={hashlib.sha256(old_text.encode('utf-8')).hexdigest()}",
    ]
    for idx in sample_positions(min(len(new_lines), len(old_lines)), sample_size, rng):
        if new_lines[idx] != old_lines[idx]:
            details.append(f"第 {idx + 1} 行不同: new={new_lines[idx]!r} | old={old_lines[idx]!r}")
    return "different", "文本内容存在差异", details


def compare_jsonl(new_path: Path, old_path: Path, sample_size: int, rng: random.Random) -> tuple[str, str, list[str]]:
    if sha256_file(new_path) == sha256_file(old_path):
        return "exact", "JSONL 文件字节级一致", []

    new_lines = read_text_with_fallback(new_path).splitlines()
    old_lines = read_text_with_fallback(old_path).splitlines()
    details = [f"行数: new={len(new_lines)}, old={len(old_lines)}"]
    for idx in sample_positions(min(len(new_lines), len(old_lines)), sample_size, rng):
        if new_lines[idx] == old_lines[idx]:
            continue
        try:
            new_obj = json.loads(new_lines[idx])
            old_obj = json.loads(old_lines[idx])
            details.append(f"第 {idx + 1} 行 JSON 不同: new={new_obj!r} | old={old_obj!r}")
        except Exception:
            details.append(f"第 {idx + 1} 行文本不同: new={new_lines[idx]!r} | old={old_lines[idx]!r}")
    return "different", "JSONL 内容存在差异", details


def compare_numeric_series(
    new_series: pd.Series,
    old_series: pd.Series,
    sample_size: int,
    rng: random.Random,
) -> list[str]:
    new_numeric = pd.to_numeric(new_series, errors="coerce")
    old_numeric = pd.to_numeric(old_series, errors="coerce")
    comparable = new_numeric.notna() & old_numeric.notna()
    positions = [int(pos) for pos in np.flatnonzero(comparable.to_numpy())]
    if not positions:
        return []
    sample_positions_list = positions if len(positions) <= sample_size else sorted(rng.sample(positions, sample_size))
    details: list[str] = []
    for pos in sample_positions_list:
        new_value = float(new_numeric.iloc[pos])
        old_value = float(old_numeric.iloc[pos])
        if math.isclose(new_value, old_value, rel_tol=0.0, abs_tol=0.0):
            continue
        details.append(
            f"第 {pos + 1} 行数值不同: new={new_value:.12g}, old={old_value:.12g}, 差值={new_value - old_value:.12g}, 百分比差={safe_pct_diff(new_value, old_value)}"
        )
    return details


def compare_csv(
    new_path: Path,
    old_path: Path,
    sample_size: int,
    rng: random.Random,
    *,
    quiet: bool,
    large_csv_threshold_bytes: int,
    large_csv_sample_rows: int,
) -> tuple[str, str, list[str]]:
    max_size = max(new_path.stat().st_size, old_path.stat().st_size)
    if max_size >= large_csv_threshold_bytes:
        log(f"统计大 CSV 行数: {new_path.name}", quiet=quiet)
        new_total_rows = count_csv_rows(new_path)
        log(f"统计大 CSV 行数: {old_path.name}", quiet=quiet)
        old_total_rows = count_csv_rows(old_path)
        sampled_count = min(large_csv_sample_rows, new_total_rows, old_total_rows)
        sampled_positions = set(sample_positions(min(new_total_rows, old_total_rows), sampled_count, rng))
        log(f"随机抽样读取大 CSV: {new_path.name}", quiet=quiet)
        new_frame = sort_frame_if_possible(normalize_frame(read_csv_sample_fallback(new_path, sampled_positions)))
        log(f"随机抽样读取大 CSV: {old_path.name}", quiet=quiet)
        old_frame = sort_frame_if_possible(normalize_frame(read_csv_sample_fallback(old_path, sampled_positions)))
        details = [
            "比较模式: sampled_random",
            f"文件大小(new)={new_path.stat().st_size} bytes, 文件大小(old)={old_path.stat().st_size} bytes",
            f"总行数(new)={new_total_rows}, 总行数(old)={old_total_rows}",
            f"抽样行数上限: {large_csv_sample_rows}",
            f"样本行数(new)={len(new_frame)}, 样本行数(old)={len(old_frame)}",
            f"列数: new={new_frame.shape[1]}, old={old_frame.shape[1]}",
            f"列名(new): {', '.join(new_frame.columns)}",
            f"列名(old): {', '.join(old_frame.columns)}",
        ]
        if list(new_frame.columns) != list(old_frame.columns):
            only_new = [column for column in new_frame.columns if column not in old_frame.columns]
            only_old = [column for column in old_frame.columns if column not in new_frame.columns]
            if only_new:
                details.append(f"仅新目录列: {only_new}")
            if only_old:
                details.append(f"仅旧目录列: {only_old}")
            return "different", "大 CSV 列结构不同", details
        if new_frame.shape != old_frame.shape:
            return "different", "大 CSV 抽样行数或列数不同", details
        if new_frame.equals(old_frame):
            return "sampled_match", "大 CSV 抽样内容一致", details

        numeric_columns = [column for column in new_frame.columns if is_numeric_like(new_frame[column]) and is_numeric_like(old_frame[column])]
        string_columns = [column for column in new_frame.columns if column not in numeric_columns]
        mismatched_numeric_examples: list[str] = []
        for column in numeric_columns:
            mismatched_numeric_examples.extend(
                f"{column}: {message}" for message in compare_numeric_series(new_frame[column], old_frame[column], sample_size, rng)
            )
            if len(mismatched_numeric_examples) >= sample_size:
                break
        string_diff_examples: list[str] = []
        for column in string_columns:
            mismatches = new_frame[column] != old_frame[column]
            mismatch_positions = [int(pos) for pos in np.flatnonzero(mismatches.to_numpy())]
            if not mismatch_positions:
                continue
            sample_positions_list = mismatch_positions if len(mismatch_positions) <= sample_size else sorted(rng.sample(mismatch_positions, sample_size))
            for pos in sample_positions_list:
                string_diff_examples.append(
                    f"{column} 第 {pos + 1} 行不同: new={new_frame.iloc[pos][column]!r} | old={old_frame.iloc[pos][column]!r}"
                )
                if len(string_diff_examples) >= sample_size:
                    break
            if len(string_diff_examples) >= sample_size:
                break
        details.extend(mismatched_numeric_examples[:sample_size])
        details.extend(string_diff_examples[:sample_size])
        return "different", "大 CSV 抽样内容存在差异", details

    log(f"读取 CSV: {new_path.name}", quiet=quiet)
    new_frame = sort_frame_if_possible(normalize_frame(read_csv_fallback(new_path)))
    log(f"读取 CSV: {old_path.name}", quiet=quiet)
    old_frame = sort_frame_if_possible(normalize_frame(read_csv_fallback(old_path)))

    details = [
        f"行数: new={len(new_frame)}, old={len(old_frame)}",
        f"列数: new={new_frame.shape[1]}, old={old_frame.shape[1]}",
        f"列名(new): {', '.join(new_frame.columns)}",
        f"列名(old): {', '.join(old_frame.columns)}",
    ]

    if list(new_frame.columns) != list(old_frame.columns):
        only_new = [column for column in new_frame.columns if column not in old_frame.columns]
        only_old = [column for column in old_frame.columns if column not in new_frame.columns]
        if only_new:
            details.append(f"仅新目录列: {only_new}")
        if only_old:
            details.append(f"仅旧目录列: {only_old}")
        return "different", "CSV 列结构不同", details

    if new_frame.shape != old_frame.shape:
        return "different", "CSV 行数或列数不同", details

    if new_frame.equals(old_frame):
        return "exact", "CSV 内容完全一致", details

    numeric_columns = [column for column in new_frame.columns if is_numeric_like(new_frame[column]) and is_numeric_like(old_frame[column])]
    string_columns = [column for column in new_frame.columns if column not in numeric_columns]

    mismatched_numeric_examples: list[str] = []
    for column in numeric_columns:
        mismatched_numeric_examples.extend(f"{column}: {message}" for message in compare_numeric_series(new_frame[column], old_frame[column], sample_size, rng))
        if len(mismatched_numeric_examples) >= sample_size:
            break

    string_diff_examples: list[str] = []
    for column in string_columns:
        mismatches = new_frame[column] != old_frame[column]
        mismatch_positions = [int(pos) for pos in np.flatnonzero(mismatches.to_numpy())]
        if not mismatch_positions:
            continue
        sample_positions_list = mismatch_positions if len(mismatch_positions) <= sample_size else sorted(rng.sample(mismatch_positions, sample_size))
        for pos in sample_positions_list:
            string_diff_examples.append(
                f"{column} 第 {pos + 1} 行不同: new={new_frame.iloc[pos][column]!r} | old={old_frame.iloc[pos][column]!r}"
            )
            if len(string_diff_examples) >= sample_size:
                break
        if len(string_diff_examples) >= sample_size:
            break

    details.append(f"检测到数值列数量: {len(numeric_columns)}")
    details.extend(mismatched_numeric_examples[:sample_size])
    details.extend(string_diff_examples[:sample_size])
    return "different", "CSV 内容存在差异", details


def format_array_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return repr(value)


def compare_numeric_arrays(new_array: np.ndarray, old_array: np.ndarray, sample_size: int, rng: random.Random) -> list[str]:
    if new_array.shape != old_array.shape:
        return [f"数组形状不同: new={new_array.shape}, old={old_array.shape}"]
    if new_array.size == 0:
        return []

    flat_new = new_array.reshape(-1)
    flat_old = old_array.reshape(-1)
    sample_idx = sample_positions(len(flat_new), sample_size, rng)
    details: list[str] = []
    for idx in sample_idx:
        new_value = float(flat_new[idx])
        old_value = float(flat_old[idx])
        if math.isclose(new_value, old_value, rel_tol=0.0, abs_tol=0.0):
            continue
        details.append(
            f"位置 {idx} 不同: new={new_value:.12g}, old={old_value:.12g}, 差值={new_value - old_value:.12g}, 百分比差={safe_pct_diff(new_value, old_value)}"
        )
    return details


def compare_ndarray(new_array: np.ndarray, old_array: np.ndarray, sample_size: int, rng: random.Random) -> tuple[str, str, list[str]]:
    details = [
        f"shape(new)={new_array.shape}, shape(old)={old_array.shape}",
        f"dtype(new)={new_array.dtype}, dtype(old)={old_array.dtype}",
    ]
    if new_array.shape != old_array.shape:
        return "different", "数组形状不同", details
    if new_array.dtype != old_array.dtype:
        details.append("dtype 不同，但继续比较内容")

    both_numeric = np.issubdtype(new_array.dtype, np.number) and np.issubdtype(old_array.dtype, np.number)
    if both_numeric:
        if np.array_equal(new_array, old_array, equal_nan=True):
            return "exact", "数组内容完全一致", details
        details.extend(compare_numeric_arrays(new_array.astype(np.float64, copy=False), old_array.astype(np.float64, copy=False), sample_size, rng))
    else:
        if np.array_equal(new_array, old_array):
            return "exact", "数组内容完全一致", details
        sample_idx = sample_positions(new_array.size, sample_size, rng)
        flat_new = new_array.reshape(-1)
        flat_old = old_array.reshape(-1)
        for idx in sample_idx:
            if flat_new[idx] != flat_old[idx]:
                details.append(f"位置 {idx} 不同: new={format_array_value(flat_new[idx])} | old={format_array_value(flat_old[idx])}")

    return "different", "数组内容存在差异", details


def compare_meta_json_array(new_array: np.ndarray, old_array: np.ndarray, sample_size: int) -> tuple[str, str, list[str]]:
    new_obj = try_parse_meta_json_scalar(new_array)
    old_obj = try_parse_meta_json_scalar(old_array)
    details = [
        f"shape(new)={new_array.shape}, shape(old)={old_array.shape}",
        f"dtype(new)={new_array.dtype}, dtype(old)={old_array.dtype}",
    ]
    if new_obj is None or old_obj is None:
        return compare_ndarray(new_array, old_array, sample_size, random.Random(0))
    if canonical_json_text(new_obj) == canonical_json_text(old_obj):
        details.append("忽略目录/路径字段后的 metadata 一致")
        return "exact", "metadata 内容一致", details

    new_flat = dict(flatten_json(new_obj))
    old_flat = dict(flatten_json(old_obj))
    new_keys = set(new_flat)
    old_keys = set(old_flat)
    only_new = sorted(new_keys - old_keys)
    only_old = sorted(old_keys - new_keys)
    common = sorted(new_keys & old_keys)
    diff_keys = [key for key in common if new_flat[key] != old_flat[key]]
    details.append(f"忽略目录/路径字段后仍有差异键数: {len(diff_keys)}")
    for key in only_new[:sample_size]:
        details.append(f"仅新 metadata 键: {key} -> {new_flat[key]!r}")
    for key in only_old[:sample_size]:
        details.append(f"仅旧 metadata 键: {key} -> {old_flat[key]!r}")
    for key in diff_keys[:sample_size]:
        details.append(f"metadata 取值不同: {key} | new={new_flat[key]!r} | old={old_flat[key]!r}")
    return "different", "metadata 内容存在差异", details


def compare_npy(
    new_path: Path,
    old_path: Path,
    sample_size: int,
    rng: random.Random,
    *,
    quiet: bool,
) -> tuple[str, str, list[str]]:
    log(f"加载 NPY: {new_path.name}", quiet=quiet)
    new_array = np.load(new_path, allow_pickle=True)
    log(f"加载 NPY: {old_path.name}", quiet=quiet)
    old_array = np.load(old_path, allow_pickle=True)
    return compare_ndarray(new_array, old_array, sample_size, rng)


def compare_npz(
    new_path: Path,
    old_path: Path,
    sample_size: int,
    rng: random.Random,
    *,
    quiet: bool,
) -> tuple[str, str, list[str]]:
    log(f"加载 NPZ: {new_path.name}", quiet=quiet)
    with np.load(new_path, allow_pickle=True) as new_obj, np.load(old_path, allow_pickle=True) as old_obj:
        new_keys = sorted(new_obj.files)
        old_keys = sorted(old_obj.files)
        details = [f"keys(new)={new_keys}", f"keys(old)={old_keys}"]
        if new_keys != old_keys:
            return "different", "NPZ 键集合不同", details

        mismatch_found = False
        for key in new_keys:
            if key == "meta_json":
                status, summary, sub_details = compare_meta_json_array(new_obj[key], old_obj[key], sample_size)
            else:
                status, summary, sub_details = compare_ndarray(new_obj[key], old_obj[key], sample_size, rng)
            details.append(f"[{key}] {summary}")
            details.extend(f"[{key}] {item}" for item in sub_details[:sample_size])
            if status != "exact":
                mismatch_found = True
        if mismatch_found:
            return "different", "NPZ 内至少一个数组存在差异", details
        return "exact", "NPZ 所有数组完全一致", details


def compare_generic(new_path: Path, old_path: Path) -> tuple[str, str, list[str]]:
    new_hash = sha256_file(new_path)
    old_hash = sha256_file(old_path)
    details = [
        f"size(new)={new_path.stat().st_size} bytes, size(old)={old_path.stat().st_size} bytes",
        f"sha256(new)={new_hash}",
        f"sha256(old)={old_hash}",
    ]
    if new_hash == old_hash:
        return "exact", "文件字节级一致", details
    return "different", "文件字节级不同", details


def compare_file(
    new_path: Path | None,
    old_path: Path | None,
    rel_key: str,
    sample_size: int,
    seed: int,
    repo_root: Path,
    *,
    quiet: bool,
    large_csv_threshold_bytes: int,
    large_csv_sample_rows: int,
) -> FileComparison:
    rng = rng_for_key(seed, rel_key)
    stage = detect_stage(rel_key)
    new_rel = to_rel_display(new_path, repo_root) if new_path is not None else None
    old_rel = to_rel_display(old_path, repo_root) if old_path is not None else None

    if new_path is None:
        return FileComparison(stage, rel_key, None, old_rel, "missing_in_new", "missing", "新目录缺失该文件")
    if old_path is None:
        return FileComparison(stage, rel_key, new_rel, None, "missing_in_old", "missing", "旧目录缺失该文件")

    suffix = new_path.suffix.lower()
    if suffix == ".json":
        status, summary, details = compare_json(new_path, old_path, sample_size)
        file_type = "json"
    elif suffix == ".csv":
        status, summary, details = compare_csv(
            new_path,
            old_path,
            sample_size,
            rng,
            quiet=quiet,
            large_csv_threshold_bytes=large_csv_threshold_bytes,
            large_csv_sample_rows=large_csv_sample_rows,
        )
        file_type = "csv"
    elif suffix == ".jsonl":
        status, summary, details = compare_jsonl(new_path, old_path, sample_size, rng)
        file_type = "jsonl"
    elif suffix == ".npy":
        status, summary, details = compare_npy(new_path, old_path, sample_size, rng, quiet=quiet)
        file_type = "npy"
    elif suffix == ".npz":
        status, summary, details = compare_npz(new_path, old_path, sample_size, rng, quiet=quiet)
        file_type = "npz"
    elif suffix == ".log":
        status, summary, details = compare_text_like(new_path, old_path, sample_size, rng)
        file_type = "log"
    else:
        status, summary, details = compare_generic(new_path, old_path)
        file_type = suffix.lstrip(".") or "binary"

    return FileComparison(stage, rel_key, new_rel, old_rel, status, file_type, summary, details)


def print_difference(result: FileComparison, *, quiet: bool) -> None:
    log(f"发现差异: {result.rel_key}", quiet=quiet)
    log(f"类型: {result.file_type} | 结论: {result.summary}", quiet=quiet)
    if result.new_rel is not None:
        log(f"新文件: {result.new_rel}", quiet=quiet)
    if result.old_rel is not None:
        log(f"旧文件: {result.old_rel}", quiet=quiet)
    if result.details:
        log("差异摘要如下:", quiet=quiet)
        for item in result.details:
            print(f"  - {item}", flush=True)


def is_effectively_exact(status: str) -> bool:
    return status in {"exact", "sampled_match"}


def prompt_user_on_difference(result: FileComparison, *, quiet: bool) -> bool:
    print_difference(result, quiet=quiet)
    while True:
        answer = input("发现差异。输入 1 继续比较，输入 2 停止后续比较: ").strip()
        if answer == "1":
            return True
        if answer == "2":
            return False
        print("无效输入，请输入 1 或 2。", flush=True)


def compare_directories(
    new_dir: Path,
    old_dir: Path,
    sample_size: int,
    seed: int,
    repo_root: Path,
    *,
    quiet: bool,
    prompt_on_diff: bool,
    workers: int,
    large_csv_threshold_bytes: int,
    large_csv_sample_rows: int,
) -> ComparisonRun:
    log(f"扫描新目录文件: {new_dir}", quiet=quiet)
    new_files = list_files(new_dir)
    log(f"扫描旧目录文件: {old_dir}", quiet=quiet)
    old_files = list_files(old_dir)
    all_keys = sorted(set(new_files) | set(old_files), key=lambda key: (STAGE_ORDER.get(detect_stage(key), 999), key))
    log(f"待比较文件数: {len(all_keys)}", quiet=quiet)

    if not prompt_on_diff and workers > 1:
        log(f"无交互模式启用并行比较 workers={workers}", quiet=quiet)
        indexed_keys = list(enumerate(all_keys, start=1))
        results_by_index: dict[int, FileComparison] = {}
        stage_announced: set[str] = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    compare_file,
                    new_files.get(key),
                    old_files.get(key),
                    key,
                    sample_size,
                    seed,
                    repo_root,
                    quiet=quiet,
                    large_csv_threshold_bytes=large_csv_threshold_bytes,
                    large_csv_sample_rows=large_csv_sample_rows,
                ): (idx, key)
                for idx, key in indexed_keys
            }
            for future in as_completed(future_map):
                idx, key = future_map[future]
                result = future.result()
                stage = detect_stage(key)
                if stage not in stage_announced:
                    stage_announced.add(stage)
                    log(f"进入 {stage}", quiet=quiet)
                log(f"[{idx}/{len(all_keys)}] 完成 {key} -> {result.status}", quiet=quiet)
                results_by_index[idx] = result
                if not is_effectively_exact(result.status) and not quiet:
                    print_difference(result, quiet=quiet)
        ordered_results = [results_by_index[idx] for idx, _ in indexed_keys]
        return ComparisonRun(results=ordered_results)

    results: list[FileComparison] = []
    current_stage: str | None = None
    for idx, key in enumerate(all_keys, start=1):
        stage = detect_stage(key)
        if stage != current_stage:
            current_stage = stage
            log(f"进入 {stage}", quiet=quiet)
        log(f"[{idx}/{len(all_keys)}] 比较 {key}", quiet=quiet)
        result = compare_file(
            new_files.get(key),
            old_files.get(key),
            key,
            sample_size,
            seed,
            repo_root,
            quiet=quiet,
            large_csv_threshold_bytes=large_csv_threshold_bytes,
            large_csv_sample_rows=large_csv_sample_rows,
        )
        log(f"[{idx}/{len(all_keys)}] 完成 {key} -> {result.status}", quiet=quiet)
        results.append(result)
        if not is_effectively_exact(result.status):
            if prompt_on_diff and not quiet:
                if not prompt_user_on_difference(result, quiet=quiet):
                    return ComparisonRun(results=results, stopped_on=result)
            elif not quiet:
                print_difference(result, quiet=quiet)
    return ComparisonRun(results=results)


def stage_header(stage: str) -> str:
    idx = STAGE_ORDER.get(stage)
    if idx is None:
        return stage
    return stage


def summarize_status(results: Sequence[FileComparison]) -> list[str]:
    counts: dict[str, int] = {}
    for result in results:
        status = "exact" if is_effectively_exact(result.status) else result.status
        counts[status] = counts.get(status, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: item[0])
    return [f"{status}: {count}" for status, count in ordered]


def build_report(
    new_dir: Path,
    old_dir: Path,
    results: Sequence[FileComparison],
    seed: int,
    sample_size: int,
    repo_root: Path,
    *,
    stopped_on: FileComparison | None,
) -> str:
    grouped_all: dict[str, list[FileComparison]] = {}
    for result in results:
        grouped_all.setdefault(result.stage, []).append(result)
    non_exact_results = [result for result in results if not is_effectively_exact(result.status)]
    lines = [
        "# Stage1 对比报告",
        "",
        f"- 新目录: `{to_rel_display(new_dir, repo_root)}`",
        f"- 旧目录: `{to_rel_display(old_dir, repo_root)}`",
        f"- 抽样大小: `{sample_size}`",
        f"- 随机种子: `{seed}`",
        f"- 已排除文件键: `{', '.join(sorted(EXCLUDED_KEYS))}`",
        f"- 文件总数: `{len(results)}`",
        f"- 非完全一致文件数: `{len(non_exact_results)}`",
        f"- 状态汇总: `{'; '.join(summarize_status(results))}`",
        "",
    ]
    if stopped_on is not None:
        lines.insert(-1, f"- 比较已中途停止于: `{stopped_on.rel_key}`")

    for stage in sorted(grouped_all, key=lambda name: STAGE_ORDER.get(name, 999)):
        stage_results = grouped_all[stage]
        stage_non_exact = [result for result in stage_results if not is_effectively_exact(result.status)]
        lines.append(f"## {stage_header(stage)}")
        lines.append("")
        lines.append(f"- 文件数: `{len(stage_results)}`")
        lines.append(f"- 状态汇总: `{'; '.join(summarize_status(stage_results))}`")
        lines.append("")
        if not stage_non_exact:
            lines.append("该阶段比较结果：全部一致。")
            lines.append("")
            continue

        for result in sorted(stage_non_exact, key=lambda item: item.rel_key):
            lines.append(f"### {result.rel_key}")
            lines.append("")
            lines.append(f"- 状态: `{result.status}`")
            lines.append(f"- 类型: `{result.file_type}`")
            if result.new_rel is not None:
                lines.append(f"- 新文件: `{result.new_rel}`")
            if result.old_rel is not None:
                lines.append(f"- 旧文件: `{result.old_rel}`")
            lines.append(f"- 结论: {result.summary}")
            if result.details:
                lines.append("- 细节:")
                for item in result.details:
                    lines.append(f"  - {item}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    new_dir = resolve_repo_path(args.new_dir)
    old_dir = resolve_repo_path(args.old_dir)
    report_path = resolve_repo_path(args.report)

    log("开始对比 stage1 输出目录", quiet=args.quiet)
    log(f"新目录: {new_dir}", quiet=args.quiet)
    log(f"旧目录: {old_dir}", quiet=args.quiet)
    log(f"报告路径: {report_path}", quiet=args.quiet)

    if not new_dir.exists():
        raise FileNotFoundError(f"新目录不存在: {new_dir}")
    if not old_dir.exists():
        raise FileNotFoundError(f"旧目录不存在: {old_dir}")

    run = compare_directories(
        new_dir=new_dir,
        old_dir=old_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        repo_root=repo_root,
        quiet=args.quiet,
        prompt_on_diff=not args.no_prompt,
        workers=max(1, args.workers),
        large_csv_threshold_bytes=max(1, args.large_csv_threshold_mb) * 1024 * 1024,
        large_csv_sample_rows=max(1, args.large_csv_sample_rows),
    )
    log("比较完成，开始生成报告", quiet=args.quiet)
    report = build_report(
        new_dir,
        old_dir,
        run.results,
        args.seed,
        args.sample_size,
        repo_root,
        stopped_on=run.stopped_on,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    log("报告写入完成", quiet=args.quiet)
    print(f"报告已写入: {report_path}")
    print(f"共比较文件: {len(run.results)}")
    print(f"状态汇总: {'; '.join(summarize_status(run.results))}")
    if run.stopped_on is not None:
        print(f"比较已停止于: {run.stopped_on.rel_key}")


if __name__ == "__main__":
    main()
