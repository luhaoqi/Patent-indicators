from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import sparse

from .config import Config
from .project_paths import build_experiment_layout, resolve_project_path


EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
DEFAULT_TOP_N = 100
DEFAULT_BOTTOM_N = 10


class PatentSimilarityCaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stage1Mode:
    exact_date: bool
    year_col: str
    date_col: Optional[str] = None
    date_ord_col: Optional[str] = None


@dataclass(frozen=True)
class PatentRow:
    row: int
    application_no: str
    year: int
    title: str
    date_value: Optional[str] = None
    date_ord: Optional[int] = None
    extras: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SimilarityRecord:
    year: int
    application_no: str
    title: str
    row: int
    similarity: float
    counted_in_bsfs: bool
    shared_term_count: int
    relative_year_gap: int
    date_value: Optional[str] = None
    relative_day_gap: Optional[int] = None


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)


def _date_from_ord(day_ord: int) -> date:
    return date.fromordinal(EPOCH_ORDINAL + int(day_ord))


def _day_ord_from_date(value: date) -> int:
    return value.toordinal() - EPOCH_ORDINAL


def _add_years(base: date, years: int) -> date:
    try:
        return base.replace(year=base.year + years)
    except ValueError:
        return base.replace(month=2, day=28, year=base.year + years)


def _extract_meta_json(value: Any) -> Dict[str, Any]:
    raw = value
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _has_index_files(stage1_dir: Path) -> bool:
    index_dir = stage1_dir / "index"
    return index_dir.exists() and any(index_dir.glob("year=*.csv"))


def resolve_stage1_dir(
    *,
    stage1_dir: Optional[str],
    experiment_id: Optional[str],
    output_root: str,
) -> Path:
    if stage1_dir:
        resolved = resolve_project_path(stage1_dir)
        if not _has_index_files(resolved):
            raise PatentSimilarityCaseError(f"给定目录下找不到 stage1 index 文件: {resolved}")
        return resolved

    if not experiment_id:
        raise PatentSimilarityCaseError("必须提供 --stage1-dir 或 --experiment-id")

    layout = build_experiment_layout(experiment_id, output_root=output_root)
    stage1 = layout.stage1_dir
    stage1_exact = layout.stage1_exact_dir
    stage1_exists = _has_index_files(stage1)
    stage1_exact_exists = _has_index_files(stage1_exact)

    if stage1_exists and not stage1_exact_exists:
        return stage1
    if stage1_exact_exists and not stage1_exists:
        return stage1_exact
    if stage1_exists and stage1_exact_exists:
        if "exact" in experiment_id.lower():
            return stage1_exact
        return stage1
    raise PatentSimilarityCaseError(f"实验目录下找不到可用的 stage1 输出: {layout.root}")


def detect_stage1_mode(stage1_dir: Path) -> Stage1Mode:
    index_dir = stage1_dir / "index"
    sample_paths = sorted(index_dir.glob("year=*.csv"))
    if not sample_paths:
        raise PatentSimilarityCaseError(f"找不到 stage1 index 文件: {index_dir}")

    with sample_paths[0].open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

    if {"公开公告年份", "公开公告日", "公开公告日_ord"}.issubset(fieldnames):
        return Stage1Mode(
            exact_date=True,
            year_col="公开公告年份",
            date_col="公开公告日",
            date_ord_col="公开公告日_ord",
        )
    if "申请年份" in fieldnames:
        return Stage1Mode(exact_date=False, year_col="申请年份")
    raise PatentSimilarityCaseError(f"无法识别 stage1 index 列结构: {sample_paths[0]}")


def resolve_vectors_base(stage1_dir: Path) -> Path:
    vectors_filtered = stage1_dir / "vectors_filtered"
    if vectors_filtered.exists() and any(vectors_filtered.glob("year=*.npz")):
        return vectors_filtered
    vectors = stage1_dir / "vectors"
    if vectors.exists() and any(vectors.glob("year=*.npz")):
        return vectors
    raise PatentSimilarityCaseError(f"在 {stage1_dir} 下找不到可用向量目录（vectors_filtered / vectors）")


def infer_runtime_params(stage1_dir: Path) -> Dict[str, Any]:
    default_cfg = Config(data_path=".")
    result: Dict[str, Any] = {
        "window_size": int(default_cfg.window_size),
        "similarity_threshold": float(default_cfg.similarity_threshold),
    }

    pair_dir = stage1_dir / "pair_contrib"
    if pair_dir.exists():
        candidates = sorted(pair_dir.glob("x=*_y=*.npz")) + sorted(pair_dir.glob("same_year=*.npz"))
        for path in candidates:
            try:
                obj = np.load(path, allow_pickle=True)
                meta = _extract_meta_json(obj["meta_json"])
            except Exception:
                continue
            if "window_size" in meta:
                result["window_size"] = int(meta["window_size"])
            if "thr" in meta:
                result["similarity_threshold"] = float(meta["thr"])
            return result

    pair_list_path = stage1_dir / "pair_list.json"
    if pair_list_path.exists():
        try:
            pair_list_obj = json.loads(pair_list_path.read_text(encoding="utf-8"))
            pairs = pair_list_obj.get("pairs", [])
            if pairs:
                result["window_size"] = max(abs(int(y) - int(x)) for x, y in pairs)
        except Exception:
            pass
    return result


def load_index_year(stage1_dir: Path, year: int, mode: Stage1Mode) -> List[PatentRow]:
    index_path = stage1_dir / "index" / f"year={year}.csv"
    if not index_path.exists():
        return []

    rows: List[PatentRow] = []
    with index_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            date_value = raw.get(mode.date_col, "").strip() if mode.date_col else None
            date_ord_text = raw.get(mode.date_ord_col, "").strip() if mode.date_ord_col else ""
            extras = {
                key: value
                for key, value in raw.items()
                if key not in {"row", "申请号", mode.year_col, "专利名称", mode.date_col, mode.date_ord_col}
            }
            rows.append(
                PatentRow(
                    row=int(raw["row"]),
                    application_no=raw["申请号"].strip(),
                    year=int(raw[mode.year_col]),
                    title=raw["专利名称"].strip(),
                    date_value=date_value or None,
                    date_ord=int(date_ord_text) if date_ord_text else None,
                    extras=extras,
                )
            )
    return rows


def locate_target_patent(
    *,
    stage1_dir: Path,
    mode: Stage1Mode,
    application_no: str,
    year: int,
    date_value: Optional[str] = None,
    title: Optional[str] = None,
) -> PatentRow:
    matches = [row for row in load_index_year(stage1_dir, year, mode) if row.application_no == application_no]
    if date_value:
        matches = [row for row in matches if row.date_value == date_value]
    if title:
        matches = [row for row in matches if row.title == title]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise PatentSimilarityCaseError(
            f"目标专利 {application_no} 未在 {stage1_dir / 'index' / f'year={year}.csv'} 中找到。"
        )

    choices = [
        {
            "row": row.row,
            "title": row.title,
            "date_value": row.date_value,
        }
        for row in matches
    ]
    raise PatentSimilarityCaseError(
        f"目标专利 {application_no} 在年份 {year} 中命中多条记录，请补充 --date 或 --title。候选: {choices}"
    )


def load_target_tokens(stage1_dir: Path, target: PatentRow) -> List[str]:
    token_path = stage1_dir / "tokens" / f"year={target.year}.jsonl"
    if not token_path.exists():
        raise PatentSimilarityCaseError(f"找不到 token 文件: {token_path}")

    matches: List[List[str]] = []
    with token_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("id", "")).strip() != target.application_no:
                continue
            if target.title and str(obj.get("title", "")).strip() != target.title:
                continue
            matches.append(list(obj.get("tokens", [])))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise PatentSimilarityCaseError(f"在 token 文件中找不到目标专利: {target.application_no}")
    raise PatentSimilarityCaseError(
        f"目标专利 {target.application_no} 在 tokens/year={target.year}.jsonl 中命中多条记录，请补充更精确条件。"
    )


def load_target_vector(vectors_base: Path, target: PatentRow) -> sparse.csr_matrix:
    matrix_path = vectors_base / f"year={target.year}.npz"
    if not matrix_path.exists():
        raise PatentSimilarityCaseError(f"缺少目标年份向量文件: {matrix_path}")
    matrix = sparse.load_npz(matrix_path).tocsr()
    if target.row >= matrix.shape[0]:
        raise PatentSimilarityCaseError(
            f"目标 row={target.row} 超出年份 {target.year} 的向量矩阵范围 {matrix.shape[0]}"
        )
    return matrix.getrow(target.row).tocsr()


def load_vocab(stage1_dir: Path) -> Dict[str, int]:
    vocab_path = stage1_dir / "vocab" / "final_vocab.json"
    if not vocab_path.exists():
        raise PatentSimilarityCaseError(f"找不到词表文件: {vocab_path}")
    obj = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = obj.get("vocab")
    if not isinstance(vocab, dict):
        raise PatentSimilarityCaseError(f"词表文件格式不合法: {vocab_path}")
    return {str(term): int(index) for term, index in vocab.items()}


def _collect_standard_window(
    *,
    stage1_dir: Path,
    mode: Stage1Mode,
    target: PatentRow,
    window_size: int,
) -> Dict[str, Any]:
    backward: List[PatentRow] = []
    forward: List[PatentRow] = []
    for year in range(target.year - window_size, target.year):
        backward.extend(load_index_year(stage1_dir, year, mode))
    for year in range(target.year + 1, target.year + window_size + 1):
        forward.extend(load_index_year(stage1_dir, year, mode))
    return {
        "backward_rows": backward,
        "forward_rows": forward,
        "window_start_year": target.year - window_size,
        "window_end_year": target.year + window_size,
    }


def _collect_exact_window(
    *,
    stage1_dir: Path,
    mode: Stage1Mode,
    target: PatentRow,
    window_size: int,
) -> Dict[str, Any]:
    if target.date_ord is None:
        raise PatentSimilarityCaseError("exact-date 模式缺少目标专利公开日期序数。")

    target_day = _date_from_ord(target.date_ord)
    backward_start_ord = _day_ord_from_date(_add_years(target_day, -window_size))
    forward_end_ord = _day_ord_from_date(_add_years(target_day, window_size))

    backward: List[PatentRow] = []
    forward: List[PatentRow] = []
    for year in range(target.year - window_size, target.year + window_size + 1):
        for row in load_index_year(stage1_dir, year, mode):
            if row.application_no == target.application_no and row.year == target.year and row.row == target.row:
                continue
            if row.date_ord is None:
                continue
            if backward_start_ord <= row.date_ord < target.date_ord:
                backward.append(row)
            elif target.date_ord < row.date_ord <= forward_end_ord:
                forward.append(row)

    return {
        "backward_rows": backward,
        "forward_rows": forward,
        "window_start_date": _date_from_ord(backward_start_ord).isoformat(),
        "window_end_date": _date_from_ord(forward_end_ord).isoformat(),
    }


def collect_window_candidates(
    *,
    stage1_dir: Path,
    mode: Stage1Mode,
    target: PatentRow,
    window_size: int,
) -> Dict[str, Any]:
    if mode.exact_date:
        return _collect_exact_window(
            stage1_dir=stage1_dir,
            mode=mode,
            target=target,
            window_size=window_size,
        )
    return _collect_standard_window(
        stage1_dir=stage1_dir,
        mode=mode,
        target=target,
        window_size=window_size,
    )


def _group_rows_by_year(rows: Sequence[PatentRow]) -> Dict[int, List[PatentRow]]:
    grouped: Dict[int, List[PatentRow]] = defaultdict(list)
    for row in rows:
        grouped[row.year].append(row)
    return dict(grouped)


def compute_similarity_records(
    *,
    vectors_base: Path,
    mode: Stage1Mode,
    target: PatentRow,
    target_vec: sparse.csr_matrix,
    candidates: Sequence[PatentRow],
    similarity_threshold: float,
) -> Dict[str, Any]:
    if not candidates:
        return {
            "records": [],
            "raw_term_sums": np.zeros(target_vec.nnz, dtype=np.float64),
            "counted_term_sums": np.zeros(target_vec.nnz, dtype=np.float64),
            "raw_term_match_counts": np.zeros(target_vec.nnz, dtype=np.int64),
            "counted_term_match_counts": np.zeros(target_vec.nnz, dtype=np.int64),
        }

    target_term_indices = target_vec.indices.astype(np.int64, copy=False)
    target_term_weights = target_vec.data.astype(np.float64, copy=False)
    if target_term_indices.size == 0:
        return {
            "records": [],
            "raw_term_sums": np.zeros(0, dtype=np.float64),
            "counted_term_sums": np.zeros(0, dtype=np.float64),
            "raw_term_match_counts": np.zeros(0, dtype=np.int64),
            "counted_term_match_counts": np.zeros(0, dtype=np.int64),
        }

    records: List[SimilarityRecord] = []
    raw_term_sums = np.zeros(target_term_indices.size, dtype=np.float64)
    counted_term_sums = np.zeros(target_term_indices.size, dtype=np.float64)
    raw_term_match_counts = np.zeros(target_term_indices.size, dtype=np.int64)
    counted_term_match_counts = np.zeros(target_term_indices.size, dtype=np.int64)

    for year, rows_in_year in _group_rows_by_year(candidates).items():
        matrix_path = vectors_base / f"year={year}.npz"
        if not matrix_path.exists():
            raise PatentSimilarityCaseError(f"缺少候选年份向量文件: {matrix_path}")

        year_matrix = sparse.load_npz(matrix_path).tocsr()
        row_ids = [row.row for row in rows_in_year]
        max_row = max(row_ids)
        if max_row >= year_matrix.shape[0]:
            raise PatentSimilarityCaseError(
                f"年份 {year} 候选 row={max_row} 超出向量矩阵范围 {year_matrix.shape[0]}"
            )

        sub_matrix = year_matrix[row_ids]
        term_sub_matrix = sub_matrix[:, target_term_indices].tocsr()
        scores = np.asarray(sub_matrix.dot(target_vec.T).toarray()).reshape(-1).astype(np.float64, copy=False)
        shared_term_counts = np.asarray(term_sub_matrix.getnnz(axis=1)).reshape(-1)

        coo = term_sub_matrix.tocoo()
        if coo.nnz > 0:
            term_contrib_values = coo.data.astype(np.float64, copy=False) * target_term_weights[coo.col]
            raw_term_sums += np.bincount(
                coo.col,
                weights=term_contrib_values,
                minlength=target_term_indices.size,
            )
            raw_term_match_counts += np.bincount(
                coo.col,
                minlength=target_term_indices.size,
            ).astype(np.int64, copy=False)

            counted_mask = scores >= float(similarity_threshold)
            if np.any(counted_mask):
                nnz_mask = counted_mask[coo.row]
                counted_term_sums += np.bincount(
                    coo.col[nnz_mask],
                    weights=term_contrib_values[nnz_mask],
                    minlength=target_term_indices.size,
                )
                counted_term_match_counts += np.bincount(
                    coo.col[nnz_mask],
                    minlength=target_term_indices.size,
                ).astype(np.int64, copy=False)
        else:
            counted_mask = scores >= float(similarity_threshold)

        for row, score, shared_count in zip(rows_in_year, scores, shared_term_counts):
            relative_day_gap = None
            if mode.exact_date and row.date_ord is not None and target.date_ord is not None:
                relative_day_gap = int(row.date_ord - target.date_ord)
            records.append(
                SimilarityRecord(
                    year=row.year,
                    application_no=row.application_no,
                    title=row.title,
                    row=row.row,
                    similarity=float(score),
                    counted_in_bsfs=bool(score >= float(similarity_threshold)),
                    shared_term_count=int(shared_count),
                    relative_year_gap=int(row.year - target.year),
                    date_value=row.date_value,
                    relative_day_gap=relative_day_gap,
                )
            )

    records.sort(key=lambda item: (-item.similarity, item.year, item.date_value or "", item.application_no, item.row))
    return {
        "records": records,
        "raw_term_sums": raw_term_sums,
        "counted_term_sums": counted_term_sums,
        "raw_term_match_counts": raw_term_match_counts,
        "counted_term_match_counts": counted_term_match_counts,
    }


def build_term_contribution_rows(
    *,
    target_tokens: Sequence[str],
    vocab: Dict[str, int],
    target_vec: sparse.csr_matrix,
    backward_analysis: Dict[str, Any],
    forward_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    term_counter = Counter(target_tokens)
    ordered_terms = list(dict.fromkeys(target_tokens))
    term_to_col = {term: int(index) for term, index in vocab.items()}
    col_to_term = {int(index): term for term, index in vocab.items()}
    vector_col_to_weight = {
        int(col): float(weight)
        for col, weight in zip(target_vec.indices.tolist(), target_vec.data.tolist())
    }
    target_term_positions = {
        int(col): pos
        for pos, col in enumerate(target_vec.indices.tolist())
    }

    rows: List[Dict[str, Any]] = []
    for term in ordered_terms:
        col = term_to_col.get(term)
        participates = col in vector_col_to_weight if col is not None else False
        final_weight = float(vector_col_to_weight.get(col, 0.0)) if col is not None else 0.0
        backward_raw = 0.0
        backward_counted = 0.0
        backward_match_count = 0
        backward_counted_match_count = 0
        forward_raw = 0.0
        forward_counted = 0.0
        forward_match_count = 0
        forward_counted_match_count = 0

        if participates and col is not None:
            pos = target_term_positions[col]
            backward_raw = float(backward_analysis["raw_term_sums"][pos])
            backward_counted = float(backward_analysis["counted_term_sums"][pos])
            backward_match_count = int(backward_analysis["raw_term_match_counts"][pos])
            backward_counted_match_count = int(backward_analysis["counted_term_match_counts"][pos])
            forward_raw = float(forward_analysis["raw_term_sums"][pos])
            forward_counted = float(forward_analysis["counted_term_sums"][pos])
            forward_match_count = int(forward_analysis["raw_term_match_counts"][pos])
            forward_counted_match_count = int(forward_analysis["counted_term_match_counts"][pos])

        rows.append(
            {
                "词汇": term,
                "stage1词频": int(term_counter[term]),
                "是否参与最终计算": 1 if participates else 0,
                "最终权重": final_weight,
                "向前命中专利数": backward_match_count,
                "向前计入BS专利数": backward_counted_match_count,
                "向前原始贡献": backward_raw,
                "向前计入BS贡献": backward_counted,
                "向后命中专利数": forward_match_count,
                "向后计入FS专利数": forward_counted_match_count,
                "向后原始贡献": forward_raw,
                "向后计入FS贡献": forward_counted,
                "总原始贡献": backward_raw + forward_raw,
                "总计入BSFS贡献": backward_counted + forward_counted,
                "词汇列索引": "" if col is None else int(col),
                "向量词项名称": "" if col is None else col_to_term.get(int(col), term),
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item["是否参与最终计算"]),
            -float(item["总计入BSFS贡献"]),
            -float(item["总原始贡献"]),
            item["词汇"],
        )
    )
    return rows


def trim_similarity_records(
    records: Sequence[SimilarityRecord],
    *,
    top_n: int,
    bottom_n: int,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    if len(records) <= top_n + bottom_n:
        selected = [(record, "all") for record in records]
    else:
        selected = [(record, "top") for record in records[:top_n]]
        selected.extend((record, "bottom") for record in records[-bottom_n:])

    rows: List[Dict[str, Any]] = []
    for record, segment in selected:
        rows.append(
            {
                "年份": int(record.year),
                "日期": record.date_value or "",
                "申请号": record.application_no,
                "专利名称": record.title,
                "row": int(record.row),
                "相似度": float(record.similarity),
                "是否计入BSFS": 1 if record.counted_in_bsfs else 0,
                "共享词数": int(record.shared_term_count),
                "相对目标年份差": int(record.relative_year_gap),
                "相对目标日期差_天": "" if record.relative_day_gap is None else int(record.relative_day_gap),
                "保存区段": segment,
            }
        )
    return rows


def default_output_dir(stage1_dir: Path, target: PatentRow) -> Path:
    case_name = f"{sanitize_filename(target.application_no)}_year_{target.year}"
    return stage1_dir.parent / "verification" / "patent_similarity_case" / case_name


def write_csv_rows(rows: Sequence[Dict[str, Any]], output_path: Path, header: Sequence[str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def _write_term_contribution_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    header = [
        "词汇",
        "stage1词频",
        "是否参与最终计算",
        "最终权重",
        "向前命中专利数",
        "向前计入BS专利数",
        "向前原始贡献",
        "向前计入BS贡献",
        "向后命中专利数",
        "向后计入FS专利数",
        "向后原始贡献",
        "向后计入FS贡献",
        "总原始贡献",
        "总计入BSFS贡献",
        "词汇列索引",
        "向量词项名称",
    ]
    return write_csv_rows(rows, output_path, header)


def _write_similarity_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    header = [
        "年份",
        "日期",
        "申请号",
        "专利名称",
        "row",
        "相似度",
        "是否计入BSFS",
        "共享词数",
        "相对目标年份差",
        "相对目标日期差_天",
        "保存区段",
    ]
    return write_csv_rows(rows, output_path, header)


def run_similarity_case_analysis(
    *,
    stage1_dir: Path,
    application_no: str,
    year: int,
    date_value: Optional[str] = None,
    title: Optional[str] = None,
    window_size: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
    top_n: int = DEFAULT_TOP_N,
    bottom_n: int = DEFAULT_BOTTOM_N,
) -> Dict[str, Any]:
    mode = detect_stage1_mode(stage1_dir)
    runtime_defaults = infer_runtime_params(stage1_dir)
    resolved_window_size = int(window_size if window_size is not None else runtime_defaults["window_size"])
    resolved_similarity_threshold = float(
        similarity_threshold
        if similarity_threshold is not None
        else runtime_defaults["similarity_threshold"]
    )

    target = locate_target_patent(
        stage1_dir=stage1_dir,
        mode=mode,
        application_no=application_no,
        year=year,
        date_value=date_value,
        title=title,
    )
    resolved_output_dir = output_dir or default_output_dir(stage1_dir, target)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    target_tokens = load_target_tokens(stage1_dir, target)
    vocab = load_vocab(stage1_dir)
    vectors_base = resolve_vectors_base(stage1_dir)
    target_vec = load_target_vector(vectors_base, target)
    window_info = collect_window_candidates(
        stage1_dir=stage1_dir,
        mode=mode,
        target=target,
        window_size=resolved_window_size,
    )

    backward_analysis = compute_similarity_records(
        vectors_base=vectors_base,
        mode=mode,
        target=target,
        target_vec=target_vec,
        candidates=window_info["backward_rows"],
        similarity_threshold=resolved_similarity_threshold,
    )
    forward_analysis = compute_similarity_records(
        vectors_base=vectors_base,
        mode=mode,
        target=target,
        target_vec=target_vec,
        candidates=window_info["forward_rows"],
        similarity_threshold=resolved_similarity_threshold,
    )

    term_rows = build_term_contribution_rows(
        target_tokens=target_tokens,
        vocab=vocab,
        target_vec=target_vec,
        backward_analysis=backward_analysis,
        forward_analysis=forward_analysis,
    )
    backward_csv_rows = trim_similarity_records(
        backward_analysis["records"],
        top_n=top_n,
        bottom_n=bottom_n,
    )
    forward_csv_rows = trim_similarity_records(
        forward_analysis["records"],
        top_n=top_n,
        bottom_n=bottom_n,
    )

    term_csv_path = _write_term_contribution_csv(term_rows, resolved_output_dir / "term_contribution.csv")
    backward_csv_path = _write_similarity_csv(backward_csv_rows, resolved_output_dir / "backward_similarity.csv")
    forward_csv_path = _write_similarity_csv(forward_csv_rows, resolved_output_dir / "forward_similarity.csv")

    summary = {
        "stage1_dir": str(stage1_dir),
        "mode": asdict(mode),
        "target": asdict(target),
        "vectors_base": str(vectors_base),
        "window_size": resolved_window_size,
        "similarity_threshold": resolved_similarity_threshold,
        "target_stage1_tokens": list(target_tokens),
        "target_stage1_token_count": len(target_tokens),
        "target_stage1_unique_token_count": len(set(target_tokens)),
        "target_final_vector_term_count": int(target_vec.nnz),
        "backward_candidate_count": len(window_info["backward_rows"]),
        "forward_candidate_count": len(window_info["forward_rows"]),
        "backward_saved_row_count": len(backward_csv_rows),
        "forward_saved_row_count": len(forward_csv_rows),
        "backward_counted_pair_count": int(
            sum(record.counted_in_bsfs for record in backward_analysis["records"])
        ),
        "forward_counted_pair_count": int(
            sum(record.counted_in_bsfs for record in forward_analysis["records"])
        ),
        "backward_similarity_sum_counted": float(
            sum(record.similarity for record in backward_analysis["records"] if record.counted_in_bsfs)
        ),
        "forward_similarity_sum_counted": float(
            sum(record.similarity for record in forward_analysis["records"] if record.counted_in_bsfs)
        ),
        "top_n": top_n,
        "bottom_n": bottom_n,
        "output_dir": str(resolved_output_dir),
        "term_contribution_csv": str(term_csv_path),
        "backward_similarity_csv": str(backward_csv_path),
        "forward_similarity_csv": str(forward_csv_path),
    }
    summary.update(
        {
            key: value
            for key, value in window_info.items()
            if key not in {"backward_rows", "forward_rows"}
        }
    )
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary

