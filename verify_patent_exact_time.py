from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from patent_quality.case_analysis import analyze_terms, summarize_initial_filters
from patent_quality.config import Config
from patent_quality.io_utils import clear_artifacts
from patent_quality.nlp import init_jieba, load_stopwords, tokenize
from patent_quality.pruning import prune_vectors_by_year
from patent_quality.project_paths import build_experiment_layout, resolve_project_path
from patent_quality.vectorizer import prepare_tokens, vectorize_by_year
from patent_quality.vocab import build_vocab


EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
DEFAULT_OUTPUT_BASE = Path("outputs/tests/verify_patent_exact_time")


@dataclass
class PatentRow:
    row: int
    application_no: str
    public_year: int
    public_date: str
    public_date_ord: int
    title: str


@dataclass
class SimilarityRecord:
    public_year: int
    public_date: str
    application_no: str
    title: str
    similarity: float
    counted_in_quantity_q: bool
    day_gap: int
    abstract: str = ""


@dataclass
class ActualQualityMetrics:
    bs: float
    fs: float
    quantity_q: float
    rank_in_year: int
    year_total: int
    better_count: int
    equal_count: int
    worse_count: int
    rank_ratio_in_year: float
    rank_percent_in_year: float
    source: str


class VerifyPatentExactTimeError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按单个专利复算 exact-time quantity_q，并导出前后窗口内全部专利相似度明细。"
    )
    parser.add_argument("application_no", help="目标专利申请号")
    parser.add_argument("public_year", type=int, help="目标专利公开公告年份")
    parser.add_argument("--k", type=int, default=1, help="exact-time 前后时间窗（年），默认 1")
    parser.add_argument(
        "--experiment-dir",
        help="实验目录或 stage1_exact 目录；提供后优先复用其中 stage1_exact 的 index/tokens/vectors_filtered",
    )
    parser.add_argument("--experiment-id", help="实验 ID；不传 experiment-dir 时可用它定位 stage1_exact")
    parser.add_argument("--output-root", default="outputs/experiments", help="实验输出根目录")
    parser.add_argument("--output-dir", help="验证结果输出目录；不传则按规则自动生成")
    parser.add_argument(
        "--config-script",
        default="run_full.py",
        help="无 experiment-dir 时，用于解析 Config(...) 字面量参数的脚本路径",
    )
    parser.add_argument(
        "--raw-data-path",
        default="data/raw/中国专利分年份保存数据1985-2025",
        help="原始数据路径；仅用于构建 Config，exact 模式实际会读 shared_authorized_parts_dir",
    )
    parser.add_argument(
        "--shared-authorized-parts-dir",
        default="outputs/shared/raw_patent_authorized_parts",
        help="exact 模式共享授权专利 parquet 目录",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="相似度阈值；不传时优先从已有 stage1_exact 的 pair_contrib 推断，否则从 config-script 或 Config 默认值读取",
    )
    parser.add_argument(
        "--include-abstract",
        action="store_true",
        help="在前后向 CSV 中额外输出摘要文本",
    )
    return parser.parse_args()


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)


def parse_config_literals(config_script: Path) -> Dict[str, Any]:
    tree = ast.parse(config_script.read_text(encoding="utf-8"), filename=str(config_script))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "Config":
            continue
        parsed: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                parsed[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                continue
        return parsed
    return {}


def build_runtime_config(
    *,
    args: argparse.Namespace,
    artifacts_dir: Path,
    similarity_threshold: Optional[float],
) -> Config:
    config_script = resolve_project_path(args.config_script)
    raw_data_path = resolve_project_path(args.raw_data_path)
    shared_parts_dir = resolve_project_path(args.shared_authorized_parts_dir)

    kwargs: Dict[str, Any] = {"data_path": str(raw_data_path)}
    if config_script.exists():
        kwargs.update(parse_config_literals(config_script))

    kwargs["data_path"] = str(raw_data_path)
    kwargs["artifacts_dir"] = str(artifacts_dir)
    kwargs["log_file"] = str(artifacts_dir / "logs" / "verify_patent_exact_time.log")
    kwargs["exact_date"] = True
    kwargs["window_size"] = int(args.k)
    kwargs["skip_if_exists"] = False
    kwargs["shared_authorized_parts_dir"] = str(shared_parts_dir)
    if similarity_threshold is not None:
        kwargs["similarity_threshold"] = float(similarity_threshold)

    cfg = Config(**kwargs)
    cfg.data_path = str(resolve_project_path(cfg.data_path))
    cfg.stopword_paths = [str(resolve_project_path(path)) for path in cfg.stopword_paths]
    if cfg.user_dict_path:
        cfg.user_dict_path = str(resolve_project_path(cfg.user_dict_path))
    if cfg.manual_stopwords_path:
        cfg.manual_stopwords_path = str(resolve_project_path(cfg.manual_stopwords_path))
    if cfg.shared_authorized_parts_dir:
        cfg.shared_authorized_parts_dir = str(resolve_project_path(cfg.shared_authorized_parts_dir))
    cfg.ensure_dirs()
    (artifacts_dir / "logs").mkdir(parents=True, exist_ok=True)
    return cfg


def resolve_stage1_exact_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.experiment_dir:
        root = resolve_project_path(args.experiment_dir)
        if root.name == "stage1_exact":
            return root
        candidate = root / "stage1_exact"
        if candidate.exists():
            return candidate
        raise VerifyPatentExactTimeError(f"给定目录下找不到 stage1_exact: {root}")

    if args.experiment_id:
        layout = build_experiment_layout(args.experiment_id, output_root=args.output_root)
        return layout.stage1_exact_dir

    return None


def resolve_output_dir(args: argparse.Namespace, stage1_dir: Optional[Path]) -> Path:
    if args.output_dir:
        return resolve_project_path(args.output_dir)

    case_name = f"{sanitize_filename(args.application_no)}_pubyear_{args.public_year}_k{args.k}"
    if stage1_dir is not None and stage1_dir.name == "stage1_exact":
        return stage1_dir.parent / "verification" / "exact_time" / case_name
    return (resolve_project_path(DEFAULT_OUTPUT_BASE) / case_name).resolve()


def _extract_meta_json(value: Any) -> Dict[str, Any]:
    raw = value
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def infer_similarity_threshold_from_stage1(stage1_dir: Path) -> Optional[float]:
    pair_dir = stage1_dir / "pair_contrib"
    if not pair_dir.exists():
        return None

    candidates = sorted(pair_dir.glob("same_year=*.npz")) + sorted(pair_dir.glob("x=*_y=*.npz"))
    for path in candidates:
        try:
            obj = np.load(path, allow_pickle=True)
            meta = _extract_meta_json(obj["meta_json"])
            if "thr" in meta:
                return float(meta["thr"])
        except Exception:
            continue
    return None


def resolve_similarity_threshold(args: argparse.Namespace, stage1_dir: Optional[Path]) -> float:
    if args.similarity_threshold is not None:
        return float(args.similarity_threshold)

    if stage1_dir is not None:
        inferred = infer_similarity_threshold_from_stage1(stage1_dir)
        if inferred is not None:
            return float(inferred)

    config_script = resolve_project_path(args.config_script)
    kwargs = {"data_path": str(resolve_project_path(args.raw_data_path))}
    if config_script.exists():
        kwargs.update(parse_config_literals(config_script))
    return float(kwargs.get("similarity_threshold", Config(data_path=".").similarity_threshold))


def resolve_vectors_base(stage1_dir: Path) -> Path:
    vectors_filtered = stage1_dir / "vectors_filtered"
    if vectors_filtered.exists() and any(vectors_filtered.glob("year=*.npz")):
        return vectors_filtered
    vectors = stage1_dir / "vectors"
    if vectors.exists() and any(vectors.glob("year=*.npz")):
        return vectors
    raise VerifyPatentExactTimeError(f"在 {stage1_dir} 下找不到可用向量目录（vectors_filtered / vectors）")


def _date_from_ord(day_ord: int) -> date:
    return date.fromordinal(EPOCH_ORDINAL + int(day_ord))


def _day_ord_from_date(value: date) -> int:
    return value.toordinal() - EPOCH_ORDINAL


def _add_years(base: date, years: int) -> date:
    try:
        return base.replace(year=base.year + years)
    except ValueError:
        return base.replace(month=2, day=28, year=base.year + years)


def load_index_year(stage1_dir: Path, public_year: int) -> List[PatentRow]:
    index_path = stage1_dir / "index" / f"year={public_year}.csv"
    if not index_path.exists():
        return []

    rows: List[PatentRow] = []
    with index_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    PatentRow(
                        row=int(row["row"]),
                        application_no=row["申请号"].strip(),
                        public_year=int(row["公开公告年份"]),
                        public_date=row["公开公告日"].strip(),
                        public_date_ord=int(row["公开公告日_ord"]),
                        title=row["专利名称"].strip(),
                    )
                )
            except KeyError as exc:
                raise VerifyPatentExactTimeError(f"stage1_exact index 缺少列: {exc}") from exc
    return rows


def load_target_tokens(stage1_dir: Path, target: PatentRow) -> List[str]:
    token_path = stage1_dir / "tokens" / f"year={target.public_year}.jsonl"
    if not token_path.exists():
        return []

    matches: List[List[str]] = []
    with token_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("id", "")).strip() != target.application_no:
                continue
            matches.append(list(obj.get("tokens", [])))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise VerifyPatentExactTimeError(
            f"目标专利 {target.application_no} 在 tokens/year={target.public_year}.jsonl 中命中多条记录。"
        )
    return []


def locate_target_patent(stage1_dir: Path, application_no: str, public_year: int) -> PatentRow:
    matches = [row for row in load_index_year(stage1_dir, public_year) if row.application_no == application_no]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise VerifyPatentExactTimeError(
            f"在 stage1_exact/index/year={public_year}.csv 中命中了多条同申请号记录: {application_no}"
        )

    token_path = stage1_dir / "tokens" / f"year={public_year}.jsonl"
    if token_path.exists():
        with token_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(obj.get("id", "")).strip() == application_no:
                    raise VerifyPatentExactTimeError(
                        f"目标专利 {application_no} 在 stage1_exact/tokens/year={public_year}.jsonl 第 {line_no} 行命中，"
                        f"但未进入 stage1_exact/index；这通常表示分词后未形成非空向量。"
                    )

    raise VerifyPatentExactTimeError(
        f"目标专利 {application_no} 未在 stage1_exact/index/year={public_year}.csv 中找到。"
    )


def load_actual_quality_metrics(stage1_dir: Path, target: PatentRow, epsilon: float) -> Optional[ActualQualityMetrics]:
    stats_path = stage1_dir / "stats" / f"bsfs_year={target.public_year}.csv"
    if stats_path.exists():
        target_bs: Optional[float] = None
        target_fs: Optional[float] = None
        target_q: Optional[float] = None
        year_q_values: List[float] = []
        with stats_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_idx = int(row["row"])
                bs = float(row["BS"])
                fs = float(row["FS"])
                q = fs / (bs + epsilon)
                year_q_values.append(q)
                if row_idx == target.row:
                    target_bs = bs
                    target_fs = fs
                    target_q = q

        if target_q is not None:
            total = len(year_q_values)
            better_count = sum(1 for value in year_q_values if value > target_q)
            equal_count = sum(1 for value in year_q_values if value == target_q)
            worse_count = sum(1 for value in year_q_values if value < target_q)
            rank = better_count + 1
            ratio = rank / total if total else 0.0
            return ActualQualityMetrics(
                bs=float(target_bs),
                fs=float(target_fs),
                quantity_q=float(target_q),
                rank_in_year=int(rank),
                year_total=int(total),
                better_count=int(better_count),
                equal_count=int(equal_count),
                worse_count=int(worse_count),
                rank_ratio_in_year=float(ratio),
                rank_percent_in_year=float(ratio * 100.0),
                source=str(stats_path),
            )

    final_path = stage1_dir / "patent_quality_output.csv"
    if not final_path.exists():
        return None

    target_key = (target.application_no, str(target.public_year), target.public_date, target.title)
    year_q_values: List[float] = []
    target_metrics: Optional[ActualQualityMetrics] = None

    with final_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            public_year = str(row.get("公开公告年份", "")).strip()
            if public_year != str(target.public_year):
                continue
            application_no = str(row.get("申请号", "")).strip()
            public_date = str(row.get("公开公告日", "")).strip()
            title = str(row.get("专利名称", "")).strip()
            key = (application_no, public_year, public_date, title)
            quality_q = float(row.get("Quality_q", "0") or 0.0)
            year_q_values.append(quality_q)
            if key == target_key:
                bs = float(row.get("BS", "0") or 0.0)
                fs = float(row.get("FS", "0") or 0.0)
                target_metrics = ActualQualityMetrics(
                    bs=bs,
                    fs=fs,
                    quantity_q=quality_q,
                    rank_in_year=0,
                    year_total=0,
                    better_count=0,
                    equal_count=0,
                    worse_count=0,
                    rank_ratio_in_year=0.0,
                    rank_percent_in_year=0.0,
                    source=str(final_path),
                )

    if target_metrics is None or not year_q_values:
        return None

    total = len(year_q_values)
    better_count = sum(1 for value in year_q_values if value > target_metrics.quantity_q)
    equal_count = sum(1 for value in year_q_values if value == target_metrics.quantity_q)
    worse_count = sum(1 for value in year_q_values if value < target_metrics.quantity_q)
    rank = better_count + 1
    ratio = rank / total if total else 0.0
    target_metrics.rank_in_year = int(rank)
    target_metrics.year_total = int(total)
    target_metrics.better_count = int(better_count)
    target_metrics.equal_count = int(equal_count)
    target_metrics.worse_count = int(worse_count)
    target_metrics.rank_ratio_in_year = float(ratio)
    target_metrics.rank_percent_in_year = float(ratio * 100.0)
    return target_metrics


def _candidate_parquets(shared_parts_dir: Path, public_year: int) -> List[Path]:
    if shared_parts_dir.is_file():
        return [shared_parts_dir]
    parquet_paths = sorted(shared_parts_dir.glob("*.parquet"))
    preferred = [path for path in parquet_paths if str(public_year) in path.stem]
    return preferred + [path for path in parquet_paths if path not in preferred]


def load_raw_exact_patent_record(
    *,
    shared_parts_dir: Path,
    cfg: Config,
    target: PatentRow,
) -> Optional[Dict[str, Any]]:
    columns = list(
        dict.fromkeys(
            [
                cfg.col_id,
                cfg.col_type,
                cfg.public_year_col,
                cfg.public_date_col,
                "专利名称",
                *cfg.col_text_parts,
                *cfg.extra_cols,
            ]
        )
    )

    target_key = (
        target.application_no,
        str(target.public_year),
        target.public_date,
        target.title,
    )
    for path in _candidate_parquets(shared_parts_dir, target.public_year):
        available_columns = set(pq.ParquetFile(path).schema_arrow.names)
        read_columns = [column for column in columns if column in available_columns]
        if cfg.col_id not in read_columns:
            continue
        frame = pd.read_parquet(path, columns=read_columns)
        if frame.empty:
            continue
        subset = frame[frame[cfg.col_id].astype("string").fillna("").str.strip() == target.application_no]
        if cfg.col_type in subset.columns:
            subset = subset[subset[cfg.col_type].astype("string").fillna("").str.strip() == "发明授权"]
        if subset.empty:
            continue
        for values in subset.to_dict(orient="records"):
            application_no = str(values.get(cfg.col_id, "")).strip()
            public_year = str(values.get(cfg.public_year_col, "")).strip()
            public_date = str(values.get(cfg.public_date_col, "")).strip()
            title = str(values.get("专利名称", "")).strip()
            if (application_no, public_year, public_date, title) == target_key:
                return values
    return None


def build_token_analysis(
    *,
    stage1_dir: Path,
    cfg: Config,
    target: PatentRow,
    target_tokens: Sequence[str],
    actual_metrics: Optional[ActualQualityMetrics],
) -> Dict[str, Any]:
    shared_parts_dir = resolve_project_path(cfg.shared_authorized_parts_dir)
    raw_record = load_raw_exact_patent_record(shared_parts_dir=shared_parts_dir, cfg=cfg, target=target)
    warnings: List[str] = []
    text_parts: Dict[str, str] = {}

    init_jieba(cfg.user_dict_path)
    stopwords = set(load_stopwords(cfg.stopword_paths)) if cfg.stopword_paths else set()
    tokenization: Dict[str, Any] = {
        "stage1_tokens": list(target_tokens),
        "stage1_token_count": len(target_tokens),
        "stage1_unique_token_count": len(set(target_tokens)),
    }

    if raw_record is not None:
        for column in cfg.col_text_parts:
            value = raw_record.get(column)
            if value is None or pd.isna(value):
                continue
            text_parts[column] = str(value)
        text_joined = cfg.text_sep.join(value for value in text_parts.values() if value)
        recomputed_tokens = tokenize(text_joined, stopwords)
        tokenization["recomputed_stage1_tokens"] = recomputed_tokens
        tokenization["recomputed_matches_stage1"] = recomputed_tokens == list(target_tokens)
        tokenization.update(
            summarize_initial_filters(
                text_joined,
                stopwords=stopwords,
                include_raw_cut=True,
            )
        )
        if recomputed_tokens != list(target_tokens):
            warnings.append("原始文本重新分词结果与 stage1 tokens 不完全一致，请检查词典、停用词或原始文本来源。")
    else:
        warnings.append("未在 shared_authorized_parts_dir 中找到目标专利原始文本记录，无法生成 raw_cut_check。")

    term_analysis = analyze_terms(
        stage1_dir=stage1_dir,
        cfg=cfg,
        year=target.public_year,
        tokens=list(target_tokens),
    )
    term_details = list(term_analysis.get("term_details", []))
    reason_breakdown = dict(Counter(item.get("reason", "unknown") for item in term_details))

    used_terms = [
        {
            "term": item["term"],
            "tf": item["tf"],
            "raw_weight": item["raw_weight"],
            "final_weight": item["final_weight"],
            "reason": item["reason"],
            "reason_detail": item["reason_detail"],
        }
        for item in sorted(
            (detail for detail in term_details if detail.get("participates_in_final_similarity")),
            key=lambda detail: (-float(detail.get("final_weight", 0.0)), detail.get("term", "")),
        )
    ]
    removed_terms = [
        {
            "term": item["term"],
            "tf": item["tf"],
            "raw_weight": item["raw_weight"],
            "final_weight": item["final_weight"],
            "status": item["status"],
            "reason": item["reason"],
            "reason_detail": item["reason_detail"],
        }
        for item in sorted(
            (detail for detail in term_details if not detail.get("participates_in_final_similarity")),
            key=lambda detail: (-float(detail.get("raw_weight", 0.0)), detail.get("term", "")),
        )
    ]

    return {
        "patent": {
            **asdict(target),
            "actual_experiment_metrics": None if actual_metrics is None else asdict(actual_metrics),
        },
        "config_used": {
            "shared_authorized_parts_dir": str(shared_parts_dir),
            "stopword_paths": cfg.stopword_paths,
            "user_dict_path": cfg.user_dict_path,
            "col_text_parts": cfg.col_text_parts,
            "text_sep": cfg.text_sep,
            "min_term_count": cfg.min_term_count,
            "max_doc_freq_ratio": cfg.max_doc_freq_ratio,
            "manual_stopwords_path": cfg.manual_stopwords_path,
            "df_ratio_threshold": cfg.df_ratio_threshold,
            "top_df_percent": cfg.top_df_percent,
            "topk_terms_per_doc": cfg.topk_terms_per_doc,
        },
        "raw_record": raw_record,
        "text_parts": text_parts,
        "tokenization": tokenization,
        "term_analysis": {
            "summary": term_analysis.get("summary", {}),
            "reason_breakdown": reason_breakdown,
            "used_terms": used_terms,
            "removed_terms": removed_terms,
            "term_details": term_details,
        },
        "warnings": warnings,
    }


def write_json(payload: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def collect_window_candidates(stage1_dir: Path, target: PatentRow, k: int) -> tuple[List[PatentRow], List[PatentRow], int, int]:
    target_day = _date_from_ord(target.public_date_ord)
    backward_start_ord = _day_ord_from_date(_add_years(target_day, -k))
    forward_end_ord = _day_ord_from_date(_add_years(target_day, k))

    backward: List[PatentRow] = []
    forward: List[PatentRow] = []
    for public_year in range(target.public_year - k, target.public_year + k + 1):
        for row in load_index_year(stage1_dir, public_year):
            if row.application_no == target.application_no and row.public_year == target.public_year and row.row == target.row:
                continue
            if backward_start_ord <= row.public_date_ord < target.public_date_ord:
                backward.append(row)
            elif target.public_date_ord < row.public_date_ord <= forward_end_ord:
                forward.append(row)

    return backward, forward, backward_start_ord, forward_end_ord


def group_rows_by_year(rows: Sequence[PatentRow]) -> Dict[int, List[PatentRow]]:
    grouped: Dict[int, List[PatentRow]] = {}
    for row in rows:
        grouped.setdefault(row.public_year, []).append(row)
    return grouped


def compute_similarity_records(
    *,
    vectors_base: Path,
    target: PatentRow,
    candidates: Sequence[PatentRow],
    threshold: float,
    direction: str,
) -> List[SimilarityRecord]:
    if not candidates:
        return []

    target_matrix_path = vectors_base / f"year={target.public_year}.npz"
    if not target_matrix_path.exists():
        raise VerifyPatentExactTimeError(f"缺少目标年份向量文件: {target_matrix_path}")
    target_matrix = sparse.load_npz(target_matrix_path).tocsr()
    if target.row >= target_matrix.shape[0]:
        raise VerifyPatentExactTimeError(
            f"目标专利 row={target.row} 超出年份 {target.public_year} 的向量矩阵范围 {target_matrix.shape[0]}"
        )
    target_vec = target_matrix.getrow(target.row)

    results: List[SimilarityRecord] = []
    for public_year, rows_in_year in group_rows_by_year(candidates).items():
        matrix_path = vectors_base / f"year={public_year}.npz"
        if not matrix_path.exists():
            raise VerifyPatentExactTimeError(f"缺少候选年份向量文件: {matrix_path}")
        year_matrix = sparse.load_npz(matrix_path).tocsr()
        row_ids = [row.row for row in rows_in_year]
        max_row = max(row_ids)
        if max_row >= year_matrix.shape[0]:
            raise VerifyPatentExactTimeError(
                f"年份 {public_year} 候选 row={max_row} 超出向量矩阵范围 {year_matrix.shape[0]}"
            )

        sims = year_matrix[row_ids].dot(target_vec.T)
        scores = np.asarray(sims.toarray()).reshape(-1)
        for row, score in zip(rows_in_year, scores):
            score_value = float(score)
            results.append(
                SimilarityRecord(
                    public_year=row.public_year,
                    public_date=row.public_date,
                    application_no=row.application_no,
                    title=row.title,
                    similarity=score_value,
                    counted_in_quantity_q=score_value >= threshold,
                    day_gap=row.public_date_ord - target.public_date_ord,
                )
            )

    results.sort(key=lambda item: (-item.similarity, item.public_date, item.application_no))
    if direction == "backward":
        return results
    if direction == "forward":
        return results
    raise VerifyPatentExactTimeError(f"未知方向: {direction}")


def attach_abstracts(
    *,
    records: List[SimilarityRecord],
    shared_parts_dir: Path,
) -> None:
    if not records:
        return

    application_nos = {record.application_no for record in records}
    desired_keys = {
        (
            record.application_no,
            str(record.public_year),
            record.public_date,
            record.title,
        )
        for record in records
    }
    abstract_lookup: Dict[tuple[str, str, str, str], str] = {}

    parquet_paths = sorted(shared_parts_dir.glob("*.parquet"))
    if not parquet_paths:
        raise VerifyPatentExactTimeError(f"找不到 shared parquet 文件: {shared_parts_dir}")

    cols = ["申请号", "公开公告年份", "公开公告日", "专利名称", "摘要文本"]
    for path in parquet_paths:
        available_columns = set(pq.ParquetFile(path).schema_arrow.names)
        read_columns = [column for column in cols if column in available_columns]
        required = {"申请号", "公开公告年份", "公开公告日", "专利名称"}
        if not required.issubset(read_columns):
            continue
        frame = pd.read_parquet(path, columns=read_columns)
        if frame.empty:
            continue
        subset = frame[frame["申请号"].astype("string").fillna("").str.strip().isin(application_nos)]
        if subset.empty:
            continue
        for values in subset.itertuples(index=False, name=None):
            application_no = "" if pd.isna(values[0]) else str(values[0]).strip()
            public_year = "" if pd.isna(values[1]) else str(values[1]).strip()
            public_date = "" if pd.isna(values[2]) else str(values[2]).strip()
            title = "" if pd.isna(values[3]) else str(values[3]).strip()
            abstract = ""
            if len(values) >= 5:
                abstract = "" if pd.isna(values[4]) else str(values[4]).strip()
            key = (application_no, public_year, public_date, title)
            if key in desired_keys and key not in abstract_lookup:
                abstract_lookup[key] = abstract

    for record in records:
        key = (record.application_no, str(record.public_year), record.public_date, record.title)
        record.abstract = abstract_lookup.get(key, "")


def write_similarity_csv(
    records: Sequence[SimilarityRecord],
    output_path: Path,
    *,
    include_abstract: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "公开公告年份",
        "公开公告日",
        "申请号",
        "专利标题",
        "相似度分数",
        "是否计入quantity_q",
        "相对目标公开日天数差",
    ]
    if include_abstract:
        header.append("摘要文本")

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for record in records:
            row = [
                record.public_year,
                record.public_date,
                record.application_no,
                record.title,
                f"{record.similarity:.10f}",
                1 if record.counted_in_quantity_q else 0,
                record.day_gap,
            ]
            if include_abstract:
                row.append(record.abstract)
            writer.writerow(row)


def rebuild_exact_stage1(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    similarity_threshold: float,
) -> tuple[Path, Config]:
    rebuild_dir = output_dir / "rebuilt_stage1_exact"
    cfg = build_runtime_config(args=args, artifacts_dir=rebuild_dir, similarity_threshold=similarity_threshold)
    clear_artifacts(cfg)

    print(f"[rebuild] stage1_exact 重建目录: {rebuild_dir}")
    print("[rebuild] 阶段1: build_vocab")
    build_vocab(cfg)
    print("[rebuild] 阶段2: prepare_tokens")
    prepare_tokens(cfg)
    print("[rebuild] 阶段3: vectorize_by_year")
    vectorize_by_year(cfg)
    print("[rebuild] 阶段4: prune_vectors_by_year")
    prune_vectors_by_year(cfg)
    return rebuild_dir, cfg


def build_summary(
    *,
    target: PatentRow,
    target_tokens: Sequence[str],
    k: int,
    threshold: float,
    backward_records: Sequence[SimilarityRecord],
    forward_records: Sequence[SimilarityRecord],
    stage1_dir: Path,
    vectors_base: Path,
    output_dir: Path,
    backward_start_ord: int,
    forward_end_ord: int,
    reused_stage1_exact: bool,
    actual_metrics: Optional[ActualQualityMetrics],
    token_analysis: Optional[Dict[str, Any]],
    token_analysis_path: Optional[Path],
) -> Dict[str, Any]:
    bs_sum = float(sum(record.similarity for record in backward_records if record.counted_in_quantity_q))
    fs_sum = float(sum(record.similarity for record in forward_records if record.counted_in_quantity_q))
    epsilon = Config(data_path=".").epsilon
    quantity_q = fs_sum / (bs_sum + epsilon)

    return {
        "target": asdict(target),
        "window_years": int(k),
        "similarity_threshold": float(threshold),
        "window_start_date": _date_from_ord(backward_start_ord).isoformat(),
        "window_end_date": _date_from_ord(forward_end_ord).isoformat(),
        "target_tokens": list(target_tokens),
        "target_token_count": len(target_tokens),
        "bs_sum": bs_sum,
        "fs_sum": fs_sum,
        "quantity_q": quantity_q,
        "backward_total_rows": len(backward_records),
        "forward_total_rows": len(forward_records),
        "backward_counted_rows": int(sum(record.counted_in_quantity_q for record in backward_records)),
        "forward_counted_rows": int(sum(record.counted_in_quantity_q for record in forward_records)),
        "reused_stage1_exact": bool(reused_stage1_exact),
        "stage1_exact_dir": str(stage1_dir),
        "vectors_base": str(vectors_base),
        "output_dir": str(output_dir),
        "backward_csv": str(output_dir / "backward_similarity.csv"),
        "forward_csv": str(output_dir / "forward_similarity.csv"),
        "actual_experiment_metrics": None if actual_metrics is None else asdict(actual_metrics),
        "quantity_q_diff_vs_actual": None if actual_metrics is None else quantity_q - float(actual_metrics.quantity_q),
        "token_analysis_json": None if token_analysis_path is None else str(token_analysis_path),
        "token_analysis_summary": None if token_analysis is None else token_analysis.get("term_analysis", {}).get("summary", {}),
        "token_analysis_reason_breakdown": None if token_analysis is None else token_analysis.get("term_analysis", {}).get("reason_breakdown", {}),
    }


def write_summary(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "summary.json"
    return write_json(summary, output_path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    requested_stage1_dir = resolve_stage1_exact_dir(args)
    output_dir = resolve_output_dir(args, requested_stage1_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    similarity_threshold = resolve_similarity_threshold(args, requested_stage1_dir)
    reused_stage1_exact = requested_stage1_dir is not None
    if reused_stage1_exact:
        stage1_dir = requested_stage1_dir
        cfg = build_runtime_config(args=args, artifacts_dir=stage1_dir, similarity_threshold=similarity_threshold)
    else:
        stage1_dir, cfg = rebuild_exact_stage1(
            args=args,
            output_dir=output_dir,
            similarity_threshold=similarity_threshold,
        )

    vectors_base = resolve_vectors_base(stage1_dir)
    target = locate_target_patent(stage1_dir, args.application_no, args.public_year)
    target_tokens = load_target_tokens(stage1_dir, target)
    backward_rows, forward_rows, backward_start_ord, forward_end_ord = collect_window_candidates(stage1_dir, target, args.k)
    actual_metrics = load_actual_quality_metrics(stage1_dir, target, cfg.epsilon) if reused_stage1_exact else None

    print(
        f"[verify] target={target.application_no} public_year={target.public_year} public_date={target.public_date} "
        f"title={target.title}"
    )
    print(f"[verify] target_token_count={len(target_tokens)}")
    print(
        f"[verify] backward_candidates={len(backward_rows)} forward_candidates={len(forward_rows)} "
        f"threshold={similarity_threshold:.6f} vectors_base={vectors_base}"
    )
    if actual_metrics is not None:
        print(
            f"[verify] actual_quantity_q={actual_metrics.quantity_q:.10f} "
            f"rank={actual_metrics.rank_in_year}/{actual_metrics.year_total} "
            f"top_percent={actual_metrics.rank_percent_in_year:.4f}% source={actual_metrics.source}"
        )

    backward_records = compute_similarity_records(
        vectors_base=vectors_base,
        target=target,
        candidates=backward_rows,
        threshold=similarity_threshold,
        direction="backward",
    )
    forward_records = compute_similarity_records(
        vectors_base=vectors_base,
        target=target,
        candidates=forward_rows,
        threshold=similarity_threshold,
        direction="forward",
    )

    if args.include_abstract:
        shared_parts_dir = resolve_project_path(cfg.shared_authorized_parts_dir)
        print(f"[verify] attach abstracts from {shared_parts_dir}")
        attach_abstracts(records=backward_records + forward_records, shared_parts_dir=shared_parts_dir)

    backward_csv = output_dir / "backward_similarity.csv"
    forward_csv = output_dir / "forward_similarity.csv"
    write_similarity_csv(backward_records, backward_csv, include_abstract=args.include_abstract)
    write_similarity_csv(forward_records, forward_csv, include_abstract=args.include_abstract)
    print(f"[verify] backward_csv_rows={len(backward_records)}")
    print(f"[verify] forward_csv_rows={len(forward_records)}")

    token_analysis = build_token_analysis(
        stage1_dir=stage1_dir,
        cfg=cfg,
        target=target,
        target_tokens=target_tokens,
        actual_metrics=actual_metrics,
    )
    token_analysis_path = write_json(token_analysis, output_dir / "token_analysis.json")
    print(
        f"[verify] token_analysis_used_terms={len(token_analysis['term_analysis']['used_terms'])} "
        f"removed_terms={len(token_analysis['term_analysis']['removed_terms'])}"
    )

    summary = build_summary(
        target=target,
        target_tokens=target_tokens,
        k=args.k,
        threshold=similarity_threshold,
        backward_records=backward_records,
        forward_records=forward_records,
        stage1_dir=stage1_dir,
        vectors_base=vectors_base,
        output_dir=output_dir,
        backward_start_ord=backward_start_ord,
        forward_end_ord=forward_end_ord,
        reused_stage1_exact=reused_stage1_exact,
        actual_metrics=actual_metrics,
        token_analysis=token_analysis,
        token_analysis_path=token_analysis_path,
    )
    summary_path = write_summary(summary, output_dir)

    print(f"[done] backward_csv={backward_csv}")
    print(f"[done] forward_csv={forward_csv}")
    print(
        f"[done] BS={summary['bs_sum']:.10f} FS={summary['fs_sum']:.10f} "
        f"quantity_q={summary['quantity_q']:.10f}"
    )
    if actual_metrics is not None:
        print(
            f"[done] actual_quantity_q={actual_metrics.quantity_q:.10f} "
            f"diff={summary['quantity_q_diff_vs_actual']:.10f}"
        )
    print(f"[done] token_analysis_json={token_analysis_path}")
    print(f"[done] summary_json={summary_path}")
    return summary


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
