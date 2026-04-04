from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import jieba
import numpy as np
import pandas as pd

from .config import Config
from .nlp import _re_cn, init_jieba, load_stopwords, tokenize
from .project_paths import resolve_project_path


TEXT_DATE_COLUMNS = ["申请日", "公开（公告）日", "公开公告日"]
AUTHORIZED_PATENT_TYPE = "发明授权"


class PatentCaseError(RuntimeError):
    pass


@dataclass
class PatentMatch:
    row_index: int
    application_no: str
    application_year: int
    title: str
    extras: Dict[str, str]


@dataclass
class TokenMatch:
    line_index: int
    application_no: str
    application_year: int
    title: str
    token_count: int
    token_unique_count: int


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


@lru_cache(maxsize=None)
def _parse_config_literals_cached(config_script_path: str) -> Dict[str, Any]:
    return parse_config_literals(Path(config_script_path))


@lru_cache(maxsize=None)
def _load_stopwords_cached(stopword_paths: tuple[str, ...]) -> frozenset[str]:
    if not stopword_paths:
        return frozenset()
    return frozenset(load_stopwords(list(stopword_paths)))


@lru_cache(maxsize=None)
def _ensure_jieba_initialized_cached(user_dict_path: Optional[str]) -> bool:
    init_jieba(user_dict_path)
    return True


@lru_cache(maxsize=None)
def _load_global_df_cached(stage1_dir: str) -> Dict[str, Any]:
    with (Path(stage1_dir) / "df" / "global_df.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def _load_year_df_cached(stage1_dir: str, year: int) -> Dict[str, Any]:
    with (Path(stage1_dir) / "df" / f"term_df_year={year}.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def _load_vocab_cached(stage1_dir: str) -> Dict[str, int]:
    with (Path(stage1_dir) / "vocab" / "final_vocab.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)["vocab"]


def build_analysis_config(
    *,
    raw_data_path: Path,
    config_script: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    kwargs: Dict[str, Any] = {"data_path": str(raw_data_path)}
    if config_script and config_script.exists():
        kwargs.update(_parse_config_literals_cached(str(config_script.resolve())))
    kwargs["data_path"] = str(raw_data_path)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
    cfg = Config(**kwargs)
    cfg.data_path = str(resolve_project_path(cfg.data_path))
    cfg.stopword_paths = [str(resolve_project_path(path)) for path in cfg.stopword_paths]
    if cfg.user_dict_path:
        cfg.user_dict_path = str(resolve_project_path(cfg.user_dict_path))
    if cfg.manual_stopwords_path:
        cfg.manual_stopwords_path = str(resolve_project_path(cfg.manual_stopwords_path))
    return cfg


def iter_stage1_candidates(
    *,
    stage1_dir: Path,
    application_no: Optional[str],
    application_year: Optional[int],
    title: Optional[str],
    title_contains: Optional[str],
    restrict_to_year: bool = True,
) -> List[PatentMatch]:
    index_dir = stage1_dir / "index"
    if not index_dir.exists():
        raise FileNotFoundError(f"找不到 stage1 index 目录: {index_dir}")

    target_files: List[Path]
    if application_year is not None and restrict_to_year:
        target_files = [index_dir / f"year={application_year}.csv"]
    else:
        target_files = sorted(index_dir.glob("year=*.csv"))

    matches: List[PatentMatch] = []
    title_contains_norm = title_contains.strip() if title_contains else None
    title_norm = title.strip() if title else None
    app_no_norm = application_no.strip() if application_no else None

    for index_path in target_files:
        if not index_path.exists():
            continue
        with index_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if app_no_norm and row.get("申请号", "").strip() != app_no_norm:
                    continue
                row_title = row.get("专利名称", "").strip()
                if title_norm and row_title != title_norm:
                    continue
                if title_contains_norm and title_contains_norm not in row_title:
                    continue
                extras = {
                    key: value
                    for key, value in row.items()
                    if key not in {"row", "申请号", "申请年份", "专利名称"} and value is not None
                }
                matches.append(
                    PatentMatch(
                        row_index=int(row["row"]),
                        application_no=row["申请号"].strip(),
                        application_year=int(row["申请年份"]),
                        title=row_title,
                        extras=extras,
                    )
                )
    return matches


def search_stage1_index(
    *,
    stage1_dir: Path,
    application_no: Optional[str],
    application_year: Optional[int],
    title: Optional[str],
    title_contains: Optional[str],
    restrict_to_year: bool = True,
) -> Dict[str, Any]:
    index_dir = stage1_dir / "index"
    if not index_dir.exists():
        raise FileNotFoundError(f"找不到 stage1 index 目录: {index_dir}")

    if application_year is not None and restrict_to_year:
        target_files = [index_dir / f"year={application_year}.csv"]
    else:
        target_files = sorted(index_dir.glob("year=*.csv"))

    app_no_norm = application_no.strip() if application_no else None
    title_norm = title.strip() if title else None
    title_contains_norm = title_contains.strip() if title_contains else None

    matches: List[PatentMatch] = []
    scanned_files: List[str] = []
    rows_scanned = 0

    for index_path in target_files:
        scanned_files.append(str(index_path))
        if not index_path.exists():
            continue
        with index_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows_scanned += 1
                if app_no_norm and row.get("申请号", "").strip() != app_no_norm:
                    continue
                row_title = row.get("专利名称", "").strip()
                if title_norm and row_title != title_norm:
                    continue
                if title_contains_norm and title_contains_norm not in row_title:
                    continue
                extras = {
                    key: value
                    for key, value in row.items()
                    if key not in {"row", "申请号", "申请年份", "专利名称"} and value is not None
                }
                matches.append(
                    PatentMatch(
                        row_index=int(row["row"]),
                        application_no=row["申请号"].strip(),
                        application_year=int(row["申请年份"]),
                        title=row_title,
                        extras=extras,
                    )
                )

    return {
        "stage": "stage1_index",
        "restrict_to_year": restrict_to_year,
        "input_year": application_year,
        "scanned_files": scanned_files,
        "scanned_file_count": len(scanned_files),
        "rows_scanned": rows_scanned,
        "match_count": len(matches),
        "matches": matches,
    }


def search_stage1_tokens(
    *,
    stage1_dir: Path,
    application_no: Optional[str],
    application_year: Optional[int],
    title: Optional[str],
    title_contains: Optional[str],
    restrict_to_year: bool = True,
) -> Dict[str, Any]:
    token_dir = stage1_dir / "tokens"
    if not token_dir.exists():
        raise FileNotFoundError(f"找不到 stage1 tokens 目录: {token_dir}")

    if application_year is not None and restrict_to_year:
        target_files = [token_dir / f"year={application_year}.jsonl"]
    else:
        target_files = sorted(token_dir.glob("year=*.jsonl"))

    app_no_norm = application_no.strip() if application_no else None
    title_norm = title.strip() if title else None
    title_contains_norm = title_contains.strip() if title_contains else None

    matches: List[TokenMatch] = []
    scanned_files: List[str] = []
    rows_scanned = 0

    for token_path in target_files:
        scanned_files.append(str(token_path))
        if not token_path.exists():
            continue
        year_match = re.search(r"year=(\d+)", token_path.name)
        year = int(year_match.group(1)) if year_match else application_year or -1
        with token_path.open("r", encoding="utf-8") as fh:
            for line_index, line in enumerate(fh):
                rows_scanned += 1
                obj = json.loads(line)
                if app_no_norm and str(obj.get("id", "")).strip() != app_no_norm:
                    continue
                row_title = str(obj.get("title", "")).strip()
                if title_norm and row_title != title_norm:
                    continue
                if title_contains_norm and title_contains_norm not in row_title:
                    continue
                tokens = list(obj.get("tokens", []))
                matches.append(
                    TokenMatch(
                        line_index=line_index,
                        application_no=str(obj.get("id", "")).strip(),
                        application_year=year,
                        title=row_title,
                        token_count=len(tokens),
                        token_unique_count=len(set(tokens)),
                    )
                )

    return {
        "stage": "stage1_tokens",
        "restrict_to_year": restrict_to_year,
        "input_year": application_year,
        "scanned_files": scanned_files,
        "scanned_file_count": len(scanned_files),
        "rows_scanned": rows_scanned,
        "match_count": len(matches),
        "matches": matches,
    }


def load_stage1_token_record(stage1_dir: Path, match: PatentMatch) -> Dict[str, Any]:
    token_path = stage1_dir / "tokens" / f"year={match.application_year}.jsonl"
    if not token_path.exists():
        raise FileNotFoundError(f"找不到 token 文件: {token_path}")
    with token_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh):
            if line_number != match.row_index:
                continue
            obj = json.loads(line)
            if obj.get("id") != match.application_no:
                break
            return obj

    with token_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("id") == match.application_no:
                return obj
    raise PatentCaseError(f"在 token 文件中找不到专利 {match.application_no}")


def load_bsfs_row(stage1_dir: Path, match: PatentMatch, epsilon: float) -> Dict[str, float]:
    stats_path = stage1_dir / "stats" / f"bsfs_year={match.application_year}.csv"
    if not stats_path.exists():
        return {"BS": 0.0, "FS": 0.0, "Quality_q": 0.0}
    with stats_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["row"]) != match.row_index:
                continue
            bs = float(row["BS"])
            fs = float(row["FS"])
            return {"BS": bs, "FS": fs, "Quality_q": fs / (bs + epsilon)}
    return {"BS": 0.0, "FS": 0.0, "Quality_q": 0.0}


def load_quality_rank(stage1_dir: Path, match: PatentMatch, epsilon: float) -> Dict[str, Any]:
    stats_path = stage1_dir / "stats" / f"bsfs_year={match.application_year}.csv"
    if not stats_path.exists():
        return {
            "available": False,
            "year": match.application_year,
            "total_patents_in_year": None,
        }

    quality_values: List[float] = []
    target_quality: Optional[float] = None
    with stats_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bs = float(row["BS"])
            fs = float(row["FS"])
            quality = fs / (bs + epsilon)
            quality_values.append(quality)
            if int(row["row"]) == match.row_index:
                target_quality = quality

    if target_quality is None:
        return {
            "available": False,
            "year": match.application_year,
            "total_patents_in_year": len(quality_values),
        }

    total = len(quality_values)
    better_count = sum(1 for value in quality_values if value > target_quality)
    equal_count = sum(1 for value in quality_values if value == target_quality)
    worse_count = sum(1 for value in quality_values if value < target_quality)
    rank_desc = better_count + 1
    top_percent = (rank_desc / total) if total else None
    outperform_percent = (worse_count / total) if total else None
    return {
        "available": True,
        "year": match.application_year,
        "quality_q": target_quality,
        "rank_desc": rank_desc,
        "total_patents_in_year": total,
        "top_percent": top_percent,
        "top_percent_display": None if top_percent is None else round(top_percent * 100, 4),
        "better_count": better_count,
        "equal_count": equal_count,
        "worse_count": worse_count,
        "outperform_percent": outperform_percent,
        "outperform_percent_display": None if outperform_percent is None else round(outperform_percent * 100, 4),
    }


def load_global_df(stage1_dir: Path) -> Dict[str, Any]:
    return _load_global_df_cached(str(stage1_dir.resolve()))


def load_year_df(stage1_dir: Path, year: int) -> Dict[str, Any]:
    return _load_year_df_cached(str(stage1_dir.resolve()), year)


def load_vocab(stage1_dir: Path) -> Dict[str, int]:
    return _load_vocab_cached(str(stage1_dir.resolve()))


def compute_cumulative_history(
    *,
    stage1_dir: Path,
    year: int,
    terms: Iterable[str],
) -> Dict[str, Any]:
    terms_list = list(dict.fromkeys(terms))
    history = {term: 0 for term in terms_list}
    docs_before = 0
    for path in sorted((stage1_dir / "df").glob("term_df_year=*.json")):
        match = re.search(r"year=(\d+)", path.name)
        if not match:
            continue
        current_year = int(match.group(1))
        if current_year >= year:
            break
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        docs_before += int(obj.get("docs", 0))
        df_map = obj.get("df", {})
        for term in terms_list:
            history[term] += int(df_map.get(term, 0))
    return {"docs_before": docs_before, "cumulative_df": history}


def _candidate_csvs(raw_data_path: Path, year: Optional[int]) -> List[Path]:
    if raw_data_path.is_file():
        return [raw_data_path]
    csv_paths = sorted(raw_data_path.glob("*.csv"))
    if year is None:
        return csv_paths
    preferred = [path for path in csv_paths if str(year) in path.stem]
    return preferred + [path for path in csv_paths if path not in preferred]


def _normalize_date_value(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return digits
    if len(digits) >= 4:
        return digits[:4]
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        return text
    if pd.isna(parsed):
        return text
    return parsed.strftime("%Y%m%d")


def _publication_matches(row: pd.Series, publication_date: Optional[str]) -> bool:
    if not publication_date:
        return True
    query = _normalize_date_value(publication_date)
    if not query:
        return True
    for column in TEXT_DATE_COLUMNS:
        if column not in row.index:
            continue
        value = _normalize_date_value(row.get(column))
        if not value:
            continue
        if len(query) == 4 and value.startswith(query):
            return True
        if value == query:
            return True
    return False


def load_raw_patent_record(
    *,
    raw_data_path: Path,
    cfg: Config,
    stage1_match: PatentMatch,
    publication_date: Optional[str],
) -> Optional[Dict[str, Any]]:
    needed_columns = [
        cfg.col_id,
        cfg.col_type,
        cfg.col_date,
        "专利名称",
        *cfg.col_text_parts,
        *TEXT_DATE_COLUMNS,
    ]
    usecols = list(dict.fromkeys([column for column in needed_columns if column]))

    for csv_path in _candidate_csvs(raw_data_path, stage1_match.application_year):
        for encoding in ("utf-8", "gb18030"):
            try:
                reader = pd.read_csv(
                    csv_path,
                    chunksize=50000,
                    usecols=lambda name: name in set(usecols),
                    encoding=encoding,
                    low_memory=False,
                    engine="c",
                )
            except Exception:
                continue

            found_candidate = False
            for chunk in reader:
                chunk = chunk[chunk[cfg.col_id].astype("string").str.strip() == stage1_match.application_no]
                if cfg.col_type in chunk.columns:
                    chunk = chunk[chunk[cfg.col_type].astype("string").str.strip() == AUTHORIZED_PATENT_TYPE]
                if chunk.empty:
                    continue
                found_candidate = True
                for _, row in chunk.iterrows():
                    if _publication_matches(row, publication_date):
                        return row.to_dict()
            if found_candidate:
                break
            break
    return None


def summarize_initial_filters(
    text: str,
    *,
    stopwords: set[str],
    include_raw_cut: bool,
) -> Dict[str, Any]:
    if not include_raw_cut:
        return {
            "raw_cut_token_count": None,
            "raw_cut_unique_count": None,
            "removed_before_stage1": {},
        }

    removed: Dict[str, Counter[str]] = {
        "empty": Counter(),
        "stopword": Counter(),
        "non_chinese": Counter(),
    }
    raw_tokens: List[str] = []
    for token in jieba.cut(text, HMM=True):
        raw_tokens.append(token)
        if not token:
            removed["empty"]["<EMPTY>"] += 1
            continue
        if token in stopwords:
            removed["stopword"][token] += 1
            continue
        if not _re_cn.fullmatch(token):
            removed["non_chinese"][token] += 1

    removed_summary = {
        name: [{"term": term, "count": count} for term, count in counter.most_common(20)]
        for name, counter in removed.items()
        if counter
    }
    return {
        "raw_cut_token_count": len(raw_tokens),
        "raw_cut_unique_count": len(set(raw_tokens)),
        "removed_before_stage1": removed_summary,
    }


def analyze_terms(
    *,
    stage1_dir: Path,
    cfg: Config,
    year: int,
    tokens: List[str],
) -> Dict[str, Any]:
    token_counter = Counter(tokens)
    token_order = list(dict.fromkeys(tokens))
    global_df_obj = load_global_df(stage1_dir)
    global_df = global_df_obj["df"]
    total_docs = int(global_df_obj["total_docs"])
    year_df_obj = load_year_df(stage1_dir, year)
    year_df = year_df_obj["df"]
    docs_in_year = int(year_df_obj["docs"])
    vocab = load_vocab(stage1_dir)

    history = compute_cumulative_history(stage1_dir=stage1_dir, year=year, terms=token_order)
    docs_before = int(history["docs_before"])
    cumulative_df = history["cumulative_df"]

    manual_stopwords = set()
    if cfg.manual_stopwords_path:
        stopword_path = Path(cfg.manual_stopwords_path)
        if stopword_path.exists():
            manual_stopwords = set(_load_stopwords_cached((cfg.manual_stopwords_path,)))

    removed_ratio_terms = {
        term
        for term, df_value in year_df.items()
        if docs_in_year > 0 and (float(df_value) / docs_in_year) >= float(cfg.df_ratio_threshold)
    }

    vocab_terms_for_top_df = [
        (term, int(df_value))
        for term, df_value in year_df.items()
        if int(df_value) > 0 and term in vocab and term not in removed_ratio_terms
    ]
    vocab_terms_for_top_df.sort(key=lambda item: item[1], reverse=True)
    vocab_size_in_year = len(vocab_terms_for_top_df)
    top_df_k = int(max(0, math.floor(vocab_size_in_year * float(cfg.top_df_percent))))
    top_df_removed_terms = {term for term, _ in vocab_terms_for_top_df[:top_df_k]}
    top_df_rank = {term: rank for rank, (term, _) in enumerate(vocab_terms_for_top_df, start=1)}

    raw_weights: Dict[str, float] = {}
    for term in token_order:
        if term not in vocab:
            continue
        if docs_before <= 0:
            raw_weights[term] = 0.0
            continue
        idf = float(np.log(docs_before / (1.0 + cumulative_df.get(term, 0))))
        raw_weights[term] = token_counter[term] * idf

    raw_norm = math.sqrt(sum(value * value for value in raw_weights.values()))
    raw_normalized = {
        term: (value / raw_norm if raw_norm > 0 else 0.0)
        for term, value in raw_weights.items()
    }

    filtered_weights = dict(raw_normalized)
    prune_reason: Dict[str, str] = {}
    for term in token_order:
        if term not in filtered_weights:
            continue
        if term in manual_stopwords:
            filtered_weights[term] = 0.0
            prune_reason[term] = "manual_stopword_pruning"
        elif term in removed_ratio_terms:
            filtered_weights[term] = 0.0
            prune_reason[term] = "year_df_ratio_pruning"
        elif term in top_df_removed_terms:
            filtered_weights[term] = 0.0
            prune_reason[term] = "year_top_df_pruning"

    kept_after_column_pruning = [term for term in token_order if filtered_weights.get(term, 0.0) > 0]
    doc_weight_rank_before_topk = {
        term: rank
        for rank, term in enumerate(
            sorted(kept_after_column_pruning, key=lambda term: filtered_weights.get(term, 0.0), reverse=True),
            start=1,
        )
    }
    if len(kept_after_column_pruning) > int(cfg.topk_terms_per_doc):
        values = np.array([filtered_weights[term] for term in kept_after_column_pruning], dtype="float64")
        keep_idx = np.argpartition(values, len(values) - int(cfg.topk_terms_per_doc))[len(values) - int(cfg.topk_terms_per_doc) :]
        keep_positions = set(int(pos) for pos in keep_idx.tolist())
        for position, term in enumerate(kept_after_column_pruning):
            if position in keep_positions:
                continue
            filtered_weights[term] = 0.0
            prune_reason[term] = "document_topk_pruning"

    term_details: List[Dict[str, Any]] = []
    for term in token_order:
        global_df_value = int(global_df.get(term, 0))
        year_df_value = int(year_df.get(term, 0))
        in_vocab = term in vocab
        raw_weight = float(raw_normalized.get(term, 0.0))
        final_weight = float(filtered_weights.get(term, 0.0))

        status = "kept_in_final_similarity"
        reason = "kept"
        reason_metrics: Dict[str, Any] = {}
        global_df_ratio = (global_df_value / total_docs) if total_docs else None
        year_df_ratio = (year_df_value / docs_in_year) if docs_in_year else None
        if not in_vocab:
            if global_df_value < int(cfg.min_term_count):
                status = "dropped_before_vectorization"
                reason = "global_df_below_min_term_count"
                reason_metrics = {
                    "global_df": global_df_value,
                    "min_term_count": int(cfg.min_term_count),
                    "shortfall_count": int(cfg.min_term_count) - global_df_value,
                    "total_docs": total_docs,
                }
            elif total_docs > 0 and (global_df_value / total_docs) > float(cfg.max_doc_freq_ratio):
                status = "dropped_before_vectorization"
                reason = "global_df_above_max_doc_freq_ratio"
                reason_metrics = {
                    "global_df": global_df_value,
                    "global_df_ratio": global_df_ratio,
                    "global_df_ratio_percent": None if global_df_ratio is None else round(global_df_ratio * 100, 4),
                    "max_doc_freq_ratio": float(cfg.max_doc_freq_ratio),
                    "max_doc_freq_ratio_percent": round(float(cfg.max_doc_freq_ratio) * 100, 4),
                    "ratio_excess": None if global_df_ratio is None else global_df_ratio - float(cfg.max_doc_freq_ratio),
                    "ratio_excess_percent": None if global_df_ratio is None else round((global_df_ratio - float(cfg.max_doc_freq_ratio)) * 100, 4),
                    "total_docs": total_docs,
                }
            else:
                status = "dropped_before_vectorization"
                reason = "not_in_final_vocab"
                reason_metrics = {
                    "global_df": global_df_value,
                    "total_docs": total_docs,
                }
        elif docs_before <= 0 and raw_weight == 0.0:
            status = "kept_but_zero_weight"
            reason = "no_history_before_target_year_idf_zero"
            reason_metrics = {
                "docs_before_year": docs_before,
            }
        elif final_weight <= 0.0:
            status = "pruned_before_final_similarity"
            reason = prune_reason.get(term, "unknown_pruning_reason")
            if reason == "manual_stopword_pruning":
                reason_metrics = {
                    "manual_stopwords_path": cfg.manual_stopwords_path,
                }
            elif reason == "year_df_ratio_pruning":
                reason_metrics = {
                    "year_df": year_df_value,
                    "docs_in_year": docs_in_year,
                    "year_df_ratio": year_df_ratio,
                    "year_df_ratio_percent": None if year_df_ratio is None else round(year_df_ratio * 100, 4),
                    "df_ratio_threshold": float(cfg.df_ratio_threshold),
                    "df_ratio_threshold_percent": round(float(cfg.df_ratio_threshold) * 100, 4),
                    "ratio_excess": None if year_df_ratio is None else year_df_ratio - float(cfg.df_ratio_threshold),
                    "ratio_excess_percent": None if year_df_ratio is None else round((year_df_ratio - float(cfg.df_ratio_threshold)) * 100, 4),
                }
            elif reason == "year_top_df_pruning":
                reason_metrics = {
                    "year_df": year_df_value,
                    "docs_in_year": docs_in_year,
                    "year_df_ratio": year_df_ratio,
                    "year_df_ratio_percent": None if year_df_ratio is None else round(year_df_ratio * 100, 4),
                    "top_df_rank": top_df_rank.get(term),
                    "top_df_cutoff_rank": top_df_k,
                    "top_df_percent": float(cfg.top_df_percent),
                    "top_df_percent_display": round(float(cfg.top_df_percent) * 100, 4),
                    "year_vocab_size_after_ratio_filter": vocab_size_in_year,
                }
            elif reason == "document_topk_pruning":
                reason_metrics = {
                    "doc_weight_rank_before_topk": doc_weight_rank_before_topk.get(term),
                    "topk_terms_per_doc": int(cfg.topk_terms_per_doc),
                    "candidate_terms_before_topk": len(kept_after_column_pruning),
                }

        reason_detail = build_reason_detail(reason=reason, metrics=reason_metrics)

        detail = {
            "term": term,
            "tf": int(token_counter[term]),
            "global_df": global_df_value,
            "global_df_ratio": global_df_ratio,
            "year_df": year_df_value,
            "year_df_ratio": year_df_ratio,
            "docs_before_year": docs_before,
            "historical_df_before_year": int(cumulative_df.get(term, 0)),
            "in_vocab": in_vocab,
            "raw_weight": raw_weight,
            "final_weight": final_weight,
            "participates_in_final_similarity": final_weight > 0.0,
            "status": status,
            "reason": reason,
            "reason_metrics": reason_metrics,
            "reason_detail": reason_detail,
        }
        term_details.append(detail)

    return {
        "summary": {
            "token_count": len(tokens),
            "unique_token_count": len(token_counter),
            "docs_before_year": docs_before,
            "docs_in_year": docs_in_year,
            "kept_term_count": sum(1 for item in term_details if item["participates_in_final_similarity"]),
            "dropped_before_vectorization_count": sum(1 for item in term_details if item["status"] == "dropped_before_vectorization"),
            "pruned_term_count": sum(1 for item in term_details if item["status"] == "pruned_before_final_similarity"),
            "zero_weight_term_count": sum(1 for item in term_details if item["status"] == "kept_but_zero_weight"),
        },
        "term_details": term_details,
    }


def build_reason_detail(*, reason: str, metrics: Dict[str, Any]) -> str:
    if reason == "global_df_below_min_term_count":
        return (
            f"全局 DF={metrics['global_df']}，低于最小阈值 {metrics['min_term_count']}，"
            f"还差 {metrics['shortfall_count']} 次。"
        )
    if reason == "global_df_above_max_doc_freq_ratio":
        return (
            f"全局 DF 占比={metrics['global_df_ratio_percent']:.4f}% ，超过阈值 "
            f"{metrics['max_doc_freq_ratio_percent']:.4f}% ，超出 {metrics['ratio_excess_percent']:.4f} 个百分点；"
            f"对应 {metrics['global_df']}/{metrics['total_docs']} 篇文档。"
        )
    if reason == "year_df_ratio_pruning":
        return (
            f"当年 DF 占比={metrics['year_df_ratio_percent']:.4f}% ，超过阈值 "
            f"{metrics['df_ratio_threshold_percent']:.4f}% ，超出 {metrics['ratio_excess_percent']:.4f} 个百分点；"
            f"对应 {metrics['year_df']}/{metrics['docs_in_year']} 篇当年专利。"
        )
    if reason == "year_top_df_pruning":
        return (
            f"该词当年 DF 排名第 {metrics['top_df_rank']}，落入按年高频剪枝前 "
            f"{metrics['top_df_cutoff_rank']} 名；当前规则删除前 {metrics['top_df_percent_display']:.4f}% 的高频词。"
            f"该词当年出现 {metrics['year_df']}/{metrics['docs_in_year']} 篇，占比 {metrics['year_df_ratio_percent']:.4f}%。"
        )
    if reason == "document_topk_pruning":
        return (
            f"该词在列剪枝后文档内权重排名第 {metrics['doc_weight_rank_before_topk']}，"
            f"但每篇文档只保留前 {metrics['topk_terms_per_doc']} 个词；"
            f"该文档进入 top-k 竞争的词共有 {metrics['candidate_terms_before_topk']} 个。"
        )
    if reason == "manual_stopword_pruning":
        return f"该词命中手工停用词表：{metrics['manual_stopwords_path']}。"
    if reason == "no_history_before_target_year_idf_zero":
        return "该年份之前没有历史文档，回顾性 IDF 为 0，因此该词当前权重为 0。"
    if reason == "kept":
        return "该词保留进入最终相似度计算。"
    return reason


def analyze_patent_case(
    *,
    stage1_dir: Path,
    raw_data_path: Path,
    application_no: Optional[str],
    application_year: Optional[int],
    title: Optional[str],
    title_contains: Optional[str],
    publication_date: Optional[str],
    config_script: Optional[Path] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    include_raw_cut: bool = True,
    expand_year_search: bool = False,
) -> Dict[str, Any]:
    cfg = build_analysis_config(
        raw_data_path=raw_data_path,
        config_script=config_script,
        overrides=config_overrides,
    )
    matches = iter_stage1_candidates(
        stage1_dir=stage1_dir,
        application_no=application_no,
        application_year=application_year,
        title=title,
        title_contains=title_contains,
        restrict_to_year=True,
    )
    year_confirmed = application_year is None
    expanded_from_year_hint = False
    input_year_found = bool(matches)
    if not matches and application_year is not None and expand_year_search:
        matches = iter_stage1_candidates(
            stage1_dir=stage1_dir,
            application_no=application_no,
            application_year=application_year,
            title=title,
            title_contains=title_contains,
            restrict_to_year=False,
        )
        expanded_from_year_hint = True
    if not matches:
        if application_year is not None:
            raise PatentCaseError(
                f"在 stage1 中未找到申请年份={application_year} 的匹配专利。"
                "如需我继续扩展到所有年份查找，请明确允许扩展查找，或在脚本中加入 --expand-year-search。"
            )
        raise PatentCaseError("没有在 stage1 index 中找到匹配专利，请优先提供申请号或申请年份。")

    selected_match = matches[0]
    year_confirmed = True
    raw_record = load_raw_patent_record(
        raw_data_path=raw_data_path,
        cfg=cfg,
        stage1_match=selected_match,
        publication_date=publication_date,
    )

    if publication_date and raw_record is None and len(matches) > 1:
        filtered_matches = []
        for match in matches:
            record = load_raw_patent_record(
                raw_data_path=raw_data_path,
                cfg=cfg,
                stage1_match=match,
                publication_date=publication_date,
            )
            if record is not None:
                filtered_matches.append((match, record))
        if len(filtered_matches) == 1:
            selected_match, raw_record = filtered_matches[0]
        elif len(filtered_matches) > 1:
            raise PatentCaseError(
                "匹配到多个候选专利，publication_date 过滤后仍不唯一，请补充申请号或申请年份。"
            )
        else:
            raise PatentCaseError("找到 stage1 候选，但原始数据中没有匹配 publication_date 的专利。")
    elif publication_date and raw_record is None:
        raise PatentCaseError("stage1 已匹配到专利，但原始数据中的 publication_date 不匹配。")
    elif len(matches) > 1 and not application_no:
        raise PatentCaseError("匹配到多个候选专利，请补充申请号或申请年份以避免全量扫描歧义。")

    token_record = load_stage1_token_record(stage1_dir, selected_match)
    tokens = list(token_record.get("tokens", []))
    bsfs = load_bsfs_row(stage1_dir, selected_match, cfg.epsilon)
    quality_rank = load_quality_rank(stage1_dir, selected_match, cfg.epsilon)

    text_parts = {}
    text_joined = ""
    warnings: List[str] = []
    stopwords = set()
    if cfg.stopword_paths:
        stopwords = set(_load_stopwords_cached(tuple(cfg.stopword_paths)))

    tokenization_info: Dict[str, Any] = {
        "stage1_tokens": tokens,
        "stage1_token_count": len(tokens),
        "stage1_unique_token_count": len(set(tokens)),
    }
    if raw_record is not None:
        for column in cfg.col_text_parts:
            value = raw_record.get(column)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            text_parts[column] = str(value)
        text_joined = cfg.text_sep.join(value for value in text_parts.values() if value)
        _ensure_jieba_initialized_cached(cfg.user_dict_path)
        recomputed_tokens = tokenize(text_joined, stopwords)
        tokenization_info["recomputed_stage1_tokens"] = recomputed_tokens
        tokenization_info["recomputed_matches_stage1"] = recomputed_tokens == tokens
        if recomputed_tokens != tokens:
            warnings.append("原始文本重新分词结果与 stage1 tokens 不完全一致，请检查配置脚本或原始文本来源。")
        tokenization_info.update(
            summarize_initial_filters(
                text_joined,
                stopwords=stopwords,
                include_raw_cut=include_raw_cut,
            )
        )
    else:
        warnings.append("未在原始数据中找到专利文本，只输出 stage1 token 与词项保留/丢弃分析。")

    term_analysis = analyze_terms(
        stage1_dir=stage1_dir,
        cfg=cfg,
        year=selected_match.application_year,
        tokens=tokens,
    )

    result = {
        "patent": {
            "application_no": selected_match.application_no,
            "application_year": selected_match.application_year,
            "title": selected_match.title,
            "row_index": selected_match.row_index,
            "stage1_extras": selected_match.extras,
            **bsfs,
        },
        "year_quality_rank": quality_rank,
        "lookup": {
            "stage1_dir": str(stage1_dir),
            "raw_data_path": str(raw_data_path),
            "matched_candidates": len(matches),
            "input_year": application_year,
            "input_year_found": input_year_found,
            "year_confirmed": year_confirmed,
            "expanded_from_year_hint": expanded_from_year_hint,
            "resolved_application_year": selected_match.application_year,
        },
        "config_used": {
            "config_script": str(config_script) if config_script else None,
            "raw_data_path": cfg.data_path,
            "stopword_paths": cfg.stopword_paths,
            "user_dict_path": cfg.user_dict_path,
            "col_text_parts": cfg.col_text_parts,
            "min_term_count": cfg.min_term_count,
            "max_doc_freq_ratio": cfg.max_doc_freq_ratio,
            "manual_stopwords_path": cfg.manual_stopwords_path,
            "df_ratio_threshold": cfg.df_ratio_threshold,
            "top_df_percent": cfg.top_df_percent,
            "topk_terms_per_doc": cfg.topk_terms_per_doc,
        },
        "raw_record": raw_record,
        "text_parts": text_parts,
        "tokenization": tokenization_info,
        "term_analysis": term_analysis,
        "warnings": warnings,
    }
    return result


def default_output_path(*, stage1_dir: Path, experiment_id: str, application_no: str) -> Path:
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", application_no)
    return stage1_dir.parent / "verification" / "case_analysis" / f"{experiment_id}_{safe_name}.json"


def write_analysis_json(result: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def format_console_summary(result: Dict[str, Any], top_terms: int = 15) -> str:
    patent = result["patent"]
    year_quality_rank = result.get("year_quality_rank", {})
    tokenization = result["tokenization"]
    term_details = result["term_analysis"]["term_details"]
    kept = [item for item in term_details if item["participates_in_final_similarity"]]
    kept.sort(key=lambda item: item["final_weight"], reverse=True)
    dropped = [item for item in term_details if not item["participates_in_final_similarity"]]
    dropped.sort(key=lambda item: item["raw_weight"], reverse=True)

    lines = [
        f"申请号: {patent['application_no']}",
        f"申请年份: {patent['application_year']}",
        f"专利名称: {patent['title']}",
        f"row_index: {patent['row_index']}",
        f"BS={patent['BS']:.8f} FS={patent['FS']:.8f} Quality_q={patent['Quality_q']:.8f}",
        "",
        f"stage1 token 数: {tokenization['stage1_token_count']} (unique={tokenization['stage1_unique_token_count']})",
    ]
    if year_quality_rank.get("available"):
        lines.extend(
            [
                (
                    f"Quality_q 年内排名: {year_quality_rank['rank_desc']}/{year_quality_rank['total_patents_in_year']} "
                    f"(Top {year_quality_rank['top_percent_display']:.4f}%)"
                ),
                (
                    f"超过同年专利占比: {year_quality_rank['outperform_percent_display']:.4f}% "
                    f"({year_quality_rank['worse_count']}/{year_quality_rank['total_patents_in_year']})"
                ),
            ]
        )
    if tokenization.get("recomputed_matches_stage1") is not None:
        lines.append(f"原始文本重分词与 stage1 一致: {tokenization['recomputed_matches_stage1']}")
    if result["warnings"]:
        lines.append("warnings:")
        for warning in result["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("保留下来的高权重词:")
    for item in kept[:top_terms]:
        lines.append(
            f"- {item['term']} tf={item['tf']} raw={item['raw_weight']:.6f} final={item['final_weight']:.6f}"
        )
    lines.append("")
    lines.append("被舍弃的高权重词:")
    for item in dropped[:top_terms]:
        lines.append(
            f"- {item['term']} tf={item['tf']} raw={item['raw_weight']:.6f} "
            f"reason={item['reason']} | {item['reason_detail']}"
        )
    return "\n".join(lines)


def resolve_optional_path(path: Optional[str], *, base_dir: Optional[Path] = None) -> Optional[Path]:
    if not path:
        return None
    return resolve_project_path(path, base_dir=base_dir)


__all__ = [
    "PatentCaseError",
    "analyze_patent_case",
    "build_analysis_config",
    "default_output_path",
    "format_console_summary",
    "parse_config_literals",
    "resolve_optional_path",
    "write_analysis_json",
]
