from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time
from typing import Any, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from common.analysis import INVALID_UCC_VALUES, normalize_string_series, to_numeric  # noqa: E402
from common.io import READ_ENCODINGS, build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


DEFAULT_RAW_PATENT_DIR = "data/raw/中国专利分年份保存数据1985-2025"

ID_COL = "申请号"
YEAR_COL = "申请年份"
PUBLIC_YEAR_COL = "公开公告年份"
PATENT_TYPE_COL = "专利类型"
APPLICANT_COL = "申请人"
UCC_COL = "统一社会信用代码"
SCORE_COL = "Quality_q"
AUTHORIZED_PATENT_TYPE = "发明授权"

RAW_PUBLIC_DATE_COLUMNS = ("公开公告日", "公开（公告）日")

OPTIONAL_PANEL_COLUMNS = [
    PUBLIC_YEAR_COL,
    PATENT_TYPE_COL,
    "专利名称",
    APPLICANT_COL,
    UCC_COL,
    "BS",
    "FS",
    "被引证次数",
]
PANEL_COLUMNS_TO_READ = [ID_COL, YEAR_COL, SCORE_COL, *OPTIONAL_PANEL_COLUMNS]
UCC_COLUMNS_TO_READ = ["Stkid", "ShortName", "year", "UCC"]

TOP_SORT_COLUMNS = [SCORE_COL, "FS", "BS", "被引证次数", ID_COL]
TOP_SORT_ASCENDING = [False, False, False, False, True]

RAW_DETAIL_OUTPUT_COLUMNS = [
    "raw_专利名称",
    "raw_摘要文本",
    "raw_申请日",
    "raw_公开公告日",
    "raw_授权公告日",
    "raw_申请人",
]

OUTPUT_COLUMNS = [
    "年内排名",
    ID_COL,
    YEAR_COL,
    PUBLIC_YEAR_COL,
    "申请日",
    "公开公告日",
    "授权公告日",
    PATENT_TYPE_COL,
    "专利名称",
    "摘要文本",
    APPLICANT_COL,
    UCC_COL,
    "公司名称",
    "公司名称来源",
    "证券ID",
    "BS",
    "FS",
    SCORE_COL,
    "被引证次数",
]


def _join_unique(values: Any) -> str:
    raw_values = values.tolist() if isinstance(values, pd.Series) else list(values)
    unique_values = sorted(
        {
            normalized
            for value in raw_values
            for normalized in [_normalize_scalar(value)]
            if normalized and normalized.lower() not in {"nan", "none"}
        }
    )
    return ";".join(unique_values)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    missing: list[str] = []
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
            missing.append(column)
    return missing


def _normalize_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _close_logger_handlers(logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _open_csv_reader(
    path: Path,
    *,
    chunksize: int,
    usecols: Any = None,
    dtype: Any = str,
):
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            return pd.read_csv(
                path,
                chunksize=chunksize,
                usecols=usecols,
                dtype=dtype,
                encoding=encoding,
                low_memory=False,
                engine="c",
                on_bad_lines="skip",
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def _sort_top_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=TOP_SORT_COLUMNS,
        ascending=TOP_SORT_ASCENDING,
        kind="mergesort",
        na_position="last",
    )


def _normalize_patent_chunk(chunk: pd.DataFrame, *, missing_optional: list[str]) -> pd.DataFrame:
    chunk = chunk.copy()
    _ensure_columns(chunk, missing_optional)

    chunk[ID_COL] = normalize_string_series(chunk[ID_COL])
    chunk[YEAR_COL] = to_numeric(chunk[YEAR_COL]).astype("Int64")
    chunk[PUBLIC_YEAR_COL] = to_numeric(chunk[PUBLIC_YEAR_COL]).astype("Int64")
    chunk[SCORE_COL] = to_numeric(chunk[SCORE_COL])
    chunk["BS"] = to_numeric(chunk["BS"])
    chunk["FS"] = to_numeric(chunk["FS"])
    chunk["被引证次数"] = to_numeric(chunk["被引证次数"])
    chunk[APPLICANT_COL] = normalize_string_series(chunk[APPLICANT_COL])
    chunk[UCC_COL] = normalize_string_series(chunk[UCC_COL])
    chunk["专利名称"] = normalize_string_series(chunk["专利名称"])
    chunk[PATENT_TYPE_COL] = normalize_string_series(chunk[PATENT_TYPE_COL])

    chunk = chunk[chunk[ID_COL] != ""].copy()
    chunk = chunk[chunk[YEAR_COL].notna() & chunk[SCORE_COL].notna()].copy()
    chunk["raw_year_hint"] = chunk[PUBLIC_YEAR_COL].fillna(chunk[YEAR_COL]).astype("Int64")
    return chunk


def _select_top_patents_by_year(
    panel_path: Path,
    *,
    top_n: int,
    batch_size: int,
    logger,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    parquet = pq.ParquetFile(panel_path)
    available_columns = parquet.schema_arrow.names

    required_columns = [ID_COL, YEAR_COL, SCORE_COL]
    missing_required = [column for column in required_columns if column not in available_columns]
    if missing_required:
        raise KeyError(f"experiment_patent_panel 缺少列: {missing_required}")

    missing_optional = [column for column in OPTIONAL_PANEL_COLUMNS if column not in available_columns]
    columns_to_read = [column for column in PANEL_COLUMNS_TO_READ if column in available_columns]

    total_rows = int(parquet.metadata.num_rows)
    logger.info(
        "开始按申请年份流式筛选年度 top%s，源文件总行数=%s，batch_size=%s，读取列数=%s",
        top_n,
        total_rows,
        batch_size,
        len(columns_to_read),
    )

    top_by_year: dict[int, pd.DataFrame] = {}
    rows_read = 0
    rows_eligible = 0
    batch_count = 0

    for batch_count, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=columns_to_read, use_threads=True),
        start=1,
    ):
        chunk = _normalize_patent_chunk(batch.to_pandas(), missing_optional=missing_optional)
        rows_read += batch.num_rows
        rows_eligible += len(chunk)

        if not chunk.empty:
            for year, year_chunk in chunk.groupby(YEAR_COL, sort=False):
                year_int = int(year)
                existing = top_by_year.get(year_int)
                merged = year_chunk if existing is None else pd.concat([existing, year_chunk], ignore_index=True)
                top_by_year[year_int] = _sort_top_frame(merged).head(top_n).copy()

        if batch_count == 1 or batch_count % 5 == 0 or rows_read >= total_rows:
            logger.info(
                "年度 top%s 流式筛选进度: batch=%s, 已读行数=%s/%s, 合格候选=%s, 当前年份数=%s",
                top_n,
                batch_count,
                rows_read,
                total_rows,
                rows_eligible,
                len(top_by_year),
            )

    if not top_by_year:
        empty = pd.DataFrame(columns=[*PANEL_COLUMNS_TO_READ, "raw_year_hint", "年内排名"])
        return empty, missing_optional, {
            "rows_total": total_rows,
            "rows_eligible": 0,
            "batches": batch_count,
            "years": 0,
            "rows_selected": 0,
        }

    top_frames: list[pd.DataFrame] = []
    years = sorted(top_by_year)
    for year in years:
        year_top = _sort_top_frame(top_by_year[year]).head(top_n).copy()
        year_top["年内排名"] = range(1, len(year_top) + 1)
        top_frames.append(year_top)
        logger.info("年度 top%s 筛选完成: 年份=%s, 输出行数=%s", top_n, year, len(year_top))

    top_df = pd.concat(top_frames, ignore_index=True)
    logger.info("年度 top%s 流式筛选完成，年份数=%s，总输出行数=%s", top_n, len(years), len(top_df))
    return top_df, missing_optional, {
        "rows_total": total_rows,
        "rows_eligible": rows_eligible,
        "batches": batch_count,
        "years": len(years),
        "rows_selected": int(len(top_df)),
    }


def _fallback_company_names(patent_df: pd.DataFrame) -> pd.DataFrame:
    fallback = patent_df.copy()
    fallback["证券ID"] = ""
    fallback["公司名称"] = ""
    fallback["公司名称来源"] = "缺失"

    use_applicant = fallback[APPLICANT_COL] != ""
    fallback.loc[use_applicant, "公司名称"] = fallback.loc[use_applicant, APPLICANT_COL]
    fallback.loc[use_applicant, "公司名称来源"] = "专利申请人回退"
    return fallback


def _normalize_ucc_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["Stkid"] = normalize_string_series(chunk["Stkid"])
    chunk["ShortName"] = normalize_string_series(chunk["ShortName"])
    chunk["year"] = to_numeric(chunk["year"]).astype("Int64")
    chunk["UCC"] = normalize_string_series(chunk["UCC"])
    chunk = chunk[
        chunk["year"].notna()
        & (chunk["UCC"] != "")
        & (~chunk["UCC"].isin(INVALID_UCC_VALUES))
    ].copy()
    return chunk


def _attach_company_names(
    patent_df: pd.DataFrame,
    *,
    ucc_path: Path,
    batch_size: int,
    logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target_uccs = {
        value
        for value in normalize_string_series(patent_df[UCC_COL]).tolist()
        if value and value not in INVALID_UCC_VALUES
    }
    if not target_uccs:
        logger.info("top 专利中没有可用 UCC，跳过 UCC 映射，直接回退到申请人")
        fallback = _fallback_company_names(patent_df)
        return fallback, {
            "target_uccs": 0,
            "rows_scanned": 0,
            "rows_matched": 0,
            "batches": 0,
        }

    logger.info(
        "开始补公司名称，目标 UCC 数=%s，UCC 源=%s，batch_size=%s",
        len(target_uccs),
        repo_relative(ucc_path),
        batch_size,
    )

    year_level_stkid: dict[tuple[str, int], set[str]] = {}
    year_level_shortname: dict[tuple[str, int], set[str]] = {}
    history_level_stkid: dict[str, set[str]] = {}
    history_level_shortname: dict[str, set[str]] = {}

    rows_scanned = 0
    rows_matched = 0
    batch_count = 0

    if ucc_path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(ucc_path)
        missing_required = [column for column in UCC_COLUMNS_TO_READ if column not in parquet.schema_arrow.names]
        if missing_required:
            raise KeyError(f"UCC 映射缺少列: {missing_required}")
        iter_source = parquet.iter_batches(batch_size=batch_size, columns=UCC_COLUMNS_TO_READ, use_threads=True)
        total_rows = int(parquet.metadata.num_rows)
        for batch_count, batch in enumerate(iter_source, start=1):
            chunk = _normalize_ucc_chunk(batch.to_pandas())
            rows_scanned += batch.num_rows
            chunk = chunk[chunk["UCC"].isin(target_uccs)].copy()
            rows_matched += len(chunk)

            for row in chunk.itertuples(index=False):
                year_key = (row.UCC, int(row.year))
                year_level_stkid.setdefault(year_key, set()).add(row.Stkid)
                year_level_shortname.setdefault(year_key, set()).add(row.ShortName)
                history_level_stkid.setdefault(row.UCC, set()).add(row.Stkid)
                history_level_shortname.setdefault(row.UCC, set()).add(row.ShortName)

            if batch_count == 1 or batch_count % 5 == 0 or rows_scanned >= total_rows:
                logger.info(
                    "UCC 映射读取进度: batch=%s, 已读行数=%s/%s, 命中行数=%s",
                    batch_count,
                    rows_scanned,
                    total_rows,
                    rows_matched,
                )
    else:
        reader = _open_csv_reader(
            ucc_path,
            chunksize=batch_size,
            usecols=lambda name: name in UCC_COLUMNS_TO_READ,
        )
        for batch_count, chunk in enumerate(reader, start=1):
            chunk = _normalize_ucc_chunk(chunk)
            rows_scanned += len(chunk)
            chunk = chunk[chunk["UCC"].isin(target_uccs)].copy()
            rows_matched += len(chunk)

            for row in chunk.itertuples(index=False):
                year_key = (row.UCC, int(row.year))
                year_level_stkid.setdefault(year_key, set()).add(row.Stkid)
                year_level_shortname.setdefault(year_key, set()).add(row.ShortName)
                history_level_stkid.setdefault(row.UCC, set()).add(row.Stkid)
                history_level_shortname.setdefault(row.UCC, set()).add(row.ShortName)

            if batch_count == 1 or batch_count % 5 == 0:
                logger.info("UCC 映射读取进度: batch=%s, 命中行数=%s", batch_count, rows_matched)

    merged = _fallback_company_names(patent_df)
    year_keys = [
        (ucc, int(year))
        if ucc and pd.notna(year)
        else None
        for ucc, year in zip(merged[UCC_COL].tolist(), merged[YEAR_COL].tolist())
    ]
    merged["证券ID"] = [
        _join_unique(year_level_stkid.get(key, set())) if key is not None else ""
        for key in year_keys
    ]
    merged["上市公司简称"] = [
        _join_unique(year_level_shortname.get(key, set())) if key is not None else ""
        for key in year_keys
    ]
    merged["证券ID_历史"] = [
        _join_unique(history_level_stkid.get(ucc, set())) if ucc else ""
        for ucc in merged[UCC_COL].tolist()
    ]
    merged["上市公司简称_历史"] = [
        _join_unique(history_level_shortname.get(ucc, set())) if ucc else ""
        for ucc in merged[UCC_COL].tolist()
    ]

    use_year_level = merged["上市公司简称"] != ""
    use_history_level = (~use_year_level) & (merged["上市公司简称_历史"] != "")
    use_applicant = (~use_year_level) & (~use_history_level) & (merged[APPLICANT_COL] != "")

    merged.loc[use_year_level, "公司名称"] = merged.loc[use_year_level, "上市公司简称"]
    merged.loc[use_year_level, "公司名称来源"] = "上市公司UCC映射(当年)"
    merged.loc[use_history_level, "公司名称"] = merged.loc[use_history_level, "上市公司简称_历史"]
    merged.loc[use_history_level, "公司名称来源"] = "上市公司UCC映射(历史)"
    merged.loc[~(use_year_level | use_history_level | use_applicant), "公司名称"] = ""
    merged.loc[~(use_year_level | use_history_level | use_applicant), "公司名称来源"] = "缺失"

    merged.loc[(merged["证券ID"] == "") & (merged["证券ID_历史"] != ""), "证券ID"] = merged.loc[
        (merged["证券ID"] == "") & (merged["证券ID_历史"] != ""),
        "证券ID_历史",
    ]

    merged = merged.drop(columns=["上市公司简称", "证券ID_历史", "上市公司简称_历史"])
    logger.info(
        "公司名称补全完成，当年映射=%s，历史映射=%s，申请人回退=%s，缺失=%s",
        int((merged["公司名称来源"] == "上市公司UCC映射(当年)").sum()),
        int((merged["公司名称来源"] == "上市公司UCC映射(历史)").sum()),
        int((merged["公司名称来源"] == "专利申请人回退").sum()),
        int((merged["公司名称来源"] == "缺失").sum()),
    )
    return merged, {
        "target_uccs": len(target_uccs),
        "rows_scanned": rows_scanned,
        "rows_matched": rows_matched,
        "batches": batch_count,
    }


def _preferred_lookup_order(lookup_dir: Path, year_hints: list[int]) -> list[Path]:
    lookup_paths = sorted(
        path
        for path in lookup_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
    )
    if not year_hints:
        return lookup_paths

    preferred: list[Path] = []
    seen: set[Path] = set()
    for year in year_hints:
        for path in lookup_paths:
            if path in seen:
                continue
            if str(year) in path.stem:
                preferred.append(path)
                seen.add(path)
    preferred.extend([path for path in lookup_paths if path not in seen])
    return preferred


def _has_authorized_parquet_parts(path: Path) -> bool:
    return path.exists() and any(part.suffix.lower() == ".parquet" for part in path.iterdir() if part.is_file())


def _resolve_raw_lookup_dir(*, raw_patent_dir: Path, shared_root: str) -> tuple[Path, str]:
    shared_paths = build_shared_paths(shared_root)
    authorized_dir = shared_paths.raw_patent_authorized_parts_dir
    default_raw_dir = resolve_repo_path(DEFAULT_RAW_PATENT_DIR)
    if (
        default_raw_dir is not None
        and raw_patent_dir.resolve() == default_raw_dir.resolve()
        and _has_authorized_parquet_parts(authorized_dir)
    ):
        return authorized_dir, "shared_authorized_parquet_parts"
    return raw_patent_dir, "raw_patent_dir"


def _select_raw_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    authorized_rows = [row for row in rows if _normalize_scalar(row.get(PATENT_TYPE_COL)) == AUTHORIZED_PATENT_TYPE]
    pool = authorized_rows or rows
    return pool[0]


def _extract_raw_fields(row: dict[str, Any]) -> dict[str, str]:
    public_date = ""
    for column in RAW_PUBLIC_DATE_COLUMNS:
        public_date = _normalize_scalar(row.get(column))
        if public_date:
            break

    return {
        "raw_专利名称": _normalize_scalar(row.get("专利名称")),
        "raw_摘要文本": _normalize_scalar(row.get("摘要文本")),
        "raw_申请日": _normalize_scalar(row.get("申请日")),
        "raw_公开公告日": public_date,
        "raw_授权公告日": _normalize_scalar(row.get("授权公告日")),
        "raw_申请人": _normalize_scalar(row.get(APPLICANT_COL)),
    }


def _lookup_raw_details(
    patents: pd.DataFrame,
    *,
    raw_lookup_dir: Path,
    chunksize: int,
    logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not raw_lookup_dir.exists():
        raise FileNotFoundError(f"找不到原始专利目录: {raw_lookup_dir}")

    target_df = patents[[ID_COL, "raw_year_hint"]].drop_duplicates().copy()
    unresolved = set(target_df[ID_COL].tolist())
    if not unresolved:
        empty = patents[[ID_COL]].drop_duplicates().copy()
        for column in RAW_DETAIL_OUTPUT_COLUMNS:
            empty[column] = ""
        return empty, {"matched": 0, "unmatched": 0, "files_scanned": 0}

    year_hints = [
        int(value)
        for value in target_df["raw_year_hint"].dropna().astype(int).drop_duplicates().tolist()
    ]
    lookup_paths = _preferred_lookup_order(raw_lookup_dir, year_hints)
    if not lookup_paths:
        raise FileNotFoundError(f"原始专利目录下找不到可用的 csv/parquet 文件: {raw_lookup_dir}")
    usecol_list = [
        ID_COL,
        PATENT_TYPE_COL,
        "专利名称",
        "摘要文本",
        "申请日",
        "授权公告日",
        APPLICANT_COL,
        *RAW_PUBLIC_DATE_COLUMNS,
    ]
    usecol_set = set(usecol_list)

    matched_rows: dict[str, dict[str, str]] = {}
    files_scanned = 0

    for source_path in lookup_paths:
        if not unresolved:
            break
        files_scanned += 1
        file_candidates: dict[str, list[dict[str, Any]]] = {}
        logger.info("回查原始专利文件: %s，待匹配申请号=%s", repo_relative(source_path), len(unresolved))

        if source_path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(source_path)
            if ID_COL not in parquet.schema_arrow.names:
                logger.warning("原始 parquet 缺少申请号列，跳过: %s", repo_relative(source_path))
                continue
            columns_to_read = [column for column in usecol_list if column in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(batch_size=chunksize, columns=columns_to_read, use_threads=True):
                chunk = batch.to_pandas()
                chunk[ID_COL] = chunk[ID_COL].astype("string").fillna("").str.strip()
                chunk = chunk[chunk[ID_COL].isin(unresolved)]
                if chunk.empty:
                    continue
                for _, row in chunk.iterrows():
                    pid = str(row.get(ID_COL, "")).strip()
                    if not pid:
                        continue
                    file_candidates.setdefault(pid, []).append(row.to_dict())
        else:
            try:
                reader = _open_csv_reader(
                    source_path,
                    chunksize=chunksize,
                    usecols=lambda name: name in usecol_set,
                )
            except RuntimeError:
                logger.warning("无法读取原始专利文件，跳过: %s", repo_relative(source_path))
                continue

            for chunk in reader:
                chunk = chunk.copy()
                chunk[ID_COL] = chunk[ID_COL].astype("string").fillna("").str.strip()
                chunk = chunk[chunk[ID_COL].isin(unresolved)]
                if chunk.empty:
                    continue
                for _, row in chunk.iterrows():
                    pid = str(row.get(ID_COL, "")).strip()
                    if not pid:
                        continue
                    file_candidates.setdefault(pid, []).append(row.to_dict())

        if not file_candidates:
            continue

        for pid, rows in file_candidates.items():
            matched_rows[pid] = _extract_raw_fields(_select_raw_candidate(rows))
            unresolved.discard(pid)

    detail_df = patents[[ID_COL]].drop_duplicates().copy()
    detail_df["raw_专利名称"] = detail_df[ID_COL].map(lambda pid: matched_rows.get(pid, {}).get("raw_专利名称", ""))
    detail_df["raw_摘要文本"] = detail_df[ID_COL].map(lambda pid: matched_rows.get(pid, {}).get("raw_摘要文本", ""))
    detail_df["raw_申请日"] = detail_df[ID_COL].map(lambda pid: matched_rows.get(pid, {}).get("raw_申请日", ""))
    detail_df["raw_公开公告日"] = detail_df[ID_COL].map(lambda pid: matched_rows.get(pid, {}).get("raw_公开公告日", ""))
    detail_df["raw_授权公告日"] = detail_df[ID_COL].map(lambda pid: matched_rows.get(pid, {}).get("raw_授权公告日", ""))
    detail_df["raw_申请人"] = detail_df[ID_COL].map(lambda pid: matched_rows.get(pid, {}).get("raw_申请人", ""))

    return detail_df, {
        "matched": len(matched_rows),
        "unmatched": len(unresolved),
        "files_scanned": files_scanned,
    }


def export_top_patents_by_year(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    experiment_patent_panel_path: Optional[Path] = None,
    ucc_exploded_path: Optional[Path] = None,
    raw_patent_dir: str = DEFAULT_RAW_PATENT_DIR,
    shared_root: str = "outputs/shared",
    top_n: int = 100,
    raw_lookup_chunksize: int = 50000,
    panel_batch_size: int = 200000,
    ucc_batch_size: int = 200000,
    skip_company_lookup: bool = False,
    skip_raw_lookup: bool = False,
) -> dict[str, object]:
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(
        f"export_top_patents_by_year.{experiment_id}",
        paths.logs_dir / "export_top_patents_by_year.log",
    )
    try:
        patent_path = experiment_patent_panel_path or (paths.data_dir / "experiment_patent_panel.parquet")
        if not patent_path.exists():
            raise FileNotFoundError(f"找不到 experiment_patent_panel: {patent_path}")

        effective_raw_patent_dir = resolve_repo_path(raw_patent_dir)
        assert effective_raw_patent_dir is not None
        effective_raw_lookup_dir, raw_lookup_source = _resolve_raw_lookup_dir(
            raw_patent_dir=effective_raw_patent_dir,
            shared_root=shared_root,
        )

        if top_n <= 0:
            summary = {
                "experiment_id": experiment_id,
                "experiment_patent_panel_path": repo_relative(patent_path),
                "raw_patent_dir": repo_relative(effective_raw_patent_dir),
                "raw_lookup_dir": repo_relative(effective_raw_lookup_dir),
                "raw_lookup_source": raw_lookup_source,
                "top_n": int(top_n),
                "skipped": True,
                "reason": "top_n <= 0",
                "output_paths": [],
            }
            write_json(paths.metadata_dir / "export_top_patents_by_year.json", summary)
            return summary

        effective_ucc_path = ucc_exploded_path
        if effective_ucc_path is None:
            shared_paths = build_shared_paths(shared_root)
            parquet_candidate = shared_paths.ucc_mapping_dir / "ucc_exploded.parquet"
            if parquet_candidate.exists():
                effective_ucc_path = parquet_candidate
        if effective_ucc_path is None and not skip_company_lookup:
            raise FileNotFoundError("缺少共享 UCC 映射，请先运行 run_shared_prep.py 生成 shared ucc_mapping")

        logger.info("读取专利实验面板并流式筛选年度 top%s: %s", top_n, repo_relative(patent_path))
        step_started = time.perf_counter()
        top_df, missing_optional, panel_selection_stats = _select_top_patents_by_year(
            patent_path,
            top_n=top_n,
            batch_size=panel_batch_size,
            logger=logger,
        )
        logger.info("年度 top%s 筛选阶段完成，用时 %.1fs", top_n, time.perf_counter() - step_started)

        company_lookup_stats = {
            "skipped": bool(skip_company_lookup),
            "target_uccs": 0,
            "rows_scanned": 0,
            "rows_matched": 0,
            "batches": 0,
        }
        if skip_company_lookup:
            logger.info("按参数跳过 UCC 公司名映射，直接使用申请人回退")
            top_df = _fallback_company_names(top_df)
        else:
            step_started = time.perf_counter()
            top_df, company_lookup_stats = _attach_company_names(
                top_df,
                ucc_path=effective_ucc_path,
                batch_size=ucc_batch_size,
                logger=logger,
            )
            company_lookup_stats["skipped"] = False
            logger.info("公司名称补全阶段完成，用时 %.1fs", time.perf_counter() - step_started)

        if skip_raw_lookup:
            logger.info("按参数跳过原始专利 CSV 回查，摘要和日期字段将保留为空")
            raw_details_df = top_df[[ID_COL]].drop_duplicates().copy()
            for column in RAW_DETAIL_OUTPUT_COLUMNS:
                raw_details_df[column] = ""
            raw_lookup_stats = {
                "matched": 0,
                "unmatched": int(len(raw_details_df)),
                "files_scanned": 0,
                "skipped": True,
            }
        else:
            logger.info(
                "开始回查原始专利明细，数据源=%s (%s)。该步骤用于补摘要和日期字段",
                repo_relative(effective_raw_lookup_dir),
                raw_lookup_source,
            )
            step_started = time.perf_counter()
            raw_details_df, raw_lookup_stats = _lookup_raw_details(
                top_df,
                raw_lookup_dir=effective_raw_lookup_dir,
                chunksize=raw_lookup_chunksize,
                logger=logger,
            )
            raw_lookup_stats["skipped"] = False
            logger.info("原始专利回查阶段完成，用时 %.1fs", time.perf_counter() - step_started)

        top_df = top_df.merge(raw_details_df, on=ID_COL, how="left")
        for column in RAW_DETAIL_OUTPUT_COLUMNS:
            top_df[column] = top_df[column].fillna("")

        top_df["专利名称"] = top_df["raw_专利名称"].where(top_df["raw_专利名称"] != "", top_df["专利名称"])
        top_df["摘要文本"] = top_df["raw_摘要文本"]
        top_df["申请日"] = top_df["raw_申请日"]
        top_df["公开公告日"] = top_df["raw_公开公告日"]
        top_df["授权公告日"] = top_df["raw_授权公告日"]
        top_df[APPLICANT_COL] = top_df["raw_申请人"].where(top_df["raw_申请人"] != "", top_df[APPLICANT_COL])
        mapped_mask = top_df["公司名称来源"].isin(["上市公司UCC映射(当年)", "上市公司UCC映射(历史)"])
        fallback_mask = (~mapped_mask) & (top_df[APPLICANT_COL] != "")
        top_df.loc[fallback_mask, "公司名称"] = top_df.loc[fallback_mask, APPLICANT_COL]
        top_df.loc[fallback_mask, "公司名称来源"] = "专利申请人回退"
        top_df.loc[(~mapped_mask) & (~fallback_mask), "公司名称"] = ""
        top_df.loc[(~mapped_mask) & (~fallback_mask), "公司名称来源"] = "缺失"

        output_dir = paths.tables_dir / "top_patents_by_year"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[str] = []
        years_exported: list[int] = []
        for year, year_df in top_df.groupby(YEAR_COL, sort=True):
            year_int = int(year)
            years_exported.append(year_int)
            output_path = output_dir / f"top_patents_year={year_int}_top{top_n}.csv"
            year_df.loc[:, OUTPUT_COLUMNS].to_csv(output_path, index=False, encoding="utf-8-sig")
            output_paths.append(repo_relative(output_path))
            logger.info("已输出年份 %s 的 top%s 专利: %s", year_int, top_n, repo_relative(output_path))

        source_counts = top_df["公司名称来源"].value_counts(dropna=False).to_dict()
        summary = {
            "experiment_id": experiment_id,
            "experiment_patent_panel_path": repo_relative(patent_path),
            "ucc_mapping_path": repo_relative(effective_ucc_path) if effective_ucc_path is not None else None,
            "raw_patent_dir": repo_relative(effective_raw_patent_dir),
            "raw_lookup_dir": repo_relative(effective_raw_lookup_dir),
            "raw_lookup_source": raw_lookup_source,
            "top_n": int(top_n),
            "rows_exported": int(len(top_df)),
            "years_exported": years_exported,
            "missing_optional_columns": missing_optional,
            "panel_selection_stats": panel_selection_stats,
            "company_lookup_stats": company_lookup_stats,
            "company_name_source_counts": {str(key): int(value) for key, value in source_counts.items()},
            "raw_lookup_stats": raw_lookup_stats,
            "output_paths": output_paths,
        }
        write_json(paths.metadata_dir / "export_top_patents_by_year.json", summary)
        logger.info("年度 top 专利明细已输出，年份数=%s，总行数=%s", len(years_exported), len(top_df))
        return summary
    finally:
        _close_logger_handlers(logger)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="按申请年份导出 Quality_q 排名前 top_n 的专利明细")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--experiment-patent-panel-path", help="experiment_patent_panel.parquet 路径")
    parser.add_argument("--ucc-exploded-path", help="共享 ucc_exploded.parquet 路径")
    parser.add_argument("--raw-patent-dir", default=DEFAULT_RAW_PATENT_DIR, help="原始专利 CSV 目录")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--top-n", type=int, default=100, help="每年导出的 top_n 专利数量")
    parser.add_argument("--raw-lookup-chunksize", type=int, default=50000, help="回查原始专利 CSV 的分块行数")
    parser.add_argument("--panel-batch-size", type=int, default=200000, help="流式读取 experiment_patent_panel 的分块行数")
    parser.add_argument("--ucc-batch-size", type=int, default=200000, help="流式读取 UCC 映射的分块行数")
    parser.add_argument("--skip-company-lookup", action="store_true", help="跳过 UCC 公司名映射，仅保留申请人回退")
    parser.add_argument("--skip-raw-lookup", action="store_true", help="跳过原始专利 CSV 回查，摘要和日期列留空")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    export_top_patents_by_year(
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        experiment_patent_panel_path=resolve_repo_path(args.experiment_patent_panel_path) if args.experiment_patent_panel_path else None,
        ucc_exploded_path=resolve_repo_path(args.ucc_exploded_path) if args.ucc_exploded_path else None,
        raw_patent_dir=args.raw_patent_dir,
        shared_root=args.shared_root,
        top_n=args.top_n,
        raw_lookup_chunksize=args.raw_lookup_chunksize,
        panel_batch_size=args.panel_batch_size,
        ucc_batch_size=args.ucc_batch_size,
        skip_company_lookup=args.skip_company_lookup,
        skip_raw_lookup=args.skip_raw_lookup,
    )


if __name__ == "__main__":
    main()
