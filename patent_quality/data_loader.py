import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import Config
from .log import get_logger


def _list_csv_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isfile(p) and p.lower().endswith(".csv"):
                files.append(p)
        files.sort()
        return files
    return [path]


def _list_parquet_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isfile(p) and p.lower().endswith(".parquet"):
                files.append(p)
        files.sort()
        return files
    return [path]


def _parse_year(row: pd.Series, col_year: str, fallback_date_col: str) -> Optional[int]:
    y = row.get(col_year)
    if _is_present(y):
        try:
            return int(str(y)[:4])
        except Exception:
            pass
    d = row.get(fallback_date_col)
    if _is_present(d):
        s = str(d)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return int(pd.to_datetime(s, format=fmt, errors="raise").year)
            except Exception:
                continue
        try:
            return int(pd.to_datetime(s, errors="coerce").year)
        except Exception:
            return None
    return None


def _concat_text(row: pd.Series, parts: List[str], sep: str) -> str:
    vals = []
    for c in parts:
        v = row.get(c)
        if not _is_present(v):
            continue
        s = str(v)
        if s:
            vals.append(s)
    return sep.join(vals)


def _tuple_text(values: tuple[Any, ...], positions: Dict[str, int], column: str) -> str:
    idx = positions.get(column)
    if idx is None:
        return ""
    value = values[idx]
    if not _is_present(value):
        return ""
    return str(value)


def _tuple_concat_text(values: tuple[Any, ...], positions: Dict[str, int], parts: List[str], sep: str) -> str:
    vals: List[str] = []
    for column in parts:
        text = _tuple_text(values, positions, column)
        if text:
            vals.append(text)
    return sep.join(vals)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    missing = pd.isna(value)
    if isinstance(missing, bool):
        return not missing
    return not cast(bool, missing.all())


def _build_public_date_ord_from_series(series: pd.Series) -> Tuple[pd.Series, int]:
    text = series.astype("string").fillna("").str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    parsed_days = parsed.to_numpy(dtype="datetime64[D]")
    valid_mask = parsed.notna().to_numpy()
    invalid_ord = np.iinfo(np.int32).max
    ord_values = np.full(len(series), invalid_ord, dtype=np.int32)
    if valid_mask.any():
        ord_values[valid_mask] = parsed_days[valid_mask].astype(np.int32, copy=False)
    return pd.Series(ord_values, index=series.index, dtype="Int64"), int((~valid_mask).sum())


def _iter_exact_frames(cfg: Config, columns: List[str]) -> Iterator[Tuple[str, pd.DataFrame]]:
    logger = get_logger(level=cfg.log_level)
    invalid_ord = np.iinfo(np.int32).max
    for path in _list_parquet_files(cfg.input_path):
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if cfg.col_id not in read_columns:
            raise KeyError(f"exact 模式输入缺少列: {cfg.col_id} ({path})")
        frame = pd.read_parquet(path, columns=read_columns)
        if cfg.col_type in frame.columns:
            frame = frame[frame[cfg.col_type].astype("string").fillna("").str.strip() == "发明授权"].copy()
        frame[cfg.col_id] = frame[cfg.col_id].astype("string").fillna("").str.strip()
        frame = frame[frame[cfg.col_id] != ""].copy()
        has_public_date = cfg.public_date_col in frame.columns
        if cfg.public_year_col not in frame.columns:
            if not has_public_date:
                raise KeyError(f"exact 模式输入缺少列: {cfg.public_year_col} / {cfg.public_date_col} ({path})")
            public_year = (
                pd.to_datetime(frame[cfg.public_date_col].astype("string").fillna("").str.strip(), errors="coerce")
                .dt.year.astype("Int64")
            )
            frame[cfg.public_year_col] = public_year
            logger.info("exact 输入缺少列 %s，已从 %s 回填: %s", cfg.public_year_col, cfg.public_date_col, os.path.basename(path))
        else:
            frame[cfg.public_year_col] = pd.to_numeric(frame[cfg.public_year_col], errors="coerce").astype("Int64")
            if has_public_date:
                missing_year = frame[cfg.public_year_col].isna()
                if missing_year.any():
                    fallback_year = (
                        pd.to_datetime(frame.loc[missing_year, cfg.public_date_col].astype("string").fillna("").str.strip(), errors="coerce")
                        .dt.year.astype("Int64")
                    )
                    frame.loc[missing_year, cfg.public_year_col] = fallback_year

        if cfg.public_date_ord_col not in frame.columns:
            if not has_public_date:
                raise KeyError(f"exact 模式输入缺少列: {cfg.public_date_ord_col} / {cfg.public_date_col} ({path})")
            frame[cfg.public_date_ord_col], invalid_count = _build_public_date_ord_from_series(frame[cfg.public_date_col])
            logger.info(
                "exact 输入缺少列 %s，已从 %s 回填: %s invalid=%s",
                cfg.public_date_ord_col,
                cfg.public_date_col,
                os.path.basename(path),
                invalid_count,
            )
        else:
            frame[cfg.public_date_ord_col] = pd.to_numeric(frame[cfg.public_date_ord_col], errors="coerce").astype("Int64")
            if has_public_date:
                missing_ord = frame[cfg.public_date_ord_col].isna() | (frame[cfg.public_date_ord_col] == invalid_ord)
                if missing_ord.any():
                    fallback_ord, invalid_count = _build_public_date_ord_from_series(frame.loc[missing_ord, cfg.public_date_col])
                    frame.loc[missing_ord, cfg.public_date_ord_col] = fallback_ord
                    logger.info(
                        "exact 输入补齐无效列 %s: %s repaired=%s invalid_after_parse=%s",
                        cfg.public_date_ord_col,
                        os.path.basename(path),
                        int(missing_ord.sum()),
                        invalid_count,
                    )

        frame[cfg.public_year_col] = pd.to_numeric(frame[cfg.public_year_col], errors="coerce").astype("Int64")
        frame[cfg.public_date_ord_col] = pd.to_numeric(frame[cfg.public_date_ord_col], errors="coerce").astype("Int64")
        before = len(frame)
        frame = frame[frame[cfg.public_year_col].notna()].copy()
        frame = frame[frame[cfg.public_date_ord_col].notna() & (frame[cfg.public_date_ord_col] != invalid_ord)].copy()
        skipped = before - len(frame)
        if skipped:
            logger.info("exact 输入过滤无效发布时间: file=%s skipped=%s kept=%s", os.path.basename(path), skipped, len(frame))
        yield path, frame


def _iter_exact_docs(
    cfg: Config,
    *,
    title_col: str,
) -> Iterable[Tuple[str, int, str, str, Dict[str, str]]]:
    cfg.ensure_dirs()
    seen_ids: Dict[str, bool] = {}
    columns = list(
        dict.fromkeys(
            [
                cfg.col_id,
                cfg.col_type,
                title_col,
                cfg.public_year_col,
                cfg.public_date_col,
                cfg.public_date_ord_col,
                *cfg.col_text_parts,
                *cfg.extra_cols,
            ]
        )
    )
    for path, frame in _iter_exact_frames(cfg, columns):
        positions = {column: idx for idx, column in enumerate(frame.columns)}
        for values in tqdm(frame.itertuples(index=False, name=None), total=len(frame), desc=f"exact:{os.path.basename(path)}"):
            pid = _tuple_text(values, positions, cfg.col_id)
            if not pid or pid in seen_ids:
                continue
            year_idx = positions[cfg.public_year_col]
            year = int(values[year_idx])
            title = _tuple_text(values, positions, title_col)
            text = _tuple_concat_text(values, positions, cfg.col_text_parts, cfg.text_sep)
            extra_data = {}
            for column in cfg.token_metadata_columns:
                extra_data[column] = _tuple_text(values, positions, column)
            for column in cfg.extra_cols:
                extra_data[column] = _tuple_text(values, positions, column)
            seen_ids[pid] = True
            yield pid, year, title, text, extra_data


def iter_clean_docs(cfg: Config, fallback_date_col: str = "申请日") -> Iterable[Tuple[str, int, str]]:
    if cfg.exact_date:
        for pid, year, _, text, _ in _iter_exact_docs(cfg, title_col="专利名称"):
            yield pid, year, text
        return

    cfg.ensure_dirs()
    usecols = cfg.usecols or list(set([cfg.col_id, cfg.col_date, cfg.col_type] + cfg.col_text_parts + [fallback_date_col]))
    files = _list_csv_files(cfg.data_path)
    seen_ids: Dict[str, bool] = {}
    for f in files:
        encodings = ([cfg.encoding] if cfg.encoding else ["utf-8", "gb18030"])
        read_ok = False
        for enc in encodings:
            try:
                for chunk in pd.read_csv(f, chunksize=cfg.chunksize, usecols=usecols, encoding=enc, low_memory=False, engine="c"):
                    chunk = chunk[chunk[cfg.col_type] == "发明授权"]
                    chunk = chunk.drop_duplicates(subset=[cfg.col_id], keep="first")
                    for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"filter:{os.path.basename(f)}"):
                        pid = str(row.get(cfg.col_id))
                        if pid in seen_ids:
                            continue
                        year = _parse_year(row, cfg.col_date, fallback_date_col)
                        if year is None:
                            continue
                        text = _concat_text(row, cfg.col_text_parts, cfg.text_sep)
                        seen_ids[pid] = True
                        yield pid, year, text
                read_ok = True
                break
            except Exception:
                continue
        if not read_ok:
            raise RuntimeError(f"failed to read {f}")


def iter_docs_with_title(
    cfg: Config,
    title_col: str = "专利名称",
    fallback_date_col: str = "申请日",
) -> Iterable[Tuple[str, int, str, str, Dict[str, str]]]:
    if cfg.exact_date:
        yield from _iter_exact_docs(cfg, title_col=title_col)
        return

    cfg.ensure_dirs()
    needed = set([cfg.col_id, cfg.col_date, cfg.col_type, title_col] + cfg.col_text_parts + [fallback_date_col])
    if cfg.extra_cols:
        needed.update(cfg.extra_cols)
    usecols = cfg.usecols or list(needed)
    files = _list_csv_files(cfg.data_path)
    seen_ids: Dict[str, bool] = {}
    for f in files:
        encodings = ([cfg.encoding] if cfg.encoding else ["utf-8", "gb18030"])
        read_ok = False
        for enc in encodings:
            try:
                for chunk in pd.read_csv(f, chunksize=cfg.chunksize, usecols=usecols, encoding=enc, low_memory=False, engine="c"):
                    chunk = chunk[chunk[cfg.col_type] == "发明授权"]
                    chunk = chunk.drop_duplicates(subset=[cfg.col_id], keep="first")
                    for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"title:{os.path.basename(f)}"):
                        pid = str(row.get(cfg.col_id))
                        if pid in seen_ids:
                            continue
                        year = _parse_year(row, cfg.col_date, fallback_date_col)
                        if year is None:
                            continue
                        title = row.get(title_col)
                        title = "" if not _is_present(title) else str(title)
                        text = _concat_text(row, cfg.col_text_parts, cfg.text_sep)
                        seen_ids[pid] = True
                        extra_data = {}
                        for c in cfg.extra_cols:
                            v = row.get(c)
                            extra_data[c] = "" if not _is_present(v) else str(v)
                        yield pid, year, title, text, extra_data
                read_ok = True
                break
            except Exception:
                continue
        if not read_ok:
            raise RuntimeError(f"failed to read {f}")
