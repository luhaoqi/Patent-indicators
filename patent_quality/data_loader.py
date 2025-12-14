import os
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from .config import Config


def _list_data_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isfile(p) and p.lower().endswith(".csv"):
                files.append(p)
        files.sort()
        return files
    return [path]


def _parse_year(row: pd.Series, col_year: str, fallback_date_col: str) -> Optional[int]:
    y = row.get(col_year)
    if pd.notna(y):
        try:
            return int(str(y)[:4])
        except Exception:
            pass
    d = row.get(fallback_date_col)
    if pd.notna(d):
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


def _concat_text(row: pd.Series, parts: List[str]) -> str:
    vals = []
    for c in parts:
        v = row.get(c)
        vals.append("" if pd.isna(v) else str(v))
    return "".join(vals)


def iter_clean_docs(cfg: Config, fallback_date_col: str = "申请日") -> Iterable[Tuple[str, int, str]]:
    cfg.ensure_dirs()
    usecols = cfg.usecols or list(set([cfg.col_id, cfg.col_date, cfg.col_type] + cfg.col_text_parts + [fallback_date_col]))
    files = _list_data_files(cfg.data_path)
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
                        text = _concat_text(row, cfg.col_text_parts)
                        seen_ids[pid] = True
                        yield pid, year, text
                read_ok = True
                break
            except Exception:
                continue
        if not read_ok:
            raise RuntimeError(f"failed to read {f}")


def iter_docs_with_title(cfg: Config, title_col: str = "专利名称", fallback_date_col: str = "申请日") -> Iterable[Tuple[str, int, str, str, Dict[str, str]]]:
    cfg.ensure_dirs()
    needed = set([cfg.col_id, cfg.col_date, cfg.col_type, title_col] + cfg.col_text_parts + [fallback_date_col])
    if cfg.extra_cols:
        needed.update(cfg.extra_cols)
    usecols = cfg.usecols or list(needed)
    files = _list_data_files(cfg.data_path)
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
                        title = "" if pd.isna(title) else str(title)
                        text = _concat_text(row, cfg.col_text_parts)
                        seen_ids[pid] = True
                        extra_data = {}
                        for c in cfg.extra_cols:
                            v = row.get(c)
                            extra_data[c] = "" if pd.isna(v) else str(v)
                        yield pid, year, title, text, extra_data
                read_ok = True
                break
            except Exception:
                continue
        if not read_ok:
            raise RuntimeError(f"failed to read {f}")
