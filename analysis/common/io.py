from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import csv
import json
import logging
import shutil

import pandas as pd

from .paths import repo_relative


READ_ENCODINGS: Sequence[str] = ("utf-8-sig", "utf-8", "gb18030")


def list_csv_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    return [path]


def read_csv_with_fallback(
    path: Path,
    *,
    dtype: Optional[Any] = None,
    usecols: Optional[Iterable[str]] = None,
    chunksize: Optional[int] = None,
    low_memory: bool = False,
    on_bad_lines: str = "error",
    engine: Optional[str] = None,
) -> Any:
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            read_kwargs: Dict[str, Any] = {
                "filepath_or_buffer": path,
                "dtype": dtype,
                "usecols": list(usecols) if usecols is not None else None,
                "chunksize": chunksize,
                "encoding": encoding,
                "on_bad_lines": on_bad_lines,
            }
            if engine is not None:
                read_kwargs["engine"] = engine
            if engine != "python":
                read_kwargs["low_memory"] = low_memory
            return pd.read_csv(**read_kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def read_csv_head(path: Path, rows: int = 5) -> List[List[str]]:
    for encoding in READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.reader(fh)
                result = []
                for idx, row in enumerate(reader):
                    result.append(row)
                    if idx + 1 >= rows:
                        break
                return result
        except Exception:
            continue
    raise RuntimeError(f"无法读取 CSV 头部: {path}")


def first_row_columns(path: Path) -> List[str]:
    rows = read_csv_head(path, rows=1)
    return rows[0] if rows else []


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def build_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    close_logger(logger)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
        except Exception:
            pass
        handler.close()


def normalize_string_series(series: Any) -> Any:
    return series.astype("string").fillna("").str.strip()


def path_metadata(source_path: Path) -> Dict[str, str]:
    return {
        "source": repo_relative(source_path),
        "resolved_source": str(source_path.resolve()),
    }
