from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Optional, TypedDict

import numpy as np
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from common.io import READ_ENCODINGS, build_logger, list_csv_files, write_json  # noqa: E402
from common.paths import build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


DEFAULT_RAW_PATENT_DIR = "data/raw/中国专利分年份保存数据1985-2025"
PATENT_TYPE_COL = "专利类型"
AUTHORIZED_PATENT_TYPE = "发明授权"
ID_COL = "申请号"
PUBLIC_DATE_COL = "公开公告日"
PUBLIC_YEAR_COL = "公开公告年份"
PUBLIC_DATE_ORD_COL = "公开公告日_ord"
INVALID_PUBLIC_DATE_ORD = np.iinfo(np.int32).max


class BuildRawPatentAuthorizedPartsResult(TypedDict):
    output_dir: Path
    metadata_path: Path


def _close_logger_handlers(logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _normalize_header(columns: list[str]) -> list[str]:
    normalized = list(columns)
    if normalized:
        normalized[0] = normalized[0].lstrip("\ufeff")
    return normalized


def _iter_csv_chunks(
    path: Path,
    *,
    chunksize: int,
    encoding: str,
    columns: list[str],
    stats: dict[str, object],
) -> Iterator[pd.DataFrame]:
    expected_width = len(columns)
    rows: list[list[str]] = []

    with path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)

        for row in reader:
            stats["rows_scanned"] = int(stats["rows_scanned"]) + 1
            if len(row) == expected_width:
                rows.append(row)
            elif len(row) == expected_width + 1 and row[-1] == "":
                rows.append(row[:-1])
                stats["rows_healed_trailing_empty_field"] = int(stats["rows_healed_trailing_empty_field"]) + 1
            else:
                stats["rows_skipped_bad_width"] = int(stats["rows_skipped_bad_width"]) + 1
                continue

            if len(rows) >= chunksize:
                chunk = pd.DataFrame(rows, columns=columns)
                stats["rows_emitted"] = int(stats["rows_emitted"]) + len(chunk)
                yield chunk
                rows = []

    if rows:
        chunk = pd.DataFrame(rows, columns=columns)
        stats["rows_emitted"] = int(stats["rows_emitted"]) + len(chunk)
        yield chunk


def _open_csv_reader(path: Path, *, chunksize: int):
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.reader(fh)
                header = _normalize_header(next(reader))
            if not header:
                raise RuntimeError(f"CSV 头为空: {path}")

            stats: dict[str, object] = {
                "encoding": encoding,
                "columns": header,
                "rows_scanned": 0,
                "rows_emitted": 0,
                "rows_healed_trailing_empty_field": 0,
                "rows_skipped_bad_width": 0,
            }
            return _iter_csv_chunks(
                path,
                chunksize=chunksize,
                encoding=encoding,
                columns=header,
                stats=stats,
            ), stats
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def _normalize_authorized_mask(chunk: pd.DataFrame) -> pd.Series:
    if PATENT_TYPE_COL not in chunk.columns:
        raise KeyError(f"原始专利文件缺少列: {PATENT_TYPE_COL}")
    patent_type = chunk[PATENT_TYPE_COL].astype("string").fillna("").str.strip()
    return patent_type == AUTHORIZED_PATENT_TYPE


def _table_from_chunk(chunk: pd.DataFrame, *, column_order: list[str]) -> pa.Table:
    ordered = chunk.reindex(columns=column_order).copy()
    for column in column_order:
        if column == PUBLIC_DATE_ORD_COL:
            ordered[column] = pd.to_numeric(ordered[column], errors="coerce").fillna(INVALID_PUBLIC_DATE_ORD).astype("int32")
        else:
            ordered[column] = ordered[column].astype("string")
    return pa.Table.from_pandas(ordered, preserve_index=False)


def _attach_table_metadata(table: pa.Table, *, invalid_publish_date_rows: int) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"sort_by": json.dumps([PUBLIC_DATE_ORD_COL, ID_COL], ensure_ascii=False).encode("utf-8"),
            b"date_col": PUBLIC_DATE_COL.encode("utf-8"),
            b"year_col": PUBLIC_YEAR_COL.encode("utf-8"),
            b"invalid_publish_date_rows": str(int(invalid_publish_date_rows)).encode("utf-8"),
        }
    )
    return table.replace_schema_metadata(metadata)


def _build_public_date_ord(series: pd.Series) -> tuple[pd.Series, int]:
    text = series.astype("string").fillna("").str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    parsed_days = parsed.to_numpy(dtype="datetime64[D]")
    valid_mask = parsed.notna().to_numpy()
    ord_values = np.full(len(series), INVALID_PUBLIC_DATE_ORD, dtype=np.int32)
    if valid_mask.any():
        ord_values[valid_mask] = parsed_days[valid_mask].astype(np.int32, copy=False)
    return pd.Series(ord_values, index=series.index, dtype="int32"), int((~valid_mask).sum())


def _ensure_public_year(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    if PUBLIC_YEAR_COL not in chunk.columns:
        chunk[PUBLIC_YEAR_COL] = pd.Series([""] * len(chunk), index=chunk.index, dtype="string")

    public_year = chunk[PUBLIC_YEAR_COL].astype("string").fillna("").str.strip()
    needs_fill = public_year.eq("")
    if needs_fill.any() and PUBLIC_DATE_COL in chunk.columns:
        parsed = pd.to_datetime(chunk.loc[needs_fill, PUBLIC_DATE_COL].astype("string").fillna("").str.strip(), errors="coerce")
        filled = parsed.dt.year.astype("Int64").astype("string").fillna("")
        public_year = public_year.copy()
        public_year.loc[needs_fill] = filled
    chunk[PUBLIC_YEAR_COL] = public_year
    return chunk


def build_raw_patent_authorized_parts(
    *,
    raw_patent_dir: Path,
    shared_root: str = "outputs/shared",
    chunksize: int = 100000,
    compression: str = "zstd",
    overwrite: bool = False,
) -> BuildRawPatentAuthorizedPartsResult:
    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()
    output_dir = shared_paths.raw_patent_authorized_parts_dir
    metadata_path = output_dir / "metadata.json"
    logger = build_logger(
        "build_raw_patent_authorized_parts",
        shared_paths.logs_dir / "build_raw_patent_authorized_parts.log",
    )
    try:
        csv_paths = list_csv_files(raw_patent_dir)
        logger.info(
            "开始构造发明授权 parquet parts，共 %s 个 CSV 文件，输出目录=%s",
            len(csv_paths),
            repo_relative(output_dir),
        )
        logger.info("CSV 读取启用容错模式：仅当坏行为“末尾多 1 个空字段”时自动裁剪，其余宽度异常仍跳过")

        parts_summary: list[dict[str, object]] = []
        total_rows_scanned = 0
        total_rows_read = 0
        total_rows_written = 0
        total_rows_healed = 0
        total_rows_skipped_bad_width = 0
        total_invalid_public_date_rows = 0

        for index, csv_path in enumerate(csv_paths, start=1):
            output_path = output_dir / f"{csv_path.stem}.parquet"
            if output_path.exists() and not overwrite:
                logger.info("跳过已有 parquet part [%s/%s]: %s", index, len(csv_paths), repo_relative(output_path))
                parts_summary.append(
                    {
                        "source_csv": repo_relative(csv_path),
                        "output_parquet": repo_relative(output_path),
                        "skipped": True,
                    }
                )
                continue
            if output_path.exists():
                output_path.unlink()

            logger.info("转换原始专利文件 [%s/%s]: %s", index, len(csv_paths), repo_relative(csv_path))
            reader, read_stats = _open_csv_reader(csv_path, chunksize=chunksize)
            column_order: list[str] = list(read_stats["columns"])
            if PUBLIC_YEAR_COL not in column_order:
                column_order.append(PUBLIC_YEAR_COL)
            if PUBLIC_DATE_ORD_COL not in column_order:
                column_order.append(PUBLIC_DATE_ORD_COL)
            rows_read = 0
            rows_written = 0
            invalid_public_date_rows = 0
            authorized_rows = 0
            authorized_chunks: list[pd.DataFrame] = []

            for chunk_index, chunk in enumerate(reader, start=1):
                rows_read += len(chunk)
                mask = _normalize_authorized_mask(chunk)
                chunk = chunk.loc[mask].copy()
                if chunk.empty:
                    continue
                if ID_COL in chunk.columns:
                    chunk[ID_COL] = chunk[ID_COL].astype("string").fillna("").str.strip()
                chunk = _ensure_public_year(chunk)
                chunk[PUBLIC_DATE_ORD_COL], invalid_count = _build_public_date_ord(
                    chunk[PUBLIC_DATE_COL] if PUBLIC_DATE_COL in chunk.columns else pd.Series([""] * len(chunk), index=chunk.index)
                )
                invalid_public_date_rows += invalid_count
                authorized_chunks.append(chunk.reindex(columns=column_order))
                authorized_rows += len(chunk)

                if chunk_index == 1 or chunk_index % 10 == 0:
                    logger.info(
                        "parquet 转换进度 [%s/%s]: %s, chunk=%s, 已读行数=%s, 已保留授权行=%s",
                        index,
                        len(csv_paths),
                        repo_relative(csv_path),
                        chunk_index,
                        rows_read,
                        authorized_rows,
                    )

            if authorized_chunks:
                authorized_df = pd.concat(authorized_chunks, ignore_index=True)
                sort_columns = [PUBLIC_DATE_ORD_COL]
                if ID_COL in authorized_df.columns:
                    sort_columns.append(ID_COL)
                authorized_df = authorized_df.sort_values(sort_columns, ascending=True, kind="mergesort").reset_index(drop=True)
                table = _attach_table_metadata(
                    _table_from_chunk(authorized_df, column_order=column_order),
                    invalid_publish_date_rows=invalid_public_date_rows,
                )
                pq.write_table(table, output_path, compression=compression)
                rows_written = int(table.num_rows)
            else:
                empty_df = pd.DataFrame(
                    {
                        column: pd.Series(dtype="int32" if column == PUBLIC_DATE_ORD_COL else "string")
                        for column in column_order
                    }
                )
                empty_table = _attach_table_metadata(pa.Table.from_pandas(empty_df, preserve_index=False), invalid_publish_date_rows=0)
                pq.write_table(empty_table, output_path, compression=compression)

            rows_scanned = int(read_stats["rows_scanned"])
            rows_healed = int(read_stats["rows_healed_trailing_empty_field"])
            rows_skipped_bad_width = int(read_stats["rows_skipped_bad_width"])
            total_rows_scanned += rows_scanned
            total_rows_read += rows_read
            total_rows_written += rows_written
            total_rows_healed += rows_healed
            total_rows_skipped_bad_width += rows_skipped_bad_width
            total_invalid_public_date_rows += invalid_public_date_rows
            logger.info(
                "parquet 转换完成 [%s/%s]: %s -> %s，扫描行数=%s，可用行数=%s，发明授权行数=%s，发布日期无效=%s，尾空字段修复=%s，异常宽度跳过=%s",
                index,
                len(csv_paths),
                repo_relative(csv_path),
                repo_relative(output_path),
                rows_scanned,
                rows_read,
                rows_written,
                invalid_public_date_rows,
                rows_healed,
                rows_skipped_bad_width,
            )
            parts_summary.append(
                {
                    "source_csv": repo_relative(csv_path),
                    "output_parquet": repo_relative(output_path),
                    "encoding": read_stats["encoding"],
                    "rows_scanned": rows_scanned,
                    "rows_read": int(rows_read),
                    "rows_written": int(rows_written),
                    "invalid_publish_date_rows": int(invalid_public_date_rows),
                    "rows_healed_trailing_empty_field": rows_healed,
                    "rows_skipped_bad_width": rows_skipped_bad_width,
                    "sort_by": [PUBLIC_DATE_ORD_COL, ID_COL],
                    "date_col": PUBLIC_DATE_COL,
                    "year_col": PUBLIC_YEAR_COL,
                    "skipped": False,
                }
            )

        summary = {
            "raw_patent_dir": repo_relative(raw_patent_dir),
            "output_dir": repo_relative(output_dir),
            "filter_patent_type": AUTHORIZED_PATENT_TYPE,
            "chunksize": int(chunksize),
            "compression": compression,
            "files_total": len(csv_paths),
            "rows_scanned_total": int(total_rows_scanned),
            "rows_read_total": int(total_rows_read),
            "rows_written_total": int(total_rows_written),
            "rows_healed_trailing_empty_field_total": int(total_rows_healed),
            "rows_skipped_bad_width_total": int(total_rows_skipped_bad_width),
            "invalid_publish_date_rows_total": int(total_invalid_public_date_rows),
            "sort_by": [PUBLIC_DATE_ORD_COL, ID_COL],
            "date_col": PUBLIC_DATE_COL,
            "year_col": PUBLIC_YEAR_COL,
            "parts": parts_summary,
        }
        write_json(metadata_path, summary)
        logger.info(
            "发明授权 parquet parts 构造完成，文件数=%s，扫描总行数=%s，可用总行数=%s，输出总行数=%s，尾空字段修复=%s，异常宽度跳过=%s",
            len(csv_paths),
            total_rows_scanned,
            total_rows_read,
            total_rows_written,
            total_rows_healed,
            total_rows_skipped_bad_width,
        )
        return {
            "output_dir": output_dir,
            "metadata_path": metadata_path,
        }
    finally:
        _close_logger_handlers(logger)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="将原始专利 CSV 按年转换为仅包含发明授权的 parquet parts")
    parser.add_argument("--raw-patent-dir", default=DEFAULT_RAW_PATENT_DIR, help="原始专利 CSV 目录")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--chunksize", type=int, default=100000, help="CSV 转换分块行数")
    parser.add_argument("--compression", default="zstd", help="parquet 压缩算法")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 parquet parts")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    raw_patent_dir = resolve_repo_path(args.raw_patent_dir)
    assert raw_patent_dir is not None
    build_raw_patent_authorized_parts(
        raw_patent_dir=raw_patent_dir,
        shared_root=args.shared_root,
        chunksize=args.chunksize,
        compression=args.compression,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
