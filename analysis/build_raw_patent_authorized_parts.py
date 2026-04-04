from __future__ import annotations

from argparse import ArgumentParser
import csv
from pathlib import Path
import sys
from typing import Any, Iterator, Optional, TypedDict

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
        ordered[column] = ordered[column].astype("string")
    return pa.Table.from_pandas(ordered, preserve_index=False)


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
            writer: Optional[pq.ParquetWriter] = None
            column_order: list[str] = list(read_stats["columns"])
            rows_read = 0
            rows_written = 0

            try:
                for chunk_index, chunk in enumerate(reader, start=1):
                    rows_read += len(chunk)
                    mask = _normalize_authorized_mask(chunk)
                    chunk = chunk.loc[mask].copy()
                    if chunk.empty:
                        continue

                    if ID_COL in chunk.columns:
                        chunk[ID_COL] = chunk[ID_COL].astype("string").fillna("").str.strip()

                    table = _table_from_chunk(chunk, column_order=column_order)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                    else:
                        table = table.cast(writer.schema, safe=False)
                    writer.write_table(table)
                    rows_written += table.num_rows

                    if chunk_index == 1 or chunk_index % 10 == 0:
                        logger.info(
                            "parquet 转换进度 [%s/%s]: %s, chunk=%s, 已读行数=%s, 已写入发明授权=%s",
                            index,
                            len(csv_paths),
                            repo_relative(csv_path),
                            chunk_index,
                            rows_read,
                            rows_written,
                        )
            finally:
                if writer is not None:
                    writer.close()

            if writer is None:
                empty_df = pd.DataFrame({column: pd.Series(dtype="string") for column in column_order})
                empty_table = pa.Table.from_pandas(empty_df, preserve_index=False)
                pq.write_table(empty_table, output_path, compression=compression)

            rows_scanned = int(read_stats["rows_scanned"])
            rows_healed = int(read_stats["rows_healed_trailing_empty_field"])
            rows_skipped_bad_width = int(read_stats["rows_skipped_bad_width"])
            total_rows_scanned += rows_scanned
            total_rows_read += rows_read
            total_rows_written += rows_written
            total_rows_healed += rows_healed
            total_rows_skipped_bad_width += rows_skipped_bad_width
            logger.info(
                "parquet 转换完成 [%s/%s]: %s -> %s，扫描行数=%s，可用行数=%s，发明授权行数=%s，尾空字段修复=%s，异常宽度跳过=%s",
                index,
                len(csv_paths),
                repo_relative(csv_path),
                repo_relative(output_path),
                rows_scanned,
                rows_read,
                rows_written,
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
                    "rows_healed_trailing_empty_field": rows_healed,
                    "rows_skipped_bad_width": rows_skipped_bad_width,
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
