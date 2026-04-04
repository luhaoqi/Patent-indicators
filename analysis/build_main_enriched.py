from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any, Dict, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common.io import build_logger, copy_if_needed, list_csv_files, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


ID_COL = "申请号"
PATENT_TYPE_COL = "专利类型"
AUTHORIZED_PATENT_TYPE = "发明授权"
CITATION_COLS = [
    "引证次数",
    "被引证次数",
    "自引次数",
    "他引次数",
    "被自引次数",
    "被他引次数",
    "家族引证次数",
    "家族被引证次数",
]
PATENT_MASTER_COLUMNS = [
    ID_COL,
    "申请年份",
    PATENT_TYPE_COL,
    "公开公告年份",
    "IPC主分类号",
    "专利权人类型",
    "统一社会信用代码",
    *CITATION_COLS,
]


def _scan_csv_utf8_lossy(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        str(path),
        encoding="utf8-lossy",
        infer_schema_length=0,
        ignore_errors=True,
    )


def _normalized_utf8_expr(name: str) -> pl.Expr:
    return (
        pl.col(name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .alias(name)
    )


def _build_part_exprs(schema_names: set[str]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for column in PATENT_MASTER_COLUMNS:
        if column in schema_names:
            exprs.append(_normalized_utf8_expr(column))
        else:
            exprs.append(pl.lit(None, dtype=pl.Utf8).alias(column))
    return exprs


def _extract_patent_parts(
    *,
    raw_patent_dir: Path,
    parts_dir: Path,
    logger,
) -> list[Path]:
    csv_paths = list_csv_files(raw_patent_dir)
    logger.info("开始构造 patent_master，共 %s 个 CSV 文件", len(csv_paths))
    part_paths: list[Path] = []

    for file_index, csv_path in enumerate(csv_paths, start=1):
        part_path = parts_dir / f"{csv_path.stem}.parquet"
        part_paths.append(part_path)
        if part_path.exists():
            logger.info("跳过已有 parquet part [%s/%s]: %s", file_index, len(csv_paths), repo_relative(part_path))
            continue

        logger.info("抽取原始专利文件 [%s/%s]: %s", file_index, len(csv_paths), repo_relative(csv_path))
        raw_lf = _scan_csv_utf8_lossy(csv_path)
        schema_names = set(raw_lf.collect_schema().names())
        part_df = (
            raw_lf
            .select(_build_part_exprs(schema_names))
            .filter(pl.col(ID_COL).is_not_null() & (pl.col(ID_COL) != ""))
            .collect(engine="streaming")
        )
        part_df.write_parquet(part_path)
        logger.info(
            "part 写出完成 [%s/%s]: %s，rows=%s",
            file_index,
            len(csv_paths),
            repo_relative(part_path),
            part_df.height,
        )

    return part_paths


def _citation_expr(column: str) -> pl.Expr:
    base = (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .replace("", None)
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
    )
    preferred = base.filter(pl.col("__authorized")).drop_nulls().max()
    fallback = base.drop_nulls().max()
    return (
        pl.when(pl.col("__authorized").sum() > 0)
        .then(preferred)
        .otherwise(fallback)
        .alias(column)
    )


def _non_citation_expr(column: str) -> pl.Expr:
    base = (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .replace("", None)
    )
    preferred = base.filter(pl.col("__authorized")).drop_nulls().first()
    fallback = base.drop_nulls().first()
    return (
        pl.when(pl.col("__authorized").sum() > 0)
        .then(preferred)
        .otherwise(fallback)
        .alias(column)
    )


def _deduplicate_parts(
    *,
    part_paths: list[Path],
    patent_master_path: Path,
    logger,
) -> int:
    if not part_paths:
        empty = pl.DataFrame(schema={column: pl.Utf8 for column in PATENT_MASTER_COLUMNS})
        empty.write_parquet(patent_master_path)
        return 0

    parts_glob = str(part_paths[0].parent / "*.parquet")
    extra_all_lf = (
        pl.scan_parquet(parts_glob)
        .select([pl.col(column) for column in PATENT_MASTER_COLUMNS])
        .with_columns(
            _normalized_utf8_expr(ID_COL),
            _normalized_utf8_expr(PATENT_TYPE_COL),
            pl.col(PATENT_TYPE_COL).cast(pl.Utf8, strict=False).str.strip_chars().eq(AUTHORIZED_PATENT_TYPE).alias("__authorized"),
        )
        .filter(pl.col(ID_COL).is_not_null() & (pl.col(ID_COL) != ""))
    )
    rows_total = extra_all_lf.select(pl.len()).collect().item()
    logger.info("patent_master 原始行数: %s", rows_total)

    agg_exprs: list[pl.Expr] = []
    for column in PATENT_MASTER_COLUMNS:
        if column == ID_COL:
            continue
        if column in CITATION_COLS:
            agg_exprs.append(_citation_expr(column))
        else:
            agg_exprs.append(_non_citation_expr(column))

    patent_master_df = (
        extra_all_lf
        .group_by(ID_COL)
        .agg(agg_exprs)
        .with_columns(
            pl.col("申请年份").cast(pl.Int32, strict=False),
            pl.col("公开公告年份").cast(pl.Int32, strict=False),
        )
        .collect(engine="streaming")
    )
    patent_master_df.write_parquet(patent_master_path)
    logger.info("patent_master 去重后行数: %s", patent_master_df.height)
    return int(rows_total)


def build_patent_master(
    *,
    raw_patent_dir: Path,
    shared_root: str = "outputs/shared",
    chunksize: int = 100000,
) -> Dict[str, Path]:
    del chunksize

    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()
    logger = build_logger("build_patent_master", shared_paths.logs_dir / "build_patent_master.log")

    parts_dir = shared_paths.patent_master_dir / "extra_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    patent_master_path = shared_paths.patent_master_dir / "patent_master.parquet"
    metadata_path = shared_paths.patent_master_dir / "metadata.json"

    part_paths = _extract_patent_parts(
        raw_patent_dir=raw_patent_dir,
        parts_dir=parts_dir,
        logger=logger,
    )
    rows_total = _deduplicate_parts(
        part_paths=part_paths,
        patent_master_path=patent_master_path,
        logger=logger,
    )

    columns = pl.read_parquet(patent_master_path, n_rows=0).columns
    row_count = pl.scan_parquet(str(patent_master_path)).select(pl.len()).collect().item()
    write_json(
        metadata_path,
        {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "inputs": {"raw_patent_dir": repo_relative(raw_patent_dir)},
            "outputs": {
                "patent_master": repo_relative(patent_master_path),
                "extra_parts_dir": repo_relative(parts_dir),
            },
            "rows": int(row_count),
            "raw_rows": int(rows_total),
            "columns": columns,
            "key_fields": [ID_COL],
            "implementation": "polars_parts_plus_streaming_groupby",
        },
    )
    logger.info("patent_master 输出: %s", repo_relative(patent_master_path))
    return {
        "patent_master_path": patent_master_path,
        "metadata_path": metadata_path,
    }


def _read_stage1_main(stage1_output_path: Path) -> pl.DataFrame:
    main_df = pl.read_csv(
        str(stage1_output_path),
        encoding="utf8-lossy",
        ignore_errors=True,
    )
    if ID_COL not in main_df.columns:
        raise KeyError(f"stage1 输出缺少列: {ID_COL}")
    return main_df.with_columns(_normalized_utf8_expr(ID_COL))


def build_experiment_patent_panel(
    *,
    experiment_id: str,
    stage1_output_path: Path,
    output_root: str = "outputs/experiments",
    patent_master_path: Path,
    shared_root: str = "outputs/shared",
    exact_date: bool = False,
) -> Dict[str, Path]:
    del shared_root

    paths = build_experiment_paths(experiment_id, output_root=output_root, exact_date=exact_date)
    paths.ensure_dirs()
    logger = build_logger(
        f"build_experiment_patent_panel.{experiment_id}",
        paths.logs_dir / "build_experiment_patent_panel.log",
    )

    if not patent_master_path.exists():
        raise FileNotFoundError(f"找不到 patent_master: {patent_master_path}")

    logger.info("读取 stage1 主结果: %s", repo_relative(stage1_output_path))
    main_df = _read_stage1_main(stage1_output_path)
    logger.info("读取 patent_master: %s", repo_relative(patent_master_path))

    stage1_copy_path = paths.data_dir / "patent_quality_output.csv"
    main_path = paths.data_dir / "main.parquet"
    panel_path = paths.data_dir / "experiment_patent_panel.parquet"
    copy_if_needed(stage1_output_path, stage1_copy_path)
    main_df.write_parquet(main_path)

    logger.info("开始按 notebook 路径构造 experiment_patent_panel")
    main_cols = pl.read_parquet(main_path, n_rows=0).columns
    extra_cols = [
        column
        for column in pl.read_parquet(patent_master_path, n_rows=0).columns
        if column == ID_COL or column not in main_cols
    ]

    experiment_patent_panel = (
        pl.scan_parquet(str(main_path))
        .with_columns(_normalized_utf8_expr(ID_COL))
        .join(
            pl.scan_parquet(str(patent_master_path))
            .with_columns(_normalized_utf8_expr(ID_COL))
            .select(extra_cols),
            on=ID_COL,
            how="left",
        )
        .collect(engine="streaming")
    )
    experiment_patent_panel.write_parquet(panel_path)

    metadata = {
        "experiment_id": experiment_id,
        "stage1_output": repo_relative(stage1_output_path),
        "patent_master_path": repo_relative(patent_master_path),
        "main_rows": int(main_df.height),
        "experiment_patent_panel_rows": int(experiment_patent_panel.height),
        "outputs": {
            "stage1_copy": repo_relative(stage1_copy_path),
            "main": repo_relative(main_path),
            "experiment_patent_panel": repo_relative(panel_path),
        },
        "implementation": "polars_parquet_join",
    }
    write_json(paths.metadata_dir / "build_experiment_patent_panel.json", metadata)
    logger.info("experiment_patent_panel 输出: %s", repo_relative(panel_path))
    return {
        "main_path": main_path,
        "experiment_patent_panel_path": panel_path,
    }


def build_main_enriched(
    *,
    experiment_id: str,
    stage1_output_path: Path,
    patent_master_path: Path,
    output_root: str = "outputs/experiments",
    shared_root: str = "outputs/shared",
    exact_date: bool = False,
):
    result = build_experiment_patent_panel(
        experiment_id=experiment_id,
        stage1_output_path=stage1_output_path,
        output_root=output_root,
        patent_master_path=patent_master_path,
        shared_root=shared_root,
        exact_date=exact_date,
    )
    metadata_path = build_experiment_paths(experiment_id, output_root=output_root, exact_date=exact_date).metadata_dir / "build_main_enriched.json"
    write_json(
        metadata_path,
        {
            "experiment_id": experiment_id,
            "stage1_output": repo_relative(stage1_output_path),
            "patent_master_path": repo_relative(patent_master_path),
            "experiment_patent_panel_path": repo_relative(result["experiment_patent_panel_path"]),
            "exact_date": bool(exact_date),
        },
    )
    return result


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="从第一阶段结果和共享 patent_master 构造 experiment_patent_panel.parquet")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--stage1-dir", help="第一阶段 stage1 输出目录")
    parser.add_argument("--stage1-output", help="第一阶段 patent_quality_output.csv 路径")
    parser.add_argument("--patent-master-path", help="共享 patent_master.parquet 路径，不传则从 shared_root 推断")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    return parser


def _resolve_stage1_output(stage1_dir: Optional[str], stage1_output: Optional[str]) -> Path:
    if stage1_output:
        resolved = resolve_repo_path(stage1_output)
        assert resolved is not None
        return resolved
    if stage1_dir:
        resolved = resolve_repo_path(stage1_dir)
        assert resolved is not None
        return resolved / "patent_quality_output.csv"
    raise ValueError("必须至少提供 --stage1-dir 或 --stage1-output")


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    patent_master_path = resolve_repo_path(args.patent_master_path) if args.patent_master_path else None
    if patent_master_path is None:
        patent_master_path = build_shared_paths(args.shared_root).patent_master_dir / "patent_master.parquet"
    build_main_enriched(
        experiment_id=args.experiment_id,
        stage1_output_path=_resolve_stage1_output(args.stage1_dir, args.stage1_output),
        patent_master_path=patent_master_path,
        output_root=args.output_root,
        shared_root=args.shared_root,
    )


if __name__ == "__main__":
    main()
