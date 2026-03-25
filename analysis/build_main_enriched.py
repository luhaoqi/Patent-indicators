from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, cast

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd  # noqa: E402

from common.io import build_logger, copy_if_needed, list_csv_files, normalize_string_series, read_csv_with_fallback, write_json  # noqa: E402
from common.paths import build_experiment_paths, repo_relative, resolve_repo_path  # noqa: E402


ID_COL = "申请号"
PATENT_TYPE_COL = "专利类型"
AUTHORIZED_PATENT_TYPE = "发明授权"


def _non_empty(series: Any) -> Any:
    cleaned = normalize_string_series(series)
    return cleaned[cleaned != ""]


def _deduplicate_extra(extra_df: Any) -> Any:
    if extra_df.empty:
        return extra_df

    extra_df = extra_df.copy()
    for column in extra_df.columns:
        if extra_df[column].dtype == object or str(extra_df[column].dtype).startswith("string"):
            extra_df[column] = normalize_string_series(extra_df[column])

    def aggregate_group(group: Any) -> Any:
        result: Dict[str, object] = {ID_COL: group.name}
        for column in group.columns:
            if column == ID_COL:
                continue
            non_empty = _non_empty(cast(Any, group[column]))
            if non_empty.empty:
                result[column] = pd.NA
                continue
            numeric = cast(Any, pd.to_numeric(non_empty, errors="coerce"))
            if cast(bool, numeric.notna().all()):
                result[column] = numeric.max()
            else:
                result[column] = non_empty.iloc[0]
        return pd.Series(result)

    preferred = extra_df
    if PATENT_TYPE_COL in extra_df.columns:
        preferred = extra_df[extra_df[PATENT_TYPE_COL] == AUTHORIZED_PATENT_TYPE].copy()
        if preferred.empty:
            preferred = extra_df

    dedup = preferred.groupby(ID_COL, as_index=False, dropna=False).apply(aggregate_group)
    if isinstance(dedup.index, pd.MultiIndex):
        dedup = dedup.reset_index(drop=True)
    return dedup


def _fill_from_raw_columns(main_df: Any, extra_df: Any) -> Any:
    merged = main_df.merge(extra_df, on=ID_COL, how="left", suffixes=("", "__raw"))
    for column in list(merged.columns):
        if not column.endswith("__raw"):
            continue
        base_column = column[:-5]
        if base_column not in merged.columns:
            merged.rename(columns={column: base_column}, inplace=True)
            continue
        left = normalize_string_series(merged[base_column])
        merged[base_column] = merged[base_column].where(left != "", merged[column])
        merged.drop(columns=[column], inplace=True)
    return merged


def _load_main_output(stage1_output_path: Path) -> Any:
    main_df = read_csv_with_fallback(stage1_output_path, dtype={ID_COL: "string"})
    main_df[ID_COL] = normalize_string_series(main_df[ID_COL])
    return main_df


def _collect_extra_rows(
    raw_patent_dir: Path,
    target_ids: set[str],
    *,
    chunksize: int,
    logger,
) -> Any:
    frames: List[Any] = []
    csv_paths = list_csv_files(raw_patent_dir)
    logger.info("开始回捞原始专利，共 %s 个 CSV 文件，目标申请号数=%s", len(csv_paths), len(target_ids))
    for file_index, csv_path in enumerate(csv_paths, start=1):
        logger.info("扫描原始专利文件 [%s/%s]: %s", file_index, len(csv_paths), repo_relative(csv_path))
        reader = read_csv_with_fallback(
            csv_path,
            dtype={ID_COL: "string"},
            chunksize=chunksize,
            low_memory=False,
        )
        file_rows = 0
        matched_rows = 0
        for chunk_index, chunk in enumerate(reader, start=1):
            chunk_df = chunk
            file_rows += len(chunk_df)
            ids = normalize_string_series(chunk_df[ID_COL])
            matched = chunk_df.loc[ids.isin(target_ids)].copy()
            if matched.empty:
                if chunk_index % 20 == 0:
                    logger.info(
                        "文件 [%s/%s] chunk=%s 已扫描 %s 行，当前匹配 %s 行",
                        file_index,
                        len(csv_paths),
                        chunk_index,
                        file_rows,
                        matched_rows,
                    )
                continue
            matched[ID_COL] = normalize_string_series(matched[ID_COL])
            frames.append(matched)
            matched_rows += len(matched)
            logger.info(
                "文件 [%s/%s] chunk=%s 命中 %s 行，累计扫描 %s 行，累计匹配 %s 行",
                file_index,
                len(csv_paths),
                chunk_index,
                len(matched),
                file_rows,
                matched_rows,
            )
        logger.info(
            "原始专利文件 [%s/%s] 扫描完成: %s，扫描 %s 行，匹配 %s 行",
            file_index,
            len(csv_paths),
            repo_relative(csv_path),
            file_rows,
            matched_rows,
        )
    if not frames:
        return pd.DataFrame(columns=[ID_COL])
    return pd.concat(frames, ignore_index=True)


def build_main_enriched(
    *,
    experiment_id: str,
    stage1_output_path: Path,
    raw_patent_dir: Path,
    output_root: str = "outputs/experiments",
    chunksize: int = 100000,
):
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(f"build_main_enriched.{experiment_id}", paths.logs_dir / "build_main_enriched.log")

    logger.info("读取 stage1 主结果: %s", repo_relative(stage1_output_path))
    main_df = _load_main_output(stage1_output_path)
    logger.info("主结果行数: %s", len(main_df))
    copy_if_needed(stage1_output_path, paths.data_dir / "patent_quality_output.csv")

    target_ids = set(main_df[ID_COL].tolist())
    extra_all = _collect_extra_rows(raw_patent_dir, target_ids, chunksize=chunksize, logger=logger)
    logger.info("回捞原始专利行数: %s", len(extra_all))
    logger.info("开始按申请号去重")
    extra_dedup = _deduplicate_extra(extra_all)
    logger.info("按申请号去重后行数: %s", len(extra_dedup))

    logger.info("开始将原始专利字段回填到主结果")
    main_enriched = _fill_from_raw_columns(main_df, extra_dedup)
    logger.info("回填完成，main_enriched 行数: %s", len(main_enriched))

    main_path = paths.data_dir / "main.parquet"
    extra_path = paths.data_dir / "extra_all_dedup.parquet"
    enriched_path = paths.data_dir / "main_enriched.parquet"

    main_df.to_parquet(main_path, index=False)
    extra_dedup.to_parquet(extra_path, index=False)
    main_enriched.to_parquet(enriched_path, index=False)
    logger.info("已写出 main/main_extra/main_enriched 三个 parquet 文件")

    metadata = {
        "experiment_id": experiment_id,
        "stage1_output": repo_relative(stage1_output_path),
        "raw_patent_dir": repo_relative(raw_patent_dir),
        "main_rows": int(len(main_df)),
        "extra_rows": int(len(extra_all)),
        "extra_dedup_rows": int(len(extra_dedup)),
        "main_enriched_rows": int(len(main_enriched)),
    }
    write_json(paths.metadata_dir / "build_main_enriched.json", metadata)
    logger.info("main_enriched 输出: %s", repo_relative(enriched_path))
    return {
        "paths": paths,
        "main_path": main_path,
        "extra_path": extra_path,
        "enriched_path": enriched_path,
    }


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="从第一阶段结果和原始专利表构造 main_enriched.parquet")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--stage1-dir", help="第一阶段 stage1 输出目录")
    parser.add_argument("--stage1-output", help="第一阶段 patent_quality_output.csv 路径")
    parser.add_argument("--raw-patent-dir", default="data/raw/中国专利分年份保存数据1985-2025", help="原始专利 CSV 目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--chunksize", type=int, default=100000, help="按块读取原始专利 CSV 的行数")
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
    raw_patent_dir = resolve_repo_path(args.raw_patent_dir)
    assert raw_patent_dir is not None
    build_main_enriched(
        experiment_id=args.experiment_id,
        stage1_output_path=_resolve_stage1_output(args.stage1_dir, args.stage1_output),
        raw_patent_dir=raw_patent_dir,
        output_root=args.output_root,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
