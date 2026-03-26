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
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


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
        effective_group = group
        if PATENT_TYPE_COL in group.columns:
            authorized_group = group[group[PATENT_TYPE_COL] == AUTHORIZED_PATENT_TYPE]
            if not authorized_group.empty:
                effective_group = authorized_group
        result: Dict[str, object] = {ID_COL: group.name}
        for column in effective_group.columns:
            if column == ID_COL:
                continue
            non_empty = _non_empty(cast(Any, effective_group[column]))
            if non_empty.empty:
                result[column] = pd.NA
                continue
            numeric = cast(Any, pd.to_numeric(non_empty, errors="coerce"))
            if cast(bool, numeric.notna().all()):
                result[column] = numeric.max()
            else:
                result[column] = non_empty.iloc[0]
        return pd.Series(result)

    dedup = extra_df.groupby(ID_COL, as_index=False, dropna=False).apply(aggregate_group)
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
    logger.info(
        "开始回捞原始专利，共 %s 个 CSV 文件，目标申请号数=%s，坏行将直接跳过",
        len(csv_paths),
        len(target_ids),
    )
    for file_index, csv_path in enumerate(csv_paths, start=1):
        logger.info("扫描原始专利文件 [%s/%s]: %s", file_index, len(csv_paths), repo_relative(csv_path))
        reader = read_csv_with_fallback(
            csv_path,
            dtype={ID_COL: "string"},
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
            engine="python",
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


def _collect_all_rows(
    raw_patent_dir: Path,
    *,
    chunksize: int,
    logger,
) -> Any:
    frames: List[Any] = []
    csv_paths = list_csv_files(raw_patent_dir)
    logger.info(
        "开始构造 patent_master，共 %s 个 CSV 文件，坏行将直接跳过",
        len(csv_paths),
    )
    for file_index, csv_path in enumerate(csv_paths, start=1):
        logger.info("扫描原始专利文件 [%s/%s]: %s", file_index, len(csv_paths), repo_relative(csv_path))
        reader = read_csv_with_fallback(
            csv_path,
            dtype={ID_COL: "string"},
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
            engine="python",
        )
        file_rows = 0
        for chunk_index, chunk in enumerate(reader, start=1):
            chunk_df = chunk.copy()
            file_rows += len(chunk_df)
            chunk_df[ID_COL] = normalize_string_series(chunk_df[ID_COL])
            frames.append(chunk_df)
            if chunk_index % 20 == 0:
                logger.info(
                    "文件 [%s/%s] chunk=%s 已累计扫描 %s 行",
                    file_index,
                    len(csv_paths),
                    chunk_index,
                    file_rows,
                )
        logger.info(
            "原始专利文件 [%s/%s] 扫描完成: %s，累计 %s 行",
            file_index,
            len(csv_paths),
            repo_relative(csv_path),
            file_rows,
        )
    if not frames:
        return pd.DataFrame(columns=[ID_COL])
    return pd.concat(frames, ignore_index=True)


def build_patent_master(
    *,
    raw_patent_dir: Path,
    shared_root: str = "outputs/shared",
    chunksize: int = 100000,
) -> Dict[str, Path]:
    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()
    logger = build_logger("build_patent_master", shared_paths.logs_dir / "build_patent_master.log")

    extra_all = _collect_all_rows(raw_patent_dir, chunksize=chunksize, logger=logger)
    logger.info("patent_master 原始行数: %s", len(extra_all))
    patent_master = _deduplicate_extra(extra_all)
    logger.info("patent_master 去重后行数: %s", len(patent_master))

    patent_master_path = shared_paths.patent_master_dir / "patent_master.parquet"
    metadata_path = shared_paths.patent_master_dir / "metadata.json"
    patent_master.to_parquet(patent_master_path, index=False)
    write_json(
        metadata_path,
        {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "inputs": {"raw_patent_dir": repo_relative(raw_patent_dir)},
            "outputs": {"patent_master": repo_relative(patent_master_path)},
            "rows": int(len(patent_master)),
            "columns": list(patent_master.columns),
            "key_fields": [ID_COL],
            "chunksize": int(chunksize),
        },
    )
    logger.info("patent_master 输出: %s", repo_relative(patent_master_path))
    return {
        "patent_master_path": patent_master_path,
        "metadata_path": metadata_path,
    }


def build_experiment_patent_panel(
    *,
    experiment_id: str,
    stage1_output_path: Path,
    output_root: str = "outputs/experiments",
    patent_master_path: Path,
    shared_root: str = "outputs/shared",
) -> Dict[str, Path]:
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(
        f"build_experiment_patent_panel.{experiment_id}",
        paths.logs_dir / "build_experiment_patent_panel.log",
    )

    effective_patent_master_path = patent_master_path
    if not effective_patent_master_path.exists():
        raise FileNotFoundError(f"找不到 patent_master: {effective_patent_master_path}")

    logger.info("读取 stage1 主结果: %s", repo_relative(stage1_output_path))
    main_df = _load_main_output(stage1_output_path)
    logger.info("读取 patent_master: %s", repo_relative(effective_patent_master_path))
    patent_master = pd.read_parquet(effective_patent_master_path)
    patent_master[ID_COL] = normalize_string_series(patent_master[ID_COL])
    logger.info("开始按申请号拼接 experiment_patent_panel")
    experiment_patent_panel = _fill_from_raw_columns(main_df, patent_master)

    stage1_copy_path = paths.data_dir / "patent_quality_output.csv"
    main_path = paths.data_dir / "main.parquet"
    panel_path = paths.data_dir / "experiment_patent_panel.parquet"
    copy_if_needed(stage1_output_path, stage1_copy_path)
    main_df.to_parquet(main_path, index=False)
    experiment_patent_panel.to_parquet(panel_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "stage1_output": repo_relative(stage1_output_path),
        "patent_master_path": repo_relative(effective_patent_master_path),
        "main_rows": int(len(main_df)),
        "experiment_patent_panel_rows": int(len(experiment_patent_panel)),
        "outputs": {
            "stage1_copy": repo_relative(stage1_copy_path),
            "main": repo_relative(main_path),
            "experiment_patent_panel": repo_relative(panel_path),
        },
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
):
    result = build_experiment_patent_panel(
        experiment_id=experiment_id,
        stage1_output_path=stage1_output_path,
        output_root=output_root,
        patent_master_path=patent_master_path,
        shared_root=shared_root,
    )
    metadata_path = build_experiment_paths(experiment_id, output_root=output_root).metadata_dir / "build_main_enriched.json"
    write_json(
        metadata_path,
        {
            "experiment_id": experiment_id,
            "stage1_output": repo_relative(stage1_output_path),
            "patent_master_path": repo_relative(patent_master_path),
            "experiment_patent_panel_path": repo_relative(result["experiment_patent_panel_path"]),
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
