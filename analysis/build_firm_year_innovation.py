from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import polars as pl  # noqa: E402

from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


def _read_ucc_list(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(str(path))
        required = {"Stkid", "ShortName", "year", "UCC"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[UCC exploded] 缺少列: {sorted(missing)}")
        return (
            df.select(
                pl.col("Stkid").cast(pl.Utf8).str.strip_chars(),
                pl.col("ShortName").cast(pl.Utf8, strict=False).alias("ShortName"),
                pl.col("year").cast(pl.Int32, strict=False),
                pl.col("UCC").cast(pl.Utf8).str.strip_chars(),
            )
            .filter(pl.col("year").is_not_null())
            .filter(pl.col("UCC").is_not_null() & (pl.col("UCC") != "") & (pl.col("UCC").str.to_lowercase() != "nan"))
            .unique(subset=["Stkid", "year", "UCC"])
        )

    df = pl.read_csv(str(path), infer_schema_length=10000, encoding="utf8-lossy", ignore_errors=True)

    rename_map = {}
    if "证券ID" in df.columns:
        rename_map["证券ID"] = "Stkid"
    if "stkid" in df.columns:
        rename_map["stkid"] = "Stkid"
    if "公司简称" in df.columns:
        rename_map["公司简称"] = "ShortName"
    if "shortname" in df.columns:
        rename_map["shortname"] = "ShortName"
    if "年份" in df.columns:
        rename_map["年份"] = "year"
    if "统一社会信用代码列表" in df.columns:
        rename_map["统一社会信用代码列表"] = "UCC_list"
    elif "统一社会信用代码" in df.columns:
        rename_map["统一社会信用代码"] = "UCC_list"

    df = df.rename(rename_map)
    required = {"Stkid", "ShortName", "year", "UCC_list"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[UCC list] 缺少列: {sorted(missing)}")

    return (
        df.with_columns(
            pl.col("Stkid").cast(pl.Utf8).str.strip_chars(),
            pl.col("ShortName").cast(pl.Utf8, strict=False).alias("ShortName"),
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("UCC_list")
            .cast(pl.Utf8)
            .fill_null("")
            .str.strip_chars()
            .str.split(";")
            .alias("UCC_arr"),
        )
        .explode("UCC_arr")
        .with_columns(pl.col("UCC_arr").cast(pl.Utf8).str.strip_chars().alias("UCC"))
        .filter(pl.col("year").is_not_null())
        .filter(pl.col("UCC").is_not_null() & (pl.col("UCC") != "") & (pl.col("UCC").str.to_lowercase() != "nan"))
        .select(["Stkid", "ShortName", "year", "UCC"])
        .unique(subset=["Stkid", "year", "UCC"])
    )


def _read_patents(path: Path, *, quality_cap: float) -> pl.DataFrame:
    return (
        pl.scan_parquet(str(path))
        .select(
            pl.col("申请年份").cast(pl.Int32, strict=False).alias("year"),
            pl.col("统一社会信用代码").cast(pl.Utf8).str.strip_chars().alias("UCC"),
            pl.col("Quality_q").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("year").is_not_null())
        .filter(pl.col("UCC").is_not_null())
        .filter(pl.col("UCC") != "")
        .filter(pl.col("Quality_q").is_not_null())
        .filter(pl.col("Quality_q") <= quality_cap)
        .collect(engine="streaming")
    )


def build_firm_year_innovation(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    experiment_patent_panel_path: Optional[Path] = None,
    ucc_exploded_path: Optional[Path] = None,
    shared_root: str = "outputs/shared",
    top_k: int = 10,
    quality_cap: float = 1000.0,
):
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(
        f"build_firm_year_innovation.{experiment_id}",
        paths.logs_dir / "build_firm_year_innovation.log",
    )

    patent_path = experiment_patent_panel_path or (paths.data_dir / "experiment_patent_panel.parquet")
    if not patent_path.exists():
        raise FileNotFoundError(f"找不到 experiment_patent_panel: {patent_path}")

    effective_ucc_path = ucc_exploded_path
    if effective_ucc_path is None:
        shared_paths = build_shared_paths(shared_root)
        parquet_candidate = shared_paths.ucc_mapping_dir / "ucc_exploded.parquet"
        if parquet_candidate.exists():
            effective_ucc_path = parquet_candidate
    if effective_ucc_path is None:
        raise FileNotFoundError("缺少共享 UCC 映射，请先运行 run_shared_prep.py 生成 shared ucc_mapping")

    logger.info("读取 UCC 面板: %s", repo_relative(effective_ucc_path))
    ucc_map = _read_ucc_list(effective_ucc_path)
    logger.info("UCC 面板展开后行数: %s", ucc_map.height)

    logger.info("读取专利主表: %s", repo_relative(patent_path))
    patents = _read_patents(patent_path, quality_cap=quality_cap)
    logger.info("专利样本过滤后行数: %s", patents.height)

    logger.info("开始按 notebook 路径聚合 firm_year_innovation，top_k=%s", top_k)
    firm_year = (
        patents.join(ucc_map, on=["UCC", "year"], how="inner")
        .select(["Stkid", "ShortName", "year", "Quality_q"])
        .group_by(["Stkid", "ShortName", "year"])
        .agg(
            pl.col("Quality_q").count().alias("PatentCount"),
            pl.col("Quality_q").sort(descending=True).head(top_k).mean().alias("Innovation_raw"),
        )
        .filter(pl.col("Innovation_raw") > 0)
        .with_columns(pl.lit(f"Top{top_k}Mean").alias("Method"))
    )

    stats = (
        firm_year.group_by("year")
        .agg(
            pl.col("Innovation_raw").mean().alias("mu"),
            pl.col("Innovation_raw").std().alias("sigma"),
        )
    )
    firm_year = (
        firm_year.join(stats, on="year", how="left")
        .with_columns(
            pl.when((pl.col("sigma").is_null()) | (pl.col("sigma") == 0))
            .then(None)
            .otherwise((pl.col("Innovation_raw") - pl.col("mu")) / pl.col("sigma"))
            .alias("Innovation_z")
        )
        .drop(["mu", "sigma"])
    )

    output_path = paths.data_dir / "firm_year_innovation.parquet"
    firm_year.write_parquet(output_path)

    metadata = {
        "experiment_id": experiment_id,
        "experiment_patent_panel_path": repo_relative(patent_path),
        "ucc_mapping_path": repo_relative(effective_ucc_path),
        "rows": int(firm_year.height),
        "top_k": int(top_k),
        "quality_cap": float(quality_cap),
        "method": f"Top{top_k}Mean",
        "implementation": "polars_streaming",
    }
    write_json(paths.metadata_dir / "build_firm_year_innovation.json", metadata)
    logger.info("firm_year_innovation 输出: %s", repo_relative(output_path))
    return output_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="根据 experiment_patent_panel 和共享 UCC 映射生成 firm_year_innovation.parquet")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--experiment-patent-panel-path", help="experiment_patent_panel.parquet 路径")
    parser.add_argument("--ucc-exploded-path", help="共享 ucc_exploded.parquet 路径")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--top-k", type=int, default=10, help="firm-year 创新指数采用的 TopK 均值")
    parser.add_argument("--quality-cap", type=float, default=1000.0, help="专利层 Quality_q 上限")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    build_firm_year_innovation(
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        experiment_patent_panel_path=resolve_repo_path(args.experiment_patent_panel_path) if args.experiment_patent_panel_path else None,
        ucc_exploded_path=resolve_repo_path(args.ucc_exploded_path) if args.ucc_exploded_path else None,
        shared_root=args.shared_root,
        top_k=args.top_k,
        quality_cap=args.quality_cap,
    )


if __name__ == "__main__":
    main()
