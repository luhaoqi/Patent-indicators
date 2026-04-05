from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import polars as pl  # noqa: E402

from common.analysis import resolve_patent_year_col  # noqa: E402
from common.io import build_logger, close_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


CURRENT_SCHEMA_VERSION = 2
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99
DEFAULT_HIGH_QUALITY_SHARE = 0.10


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


def _valid_ucc_expr(column: str = "UCC") -> pl.Expr:
    return (
        pl.col(column).is_not_null()
        & (pl.col(column) != "")
        & (~pl.col(column).str.to_lowercase().is_in(["nan", "none", "null"]))
    )


def _read_patents(path: Path, *, quality_cap: float, year_col: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(str(path))
        .select(
            pl.col(year_col).cast(pl.Int32, strict=False).alias("year"),
            pl.col("统一社会信用代码").cast(pl.Utf8).str.strip_chars().alias("UCC"),
            pl.col("Quality_q").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("year").is_not_null())
        .filter(_valid_ucc_expr())
        .filter(pl.col("Quality_q").is_not_null())
        .filter(pl.col("Quality_q") > 0)
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
    winsor_lower: float = DEFAULT_WINSOR_LOWER,
    winsor_upper: float = DEFAULT_WINSOR_UPPER,
    high_quality_share: float = DEFAULT_HIGH_QUALITY_SHARE,
    exact_date: bool = False,
):
    if not (0.0 <= winsor_lower < winsor_upper <= 1.0):
        raise ValueError("winsor_lower / winsor_upper 必须满足 0 <= lower < upper <= 1")
    if not (0.0 < high_quality_share < 1.0):
        raise ValueError("high_quality_share 必须位于 (0, 1)")

    paths = build_experiment_paths(experiment_id, output_root=output_root, exact_date=exact_date)
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
    patent_columns = pl.read_parquet(str(patent_path), n_rows=0).columns
    year_col = resolve_patent_year_col(patent_columns, exact_date=exact_date)
    patents = _read_patents(patent_path, quality_cap=quality_cap, year_col=year_col)
    logger.info("专利样本过滤后行数: %s", patents.height)

    logger.info(
        "开始构造 firm_year_innovation: quality_cap=%s winsor=[%.3f, %.3f] high_quality_share=%.3f legacy_top_k=%s",
        quality_cap,
        winsor_lower,
        winsor_upper,
        high_quality_share,
        top_k,
    )
    patents_enriched = (
        patents.with_columns(
            pl.col("Quality_q").quantile(winsor_lower).over("year").alias("quality_q_year_pctl_low"),
            pl.col("Quality_q").quantile(winsor_upper).over("year").alias("quality_q_year_pctl_high"),
        )
        .with_columns(
            pl.when(pl.col("Quality_q") < pl.col("quality_q_year_pctl_low"))
            .then(pl.col("quality_q_year_pctl_low"))
            .when(pl.col("Quality_q") > pl.col("quality_q_year_pctl_high"))
            .then(pl.col("quality_q_year_pctl_high"))
            .otherwise(pl.col("Quality_q"))
            .alias("Quality_q_w"),
        )
        .with_columns(
            pl.col("Quality_q_w").mean().over("year").alias("quality_q_w_year_mean"),
            pl.col("Quality_q_w").std().over("year").alias("quality_q_w_year_sd"),
            pl.col("Quality_q_w").quantile(1.0 - high_quality_share).over("year").alias("quality_q_w_year_top_cutoff"),
        )
        .with_columns(
            pl.when((pl.col("quality_q_w_year_sd").is_null()) | (pl.col("quality_q_w_year_sd") == 0))
            .then(0.0)
            .otherwise((pl.col("Quality_q_w") - pl.col("quality_q_w_year_mean")) / pl.col("quality_q_w_year_sd"))
            .alias("z_q_pft"),
            (pl.col("Quality_q_w") >= pl.col("quality_q_w_year_top_cutoff")).cast(pl.Int32).alias("top10_q_pft"),
        )
    )

    matched_patents = patents_enriched.join(ucc_map, on=["UCC", "year"], how="inner")
    logger.info("专利与上市公司 UCC 面板匹配后行数: %s", matched_patents.height)

    firm_year = (
        matched_patents
        .select(["Stkid", "ShortName", "year", "Quality_q", "Quality_q_w", "z_q_pft", "top10_q_pft"])
        .group_by(["Stkid", "ShortName", "year"])
        .agg(
            pl.len().alias("PatentCount"),
            pl.col("z_q_pft").mean().alias("mean_z_q_ft"),
            pl.col("top10_q_pft").mean().alias("highq_share_ft"),
            pl.col("top10_q_pft").sum().cast(pl.Int32).alias("highq_count_ft"),
            pl.col("Quality_q_w").mean().alias("mean_raw_q_w_ft"),
            pl.col("Quality_q").mean().alias("mean_raw_q_ft"),
            pl.col("Quality_q_w").sort(descending=True).head(top_k).mean().alias("legacy_topk_mean_q_w_ft"),
        )
        .with_columns(
            (pl.col("highq_count_ft").cast(pl.Float64) + 1.0).log().alias("log_highq_count_ft"),
            (pl.col("PatentCount").cast(pl.Float64) + 1.0).log().alias("log_patent_count_ft"),
            pl.lit(f"patent_year_winsorized_top{int(round(high_quality_share * 100))}").alias("Method"),
            pl.col("mean_raw_q_w_ft").alias("Innovation_raw"),
            pl.col("mean_z_q_ft").alias("Innovation_z"),
        )
    )

    output_path = paths.data_dir / "firm_year_innovation.parquet"
    firm_year.write_parquet(output_path)

    metadata = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_patent_panel_path": repo_relative(patent_path),
        "ucc_mapping_path": repo_relative(effective_ucc_path),
        "patent_rows_after_filters": int(patents.height),
        "matched_patent_rows": int(matched_patents.height),
        "rows": int(firm_year.height),
        "legacy_top_k": int(top_k),
        "quality_cap": float(quality_cap),
        "winsor_lower": float(winsor_lower),
        "winsor_upper": float(winsor_upper),
        "high_quality_share": float(high_quality_share),
        "method": "patent_year_winsorize_then_firm_year_aggregate",
        "implementation": "polars_window",
        "year_col": year_col,
        "exact_date": bool(exact_date),
    }
    write_json(paths.metadata_dir / "build_firm_year_innovation.json", metadata)
    logger.info("firm_year_innovation 输出: %s", repo_relative(output_path))
    close_logger(logger)
    return output_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="根据 experiment_patent_panel 和共享 UCC 映射生成 firm_year_innovation.parquet")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--experiment-patent-panel-path", help="experiment_patent_panel.parquet 路径")
    parser.add_argument("--ucc-exploded-path", help="共享 ucc_exploded.parquet 路径")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--top-k", type=int, default=10, help="保留历史 TopK 口径时使用的 K 值")
    parser.add_argument("--quality-cap", type=float, default=1000.0, help="专利层 Quality_q 工程型硬上限")
    parser.add_argument("--winsor-lower", type=float, default=DEFAULT_WINSOR_LOWER, help="年内 winsorize 下分位数")
    parser.add_argument("--winsor-upper", type=float, default=DEFAULT_WINSOR_UPPER, help="年内 winsorize 上分位数")
    parser.add_argument("--high-quality-share", type=float, default=DEFAULT_HIGH_QUALITY_SHARE, help="年内高质量专利占比阈值")
    parser.add_argument("--exact-date", action="store_true", help="使用 exact_date 模式，读取/输出 stage2_exact")
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
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        high_quality_share=args.high_quality_share,
        exact_date=args.exact_date,
    )


if __name__ == "__main__":
    main()
