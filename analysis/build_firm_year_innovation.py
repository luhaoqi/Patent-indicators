from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.io import build_logger, read_csv_with_fallback, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


def _read_ucc_list(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        required = {"Stkid", "ShortName", "year", "UCC"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[UCC exploded] 缺少列: {sorted(missing)}")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["UCC"] = df["UCC"].astype("string").fillna("").str.strip()
        df = df[
            df["year"].notna()
            & (df["UCC"] != "")
            & (df["UCC"].str.lower() != "nan")
        ].copy()
        df["year"] = df["year"].astype(int)
        return df[["Stkid", "ShortName", "year", "UCC"]].drop_duplicates()

    df = read_csv_with_fallback(path, dtype=str, low_memory=False)

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

    df = df.rename(columns=rename_map)
    required = {"Stkid", "ShortName", "year", "UCC_list"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[UCC list] 缺少列: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["UCC_list"] = df["UCC_list"].astype("string").fillna("").str.strip()
    exploded = df.assign(UCC=df["UCC_list"].str.split(";")).explode("UCC")
    exploded["UCC"] = exploded["UCC"].astype("string").fillna("").str.strip()
    exploded = exploded[
        exploded["year"].notna()
        & (exploded["UCC"] != "")
        & (exploded["UCC"].str.lower() != "nan")
    ].copy()
    exploded["year"] = exploded["year"].astype(int)
    return exploded[["Stkid", "ShortName", "year", "UCC"]].drop_duplicates()


def _read_patents(path: Path, *, quality_cap: float) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["申请年份", "统一社会信用代码", "Quality_q"])
    df = df.rename(columns={"申请年份": "year", "统一社会信用代码": "UCC"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["UCC"] = df["UCC"].astype("string").fillna("").str.strip()
    df["Quality_q"] = pd.to_numeric(df["Quality_q"], errors="coerce")
    df = df[
        df["year"].notna()
        & (df["UCC"] != "")
        & df["Quality_q"].notna()
        & (df["Quality_q"] <= quality_cap)
    ].copy()
    df["year"] = df["year"].astype(int)
    return df


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
    logger.info("UCC 面板展开后行数: %s", len(ucc_map))

    logger.info("读取专利主表: %s", repo_relative(patent_path))
    patents = _read_patents(patent_path, quality_cap=quality_cap)
    logger.info("专利样本过滤后行数: %s", len(patents))

    logger.info("开始将专利与 UCC 面板按 [UCC, year] 匹配")
    joined = patents.merge(ucc_map, on=["UCC", "year"], how="inner")
    logger.info("专利与 UCC 面板匹配后行数: %s", len(joined))

    logger.info("开始按公司-年份聚合创新指标，top_k=%s", top_k)
    grouped = (
        joined.groupby(["Stkid", "ShortName", "year"], dropna=False)["Quality_q"]
        .agg(
            PatentCount="size",
            Innovation_raw=lambda series: series.nlargest(top_k).mean(),
        )
        .reset_index()
    )
    grouped = grouped[grouped["Innovation_raw"] > 0].copy()
    grouped["Method"] = f"Top{top_k}Mean"
    logger.info("公司-年份聚合后行数: %s", len(grouped))

    logger.info("开始按年份标准化 Innovation_raw")
    stats = grouped.groupby("year")["Innovation_raw"].agg(mu="mean", sigma="std").reset_index()
    grouped = grouped.merge(stats, on="year", how="left")
    grouped["Innovation_z"] = np.where(
        grouped["sigma"].isna() | (grouped["sigma"] == 0),
        np.nan,
        (grouped["Innovation_raw"] - grouped["mu"]) / grouped["sigma"],
    )
    grouped = grouped.drop(columns=["mu", "sigma"])
    logger.info("标准化完成，开始写出 parquet")

    output_path = paths.data_dir / "firm_year_innovation.parquet"
    grouped.to_parquet(output_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "experiment_patent_panel_path": repo_relative(patent_path),
        "ucc_mapping_path": repo_relative(effective_ucc_path),
        "rows": int(len(grouped)),
        "top_k": int(top_k),
        "quality_cap": float(quality_cap),
        "method": f"Top{top_k}Mean",
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
