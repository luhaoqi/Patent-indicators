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

from common.io import build_logger, close_logger, write_json  # noqa: E402
from common.paths import build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402


CURRENT_SCHEMA_VERSION = 1
DEFAULT_HIGH_CITE_SHARE = 0.10
PRIMARY_CITE_COL = "被他引次数"
SECONDARY_CITE_COL = "被引证次数"


def _valid_ucc_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.notna() & (text != "") & (~text.str.lower().isin(["nan", "none", "null"]))


def _read_patent_master(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=["公开公告年份", "统一社会信用代码", PRIMARY_CITE_COL, SECONDARY_CITE_COL],
    )
    df = df.rename(
        columns={
            "公开公告年份": "year",
            "统一社会信用代码": "UCC",
            PRIMARY_CITE_COL: "cite_other_raw",
            SECONDARY_CITE_COL: "cite_total_raw",
        }
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["UCC"] = df["UCC"].astype("string").str.strip()
    df = df[df["year"].notna() & _valid_ucc_mask(df["UCC"])].copy()
    df["year"] = df["year"].astype(int)
    df["cite_other_raw"] = pd.to_numeric(df["cite_other_raw"], errors="coerce").fillna(0.0)
    df["cite_total_raw"] = pd.to_numeric(df["cite_total_raw"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _read_ucc_exploded(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["Stkid", "ShortName", "year", "UCC"])
    df["Stkid"] = df["Stkid"].astype("string").str.strip()
    df["ShortName"] = df["ShortName"].astype("string")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["UCC"] = df["UCC"].astype("string").str.strip()
    df = df[df["year"].notna() & _valid_ucc_mask(df["UCC"])].copy()
    df["year"] = df["year"].astype(int)
    return df.drop_duplicates(["Stkid", "year", "UCC"]).reset_index(drop=True)


def _add_year_top_flags(df: pd.DataFrame, *, high_cite_share: float) -> pd.DataFrame:
    cutoff_q = 1.0 - high_cite_share
    for source, flag in (
        ("cite_other_raw", "top_other_flag"),
        ("cite_total_raw", "top_total_flag"),
    ):
        cutoff = df.groupby("year")[source].transform(lambda s: s.quantile(cutoff_q))
        df[flag] = (df[source] >= cutoff).astype(np.int32)
    return df


def build_firm_year_citations(
    *,
    shared_root: str = "outputs/shared",
    patent_master_path: Optional[Path] = None,
    ucc_exploded_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    high_cite_share: float = DEFAULT_HIGH_CITE_SHARE,
) -> dict[str, object]:
    if not (0.0 < high_cite_share < 1.0):
        raise ValueError("high_cite_share 必须位于 (0, 1)")

    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()

    if patent_master_path is None:
        patent_master_path = shared_paths.patent_master_dir / "patent_master.parquet"
    if ucc_exploded_path is None:
        ucc_exploded_path = shared_paths.ucc_mapping_dir / "ucc_exploded.parquet"
    if output_dir is None:
        output_dir = shared_paths.root / "firm_year_citations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger("build_firm_year_citations", shared_paths.logs_dir / "build_firm_year_citations.log")
    logger.info("读取 patent_master: %s", repo_relative(patent_master_path))
    patents = _read_patent_master(patent_master_path)
    logger.info("patent_master 过滤后行数: %s", len(patents))

    logger.info("计算年内 top%.0f%% 引用阈值", high_cite_share * 100)
    patents = _add_year_top_flags(patents, high_cite_share=high_cite_share)

    logger.info("读取 ucc_exploded: %s", repo_relative(ucc_exploded_path))
    ucc = _read_ucc_exploded(ucc_exploded_path)
    logger.info("ucc_exploded 行数: %s", len(ucc))

    logger.info("按 (UCC, year) inner join 专利与上市公司")
    merged = patents.merge(ucc, on=["UCC", "year"], how="inner")
    logger.info("匹配后专利-公司行数: %s", len(merged))

    logger.info("按 (Stkid, year) 聚合公司年引用指标")
    grouped = (
        merged.groupby(["Stkid", "ShortName", "year"], dropna=False)
        .agg(
            PatentCount_cite=("cite_other_raw", "size"),
            cite_other_sum_ft=("cite_other_raw", "sum"),
            mean_cite_other_ft=("cite_other_raw", "mean"),
            highcite_other_count_ft=("top_other_flag", "sum"),
            highcite_other_share_ft=("top_other_flag", "mean"),
            cite_total_sum_ft=("cite_total_raw", "sum"),
            mean_cite_total_ft=("cite_total_raw", "mean"),
            highcite_total_count_ft=("top_total_flag", "sum"),
            highcite_total_share_ft=("top_total_flag", "mean"),
        )
        .reset_index()
    )

    grouped["highcite_other_count_ft"] = grouped["highcite_other_count_ft"].astype(np.int64)
    grouped["highcite_total_count_ft"] = grouped["highcite_total_count_ft"].astype(np.int64)
    grouped["log_cite_other_sum_ft"] = np.log1p(grouped["cite_other_sum_ft"].clip(lower=0))
    grouped["log_cite_total_sum_ft"] = np.log1p(grouped["cite_total_sum_ft"].clip(lower=0))
    grouped["log_highcite_other_count_ft"] = np.log1p(grouped["highcite_other_count_ft"].clip(lower=0))
    grouped["log_highcite_total_count_ft"] = np.log1p(grouped["highcite_total_count_ft"].clip(lower=0))
    grouped["Method"] = f"patent_master_fillna0_top{int(round(high_cite_share * 100))}"

    output_path = output_dir / "firm_year_citations.parquet"
    grouped.to_parquet(output_path, index=False)
    logger.info("firm_year_citations 写出: %s (rows=%s)", repo_relative(output_path), len(grouped))

    metadata = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "patent_master_path": repo_relative(patent_master_path),
        "ucc_exploded_path": repo_relative(ucc_exploded_path),
        "output_path": repo_relative(output_path),
        "rows": int(len(grouped)),
        "firms": int(grouped["Stkid"].nunique()),
        "year_min": int(grouped["year"].min()),
        "year_max": int(grouped["year"].max()),
        "high_cite_share": float(high_cite_share),
        "primary_cite_source": PRIMARY_CITE_COL,
        "secondary_cite_source": SECONDARY_CITE_COL,
        "nan_treatment": "fillna(0)",
        "year_attribution": "公开公告年份",
    }
    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    logger.info("metadata 写出: %s", repo_relative(metadata_path))
    close_logger(logger)

    return {
        "firm_year_citations_path": output_path,
        "metadata_path": metadata_path,
        **metadata,
    }


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="基于 patent_master + ucc_exploded 聚合公司年专利被引指标")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--patent-master-path", help="patent_master.parquet 路径")
    parser.add_argument("--ucc-exploded-path", help="ucc_exploded.parquet 路径")
    parser.add_argument("--output-dir", help="输出目录，默认 <shared_root>/firm_year_citations")
    parser.add_argument(
        "--high-cite-share",
        type=float,
        default=DEFAULT_HIGH_CITE_SHARE,
        help="年内高被引专利占比阈值，默认 0.10 即 top10",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    build_firm_year_citations(
        shared_root=args.shared_root,
        patent_master_path=resolve_repo_path(args.patent_master_path) if args.patent_master_path else None,
        ucc_exploded_path=resolve_repo_path(args.ucc_exploded_path) if args.ucc_exploded_path else None,
        output_dir=resolve_repo_path(args.output_dir) if args.output_dir else None,
        high_cite_share=args.high_cite_share,
    )


if __name__ == "__main__":
    main()
