from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re
import sys
from typing import Dict, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd  # noqa: E402

from common.io import build_logger, read_csv_with_fallback, write_json  # noqa: E402
from common.paths import build_experiment_paths, repo_relative, resolve_repo_path  # noqa: E402


SEP = ";"
YEAR4_RE = re.compile(r"^\d{4}$")


def normalize_seps(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip(SEP)
    while f"{SEP}{SEP}" in text:
        text = text.replace(f"{SEP}{SEP}", SEP)
    return text


def _load_parent_table(path: Path) -> pd.DataFrame:
    parent = read_csv_with_fallback(path, dtype=str, low_memory=False)
    parent.columns = [column.strip() for column in parent.columns]
    required = ["stkid", "shortname", "SocialCreditCode", "FirstYear", "LastYear"]
    missing = [column for column in required if column not in parent.columns]
    if missing:
        raise KeyError(f"母公司表缺少列: {missing}")

    parent["stkid"] = parent["stkid"].astype(str).str.strip()
    parent["shortname"] = parent["shortname"].astype(str).str.strip()
    parent["SocialCreditCode"] = parent["SocialCreditCode"].astype(str).str.strip()
    parent["FirstYear"] = pd.to_numeric(parent["FirstYear"], errors="coerce")
    parent["LastYear"] = pd.to_numeric(parent["LastYear"], errors="coerce")
    parent = parent[
        parent["stkid"].notna()
        & (parent["stkid"] != "")
        & parent["SocialCreditCode"].notna()
        & (parent["SocialCreditCode"] != "")
        & parent["FirstYear"].notna()
        & parent["LastYear"].notna()
    ].copy()
    parent["FirstYear"] = parent["FirstYear"].astype(int)
    parent["LastYear"] = parent["LastYear"].astype(int)
    return parent


def _load_subsidiary_mapping(path: Path) -> Dict[str, str]:
    subs_map = read_csv_with_fallback(path, dtype=str, low_memory=False)
    subs_map.columns = [column.strip() for column in subs_map.columns]
    required = ["企业名称", "统一社会信用代码"]
    missing = [column for column in required if column not in subs_map.columns]
    if missing:
        raise KeyError(f"子公司映射表缺少列: {missing}")

    subs_map["企业名称"] = subs_map["企业名称"].astype(str).str.strip()
    subs_map["统一社会信用代码"] = subs_map["统一社会信用代码"].astype(str).str.strip()
    subs_map = subs_map[(subs_map["企业名称"] != "") & (subs_map["统一社会信用代码"] != "")].copy()
    return (
        subs_map.groupby("企业名称")["统一社会信用代码"]
        .apply(lambda values: SEP.join(sorted(set(values))))
        .to_dict()
    )


def build_ucc_panel(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    parent_csv_path: Path,
    subsidiary_mapping_path: Path,
    subjoint_csv_path: Path,
    output_path: Optional[Path] = None,
    chunksize: int = 300000,
) -> Path:
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(f"build_ucc_panel.{experiment_id}", paths.logs_dir / "build_ucc_panel.log")

    parent = _load_parent_table(parent_csv_path)
    name_to_ucc = _load_subsidiary_mapping(subsidiary_mapping_path)
    logger.info("母公司行数=%s, 唯一证券ID=%s", len(parent), parent["stkid"].nunique())
    logger.info("子公司名称映射数量=%s", len(name_to_ucc))

    usecols = ["Symbol", "EndDate", "RalatedParty", "Relationship"]
    chunk_iter = read_csv_with_fallback(
        subjoint_csv_path,
        dtype=str,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    child_acc: dict[tuple[str, int], str] = {}
    total_rows = 0
    kept_rows = 0
    miss_map_rows = 0

    for chunk_index, chunk in enumerate(chunk_iter, start=1):
        total_rows += len(chunk)
        chunk["Symbol"] = chunk["Symbol"].astype(str).str.strip()
        chunk["EndDate"] = chunk["EndDate"].astype(str).str.strip()
        chunk["RalatedParty"] = chunk["RalatedParty"].astype(str).str.strip()
        chunk["Year"] = chunk["EndDate"].str.slice(0, 4)
        chunk = chunk[chunk["Year"].map(lambda value: bool(YEAR4_RE.match(str(value))))].copy()
        chunk["Year"] = chunk["Year"].astype(int)
        chunk["ChildUCCStr"] = chunk["RalatedParty"].map(name_to_ucc)
        miss_map_rows += int(chunk["ChildUCCStr"].isna().sum())
        chunk = chunk[chunk["ChildUCCStr"].notna()].copy()
        kept_rows += len(chunk)

        if not chunk.empty:
            grouped = chunk.groupby(["Symbol", "Year"])["ChildUCCStr"].apply(lambda values: SEP.join(values.astype(str).tolist()))
            for (symbol, year), values in grouped.items():
                key = (symbol, int(year))
                child_acc[key] = f"{child_acc[key]}{SEP}{values}" if key in child_acc else str(values)

        if chunk_index % 10 == 0:
            logger.info(
                "扫描子公司明细 chunk=%s total_rows=%s kept_rows=%s acc_keys=%s",
                chunk_index,
                total_rows,
                kept_rows,
                len(child_acc),
            )

    logger.info(
        "子公司明细扫描完成 total_rows=%s kept_rows=%s miss_map_rows=%s acc_keys=%s",
        total_rows,
        kept_rows,
        miss_map_rows,
        len(child_acc),
    )

    out_rows: list[list[object]] = []
    for _, row in parent.iterrows():
        stkid = row["stkid"]
        shortname = row["shortname"]
        parent_ucc = normalize_seps(row["SocialCreditCode"])
        first_year = int(row["FirstYear"])
        last_year = int(row["LastYear"])
        if first_year > last_year or first_year < 1900 or last_year > 2100:
            continue
        for year in range(first_year, last_year + 1):
            child_str = child_acc.get((stkid, year), "")
            ucc_list = normalize_seps(f"{parent_ucc}{SEP}{child_str}" if child_str else parent_ucc)
            out_rows.append([stkid, shortname, year, ucc_list])

    out_df = pd.DataFrame(out_rows, columns=["证券ID", "公司简称", "年份", "统一社会信用代码列表"])
    final_output = output_path or (paths.data_dir / "ucc_panel.csv")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(final_output, index=False, encoding="utf-8-sig")

    write_json(
        paths.metadata_dir / "build_ucc_panel.json",
        {
            "experiment_id": experiment_id,
            "parent_csv_path": repo_relative(parent_csv_path),
            "subsidiary_mapping_path": repo_relative(subsidiary_mapping_path),
            "subjoint_csv_path": repo_relative(subjoint_csv_path),
            "output_path": repo_relative(final_output),
            "rows": int(len(out_df)),
            "unique_stkid": int(out_df["证券ID"].nunique()),
            "year_min": int(out_df["年份"].min()) if not out_df.empty else None,
            "year_max": int(out_df["年份"].max()) if not out_df.empty else None,
        },
    )
    logger.info("UCC 面板输出: %s", repo_relative(final_output))
    return final_output


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="生成上市公司（包括子公司）年度统一社会信用代码面板")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument(
        "--parent-csv-path",
        default="analysis/公司财务/数据/上市公司基本信息年度表/上市公司统一社会信用代码.csv",
        help="母公司统一社会信用代码表",
    )
    parser.add_argument(
        "--subsidiary-mapping-path",
        default="analysis/公司财务/数据/爱企查结果/上市公司子公司对应统一社会信用代码.csv",
        help="子公司名称到统一社会信用代码映射表",
    )
    parser.add_argument(
        "--subjoint-csv-path",
        default="analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint_merged.csv",
        help="上市公司子公司联营合营情况明细表",
    )
    parser.add_argument("--output-path", help="输出 CSV 路径，不传则写入 experiment data 目录")
    parser.add_argument("--chunksize", type=int, default=300000, help="分块读取子公司明细的行数")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    build_ucc_panel(
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        parent_csv_path=resolve_repo_path(args.parent_csv_path),  # type: ignore[arg-type]
        subsidiary_mapping_path=resolve_repo_path(args.subsidiary_mapping_path),  # type: ignore[arg-type]
        subjoint_csv_path=resolve_repo_path(args.subjoint_csv_path),  # type: ignore[arg-type]
        output_path=resolve_repo_path(args.output_path) if args.output_path else None,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
