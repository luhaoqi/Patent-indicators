from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_main_enriched import build_patent_master  # noqa: E402
from build_raw_patent_authorized_parts import build_raw_patent_authorized_parts  # noqa: E402
from build_ucc_panel import build_ucc_mapping  # noqa: E402
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402
from shared_prep import build_financial_annual_panel, build_special_firm_labels  # noqa: E402


DEFAULT_RAW_PATENT_DIR = "data/raw/中国专利分年份保存数据1985-2025"
DEFAULT_SPECIAL_LIST_PATH = "analysis/graph/科创企业名单2024.dta"
DEFAULT_FINANCIAL_DATA_PATH = "analysis/公司财务/数据/上市公司财务数据/上市公司财务数据.dta"
DEFAULT_LISTEDCO_PARENT_PATH = "analysis/公司财务/数据/上市公司基本信息年度表/上市公司统一社会信用代码.csv"
DEFAULT_SUBSIDIARY_MAPPING_PATH = "analysis/公司财务/数据/爱企查结果/上市公司子公司对应统一社会信用代码.csv"
DEFAULT_SUBJOINT_CSV_PATH = "analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint_merged.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="一次性生成 stage2 共享预处理产物")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--raw-patent-dir", default=DEFAULT_RAW_PATENT_DIR, help="原始专利 CSV 目录")
    parser.add_argument("--special-list-path", default=DEFAULT_SPECIAL_LIST_PATH, help="特殊企业名单 dta 路径")
    parser.add_argument("--financial-data-path", default=DEFAULT_FINANCIAL_DATA_PATH, help="上市公司财务数据 dta 路径")
    parser.add_argument("--listedco-parent-path", default=DEFAULT_LISTEDCO_PARENT_PATH, help="母公司统一社会信用代码表")
    parser.add_argument("--subsidiary-mapping-path", default=DEFAULT_SUBSIDIARY_MAPPING_PATH, help="子公司名称到统一社会信用代码映射表")
    parser.add_argument("--subjoint-csv-path", default=DEFAULT_SUBJOINT_CSV_PATH, help="上市公司子公司联营合营明细表")
    parser.add_argument("--patent-chunksize", type=int, default=100000, help="构造 patent_master 的分块读取行数")
    parser.add_argument("--ucc-chunksize", type=int, default=300000, help="构造 ucc_mapping 的分块读取行数")
    parser.add_argument("--financial-year-min", type=int, default=2000, help="共享财务面板最小年份")
    parser.add_argument("--financial-year-max", type=int, default=2023, help="共享财务面板最大年份")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    shared_paths = build_shared_paths(args.shared_root)
    shared_paths.ensure_dirs()
    logger = build_logger("run_shared_prep", shared_paths.logs_dir / "run_shared_prep.log")

    raw_patent_dir = resolve_repo_path(args.raw_patent_dir)
    special_list_path = resolve_repo_path(args.special_list_path)
    financial_data_path = resolve_repo_path(args.financial_data_path)
    listedco_parent_path = resolve_repo_path(args.listedco_parent_path)
    subsidiary_mapping_path = resolve_repo_path(args.subsidiary_mapping_path)
    subjoint_csv_path = resolve_repo_path(args.subjoint_csv_path)

    assert raw_patent_dir is not None
    assert special_list_path is not None
    assert financial_data_path is not None
    assert listedco_parent_path is not None
    assert subsidiary_mapping_path is not None
    assert subjoint_csv_path is not None

    logger.info("开始构造共享产物，shared_root=%s", repo_relative(shared_paths.root))

    patent_master = build_patent_master(
        raw_patent_dir=raw_patent_dir,
        shared_root=args.shared_root,
        chunksize=args.patent_chunksize,
    )
    raw_authorized_parts = build_raw_patent_authorized_parts(
        raw_patent_dir=raw_patent_dir,
        shared_root=args.shared_root,
        chunksize=args.patent_chunksize,
    )
    special_labels = build_special_firm_labels(
        special_list_path=special_list_path,
        shared_root=args.shared_root,
    )
    ucc_mapping = build_ucc_mapping(
        parent_csv_path=listedco_parent_path,
        subsidiary_mapping_path=subsidiary_mapping_path,
        subjoint_csv_path=subjoint_csv_path,
        shared_root=args.shared_root,
        chunksize=args.ucc_chunksize,
    )
    financial_panel = build_financial_annual_panel(
        financial_data_path=financial_data_path,
        shared_root=args.shared_root,
        year_min=args.financial_year_min,
        year_max=args.financial_year_max,
    )

    summary = {
        "shared_root": repo_relative(shared_paths.root),
        "outputs": {
            "patent_master": repo_relative(patent_master["patent_master_path"]),
            "raw_patent_authorized_parts_dir": repo_relative(raw_authorized_parts["output_dir"]),
            "raw_patent_authorized_parts_metadata": repo_relative(raw_authorized_parts["metadata_path"]),
            "firm_year_special_labels": repo_relative(special_labels["firm_year_special_labels_path"]),
            "special_ucc_set": repo_relative(special_labels["special_ucc_set_path"]),
            "ucc_panel": repo_relative(ucc_mapping["ucc_panel_path"]),
            "ucc_exploded": repo_relative(ucc_mapping["ucc_exploded_path"]),
            "financial_annual_clean": repo_relative(financial_panel["financial_annual_clean_path"]),
        },
    }
    write_json(shared_paths.metadata_dir / "run_shared_prep.json", summary)
    logger.info("共享预处理完成")


if __name__ == "__main__":
    main()
