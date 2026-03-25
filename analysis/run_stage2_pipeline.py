from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analyze_quality_basic import analyze_quality_basic  # noqa: E402
from analyze_special_firms import analyze_special_firms  # noqa: E402
from build_firm_year_innovation import build_firm_year_innovation  # noqa: E402
from build_main_enriched import build_main_enriched  # noqa: E402
from build_ucc_panel import build_ucc_panel  # noqa: E402
from common.config import Stage2Config  # noqa: E402
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, repo_relative, resolve_repo_path  # noqa: E402
from run_regressions import run_regressions  # noqa: E402


def run_stage2(
    *,
    experiment_id: str,
    stage1_dir: Path,
    output_root: str = "outputs/experiments",
    raw_patent_dir: Path,
    special_list_path: Optional[Path] = None,
    financial_data_path: Optional[Path] = None,
    ucc_panel_path: Optional[Path] = None,
    listedco_parent_path: Optional[Path] = None,
    subsidiary_mapping_path: Optional[Path] = None,
    subjoint_csv_path: Optional[Path] = None,
    topk_values: Sequence[int] = (10, 30, 50),
    yearly_top_vocab_k: int = 50,
    max_year_gap: int = 5,
    exclude_years: Sequence[int] = (1985, 1986),
    quality_min: float = 1e-5,
    bs_min: float = 1e-6,
    analysis_quality_threshold: float = 1.0,
    quality_desc_threshold: float = 5.0,
    policy_start_year: int = 2008,
    event_window: int = 5,
    innovation_top_k: int = 10,
    innovation_quality_cap: float = 1000.0,
    regression_year_min: int = 2000,
    regression_year_max: int = 2023,
    chunksize: int = 100000,
) -> Dict[str, Any]:
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(f"run_stage2_pipeline.{experiment_id}", paths.stage2_log_path())
    stage2_config = Stage2Config.from_runtime(
        experiment_id=experiment_id,
        stage1_dir=stage1_dir,
        raw_patent_dir=raw_patent_dir,
        output_root=output_root,
        special_list_path=special_list_path,
        financial_data_path=financial_data_path,
        ucc_panel_path=ucc_panel_path,
        listedco_parent_path=listedco_parent_path,
        subsidiary_mapping_path=subsidiary_mapping_path,
        subjoint_csv_path=subjoint_csv_path,
        topk_values=topk_values,
        yearly_top_vocab_k=yearly_top_vocab_k,
        max_year_gap=max_year_gap,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        analysis_quality_threshold=analysis_quality_threshold,
        quality_desc_threshold=quality_desc_threshold,
        policy_start_year=policy_start_year,
        event_window=event_window,
        innovation_top_k=innovation_top_k,
        innovation_quality_cap=innovation_quality_cap,
        regression_year_min=regression_year_min,
        regression_year_max=regression_year_max,
        chunksize=chunksize,
    )
    write_json(paths.metadata_dir / "stage2_config.json", stage2_config.to_payload())

    logger.info("Stage2 开始: experiment_id=%s", experiment_id)
    logger.info("stage1_dir=%s", repo_relative(stage1_dir))
    logger.info("raw_patent_dir=%s", repo_relative(raw_patent_dir))

    step_summaries: Dict[str, Any] = {}

    logger.info("[1/7] 运行 diagnostics")
    from common.diagnostics import run_diagnostics as run_diagnostics_outputs  # noqa: E402

    diagnostics_logger = build_logger(f"run_diagnostics.{experiment_id}", paths.logs_dir / "run_diagnostics.log")
    step_start = time.perf_counter()
    diagnostics_written = run_diagnostics_outputs(
        stage1_dir=stage1_dir,
        diagnostics_dir=paths.diagnostics_dir,
        topk_values=topk_values,
        yearly_top_vocab_k=yearly_top_vocab_k,
        max_year_gap=max_year_gap,
        logger=diagnostics_logger,
    )
    step_summaries["diagnostics"] = [repo_relative(path) for path in diagnostics_written]
    logger.info("[1/7] diagnostics 完成，用时 %.1fs，输出 %s 个文件", time.perf_counter() - step_start, len(diagnostics_written))

    logger.info("[2/7] 构造 main_enriched")
    step_start = time.perf_counter()
    stage1_output = stage1_dir / "patent_quality_output.csv"
    main_result = build_main_enriched(
        experiment_id=experiment_id,
        stage1_output_path=stage1_output,
        raw_patent_dir=raw_patent_dir,
        output_root=output_root,
        chunksize=chunksize,
    )
    main_enriched_path = main_result["enriched_path"]
    step_summaries["build_main_enriched"] = {key: repo_relative(value) if isinstance(value, Path) else value for key, value in main_result.items() if key != "paths"}
    logger.info("[2/7] build_main_enriched 完成，用时 %.1fs", time.perf_counter() - step_start)

    logger.info("[3/7] 输出基础图表与描述统计")
    step_start = time.perf_counter()
    basic_summary = analyze_quality_basic(
        experiment_id=experiment_id,
        output_root=output_root,
        main_enriched_path=main_enriched_path,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        quality_desc_threshold=quality_desc_threshold,
    )
    step_summaries["analyze_quality_basic"] = basic_summary
    logger.info("[3/7] analyze_quality_basic 完成，用时 %.1fs", time.perf_counter() - step_start)

    if special_list_path is not None:
        logger.info("[4/7] 输出特殊企业对比分析")
        step_start = time.perf_counter()
        special_summary = analyze_special_firms(
            experiment_id=experiment_id,
            output_root=output_root,
            main_enriched_path=main_enriched_path,
            special_list_path=special_list_path,
            exclude_years=exclude_years,
            quality_min=quality_min,
            bs_min=bs_min,
            quality_threshold=analysis_quality_threshold,
            policy_start_year=policy_start_year,
            event_window=event_window,
        )
        step_summaries["analyze_special_firms"] = special_summary
        logger.info("[4/7] analyze_special_firms 完成，用时 %.1fs", time.perf_counter() - step_start)
    else:
        logger.info("[4/7] 跳过特殊企业分析: 未提供 special_list_path")

    effective_ucc_panel_path = ucc_panel_path if (ucc_panel_path is not None and ucc_panel_path.exists()) else None
    if effective_ucc_panel_path is None and listedco_parent_path and subsidiary_mapping_path and subjoint_csv_path:
        logger.info("[5/7] 生成 UCC 面板")
        step_start = time.perf_counter()
        effective_ucc_panel_path = build_ucc_panel(
            experiment_id=experiment_id,
            output_root=output_root,
            parent_csv_path=listedco_parent_path,
            subsidiary_mapping_path=subsidiary_mapping_path,
            subjoint_csv_path=subjoint_csv_path,
        )
        step_summaries["build_ucc_panel"] = repo_relative(effective_ucc_panel_path)
        logger.info("[5/7] build_ucc_panel 完成，用时 %.1fs", time.perf_counter() - step_start)
    elif effective_ucc_panel_path is not None:
        logger.info("[5/7] 使用现成 UCC 面板: %s", repo_relative(effective_ucc_panel_path))
        step_summaries["build_ucc_panel"] = repo_relative(effective_ucc_panel_path)
    else:
        logger.info("[5/7] 跳过 UCC 面板生成: 未提供输入路径")

    innovation_path: Optional[Path] = None
    if effective_ucc_panel_path is not None:
        logger.info("[6/7] 构造 firm_year_innovation")
        step_start = time.perf_counter()
        innovation_path = build_firm_year_innovation(
            experiment_id=experiment_id,
            output_root=output_root,
            main_enriched_path=main_enriched_path,
            ucc_panel_path=effective_ucc_panel_path,
            top_k=innovation_top_k,
            quality_cap=innovation_quality_cap,
        )
        step_summaries["build_firm_year_innovation"] = repo_relative(innovation_path)
        logger.info("[6/7] build_firm_year_innovation 完成，用时 %.1fs", time.perf_counter() - step_start)
    else:
        logger.info("[6/7] 跳过 firm_year_innovation: 缺少 UCC 面板")

    if innovation_path is not None and financial_data_path is not None:
        logger.info("[7/7] 运行固定效应回归")
        step_start = time.perf_counter()
        regression_summary = run_regressions(
            experiment_id=experiment_id,
            output_root=output_root,
            firm_year_innovation_path=innovation_path,
            financial_data_path=financial_data_path,
            year_min=regression_year_min,
            year_max=regression_year_max,
        )
        step_summaries["run_regressions"] = regression_summary
        logger.info("[7/7] run_regressions 完成，用时 %.1fs", time.perf_counter() - step_start)
    else:
        logger.info("[7/7] 跳过固定效应回归: 缺少 firm_year_innovation 或 financial_data_path")

    summary = {
        "experiment_id": experiment_id,
        "stage1_dir": repo_relative(stage1_dir),
        "output_root": output_root,
        "config_path": repo_relative(paths.metadata_dir / "stage2_config.json"),
        "steps": step_summaries,
    }
    write_json(paths.metadata_dir / "run_stage2_pipeline.json", summary)
    logger.info("Stage2 完成: %s", experiment_id)
    return summary


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="运行单个实验的完整 stage2 流程")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--stage1-dir", required=True, help="stage1 结果目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--raw-patent-dir", default="data/raw/中国专利分年份保存数据1985-2025", help="原始专利 CSV 目录")
    parser.add_argument("--special-list-path", help="特殊企业名单 dta 路径")
    parser.add_argument("--financial-data-path", help="上市公司财务数据 dta 路径")
    parser.add_argument("--ucc-panel-path", help="现成的 UCC 面板 CSV 路径")
    parser.add_argument("--listedco-parent-path", help="母公司统一社会信用代码表")
    parser.add_argument("--subsidiary-mapping-path", help="子公司名称到统一社会信用代码映射表")
    parser.add_argument("--subjoint-csv-path", help="上市公司子公司联营合营明细表")
    parser.add_argument("--chunksize", type=int, default=100000, help="build_main_enriched 分块读取行数")
    parser.add_argument("--innovation-top-k", type=int, default=10, help="firm-year 创新指数 TopK")
    parser.add_argument("--innovation-quality-cap", type=float, default=1000.0, help="firm-year 创新指数 Quality_q 上限")
    parser.add_argument("--analysis-quality-threshold", type=float, default=1.0, help="企业对比中的高质量阈值")
    parser.add_argument("--quality-desc-threshold", type=float, default=5.0, help="基础描述统计中的高质量阈值")
    parser.add_argument("--quality-min", type=float, default=1e-5, help="Quality_q 最小阈值")
    parser.add_argument("--bs-min", type=float, default=1e-6, help="BS 最小阈值")
    parser.add_argument("--policy-start-year", type=int, default=2008, help="特殊企业政策起始年份")
    parser.add_argument("--event-window", type=int, default=5, help="事件研究窗口")
    parser.add_argument("--regression-year-min", type=int, default=2000, help="回归最小年份")
    parser.add_argument("--regression-year-max", type=int, default=2023, help="回归最大年份")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    stage1_dir = resolve_repo_path(args.stage1_dir)
    raw_patent_dir = resolve_repo_path(args.raw_patent_dir)
    assert stage1_dir is not None
    assert raw_patent_dir is not None
    run_stage2(
        experiment_id=args.experiment_id,
        stage1_dir=stage1_dir,
        output_root=args.output_root,
        raw_patent_dir=raw_patent_dir,
        special_list_path=resolve_repo_path(args.special_list_path) if args.special_list_path else None,
        financial_data_path=resolve_repo_path(args.financial_data_path) if args.financial_data_path else None,
        ucc_panel_path=resolve_repo_path(args.ucc_panel_path) if args.ucc_panel_path else None,
        listedco_parent_path=resolve_repo_path(args.listedco_parent_path) if args.listedco_parent_path else None,
        subsidiary_mapping_path=resolve_repo_path(args.subsidiary_mapping_path) if args.subsidiary_mapping_path else None,
        subjoint_csv_path=resolve_repo_path(args.subjoint_csv_path) if args.subjoint_csv_path else None,
        chunksize=args.chunksize,
        innovation_top_k=args.innovation_top_k,
        innovation_quality_cap=args.innovation_quality_cap,
        analysis_quality_threshold=args.analysis_quality_threshold,
        quality_desc_threshold=args.quality_desc_threshold,
        quality_min=args.quality_min,
        bs_min=args.bs_min,
        policy_start_year=args.policy_start_year,
        event_window=args.event_window,
        regression_year_min=args.regression_year_min,
        regression_year_max=args.regression_year_max,
    )


if __name__ == "__main__":
    main()
