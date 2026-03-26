from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time
from typing import Any, Dict, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analyze_quality_basic import analyze_quality_basic  # noqa: E402
from analyze_special_firms import analyze_special_firms  # noqa: E402
from build_firm_year_innovation import build_firm_year_innovation  # noqa: E402
from build_main_enriched import build_experiment_patent_panel  # noqa: E402
from common.config import Stage2Config  # noqa: E402
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402
from run_regressions import run_regressions  # noqa: E402


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    return path


def run_stage2(
    *,
    experiment_id: str,
    stage1_dir: Path,
    shared_root: str = "outputs/shared",
    output_root: str = "outputs/experiments",
    skip_diagnostics: bool = False,
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
    shared_paths = build_shared_paths(shared_root)

    stage1_output = _require_file(stage1_dir / "patent_quality_output.csv", "stage1 patent_quality_output.csv")
    patent_master_path = _require_file(shared_paths.patent_master_dir / "patent_master.parquet", "shared patent_master")
    firm_year_special_labels_path = _require_file(
        shared_paths.special_firm_labels_dir / "firm_year_special_labels.parquet",
        "shared firm_year_special_labels",
    )
    special_ucc_set_path = _require_file(
        shared_paths.special_firm_labels_dir / "special_ucc_set.parquet",
        "shared special_ucc_set",
    )
    ucc_exploded_path = _require_file(shared_paths.ucc_mapping_dir / "ucc_exploded.parquet", "shared ucc_exploded")
    financial_panel_path = _require_file(
        shared_paths.financial_panel_dir / "financial_annual_clean.parquet",
        "shared financial_annual_clean",
    )

    logger = build_logger(f"run_stage2_pipeline.{experiment_id}", paths.stage2_log_path())
    stage2_config = Stage2Config.from_runtime(
        experiment_id=experiment_id,
        stage1_dir=stage1_dir,
        shared_root=shared_paths.root,
        output_root=output_root,
        skip_diagnostics=skip_diagnostics,
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
        notes={
            "shared_inputs": {
                "patent_master_path": repo_relative(patent_master_path),
                "firm_year_special_labels_path": repo_relative(firm_year_special_labels_path),
                "special_ucc_set_path": repo_relative(special_ucc_set_path),
                "ucc_exploded_path": repo_relative(ucc_exploded_path),
                "financial_panel_path": repo_relative(financial_panel_path),
            }
        },
    )
    write_json(paths.metadata_dir / "stage2_config.json", stage2_config.to_payload())

    logger.info("Stage2 开始: experiment_id=%s", experiment_id)
    logger.info("stage1_dir=%s", repo_relative(stage1_dir))
    logger.info("shared_root=%s", repo_relative(shared_paths.root))

    step_summaries: Dict[str, Any] = {}

    if skip_diagnostics:
        logger.info("[1/6] 跳过 diagnostics")
        step_summaries["diagnostics"] = {"skipped": True}
    else:
        logger.info("[1/6] 运行 diagnostics")
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
        logger.info("[1/6] diagnostics 完成，用时 %.1fs，输出 %s 个文件", time.perf_counter() - step_start, len(diagnostics_written))

    logger.info("[2/6] 构造 experiment_patent_panel")
    step_start = time.perf_counter()
    experiment_panel_result = build_experiment_patent_panel(
        experiment_id=experiment_id,
        stage1_output_path=stage1_output,
        output_root=output_root,
        patent_master_path=patent_master_path,
        shared_root=shared_root,
    )
    experiment_patent_panel_path = experiment_panel_result["experiment_patent_panel_path"]
    step_summaries["build_experiment_patent_panel"] = {
        key: repo_relative(value) if isinstance(value, Path) else value
        for key, value in experiment_panel_result.items()
    }
    logger.info("[2/6] build_experiment_patent_panel 完成，用时 %.1fs", time.perf_counter() - step_start)

    logger.info("[3/6] 输出基础图表与描述统计")
    step_start = time.perf_counter()
    basic_summary = analyze_quality_basic(
        experiment_id=experiment_id,
        output_root=output_root,
        experiment_patent_panel_path=experiment_patent_panel_path,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        quality_desc_threshold=quality_desc_threshold,
    )
    step_summaries["analyze_quality_basic"] = basic_summary
    logger.info("[3/6] analyze_quality_basic 完成，用时 %.1fs", time.perf_counter() - step_start)

    logger.info("[4/6] 输出特殊企业对比分析")
    step_start = time.perf_counter()
    special_summary = analyze_special_firms(
        experiment_id=experiment_id,
        output_root=output_root,
        experiment_patent_panel_path=experiment_patent_panel_path,
        firm_year_special_labels_path=firm_year_special_labels_path,
        special_ucc_set_path=special_ucc_set_path,
        shared_root=shared_root,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        quality_threshold=analysis_quality_threshold,
        policy_start_year=policy_start_year,
        event_window=event_window,
    )
    step_summaries["analyze_special_firms"] = special_summary
    logger.info("[4/6] analyze_special_firms 完成，用时 %.1fs", time.perf_counter() - step_start)

    logger.info("[5/6] 构造 firm_year_innovation")
    step_start = time.perf_counter()
    innovation_path = build_firm_year_innovation(
        experiment_id=experiment_id,
        output_root=output_root,
        experiment_patent_panel_path=experiment_patent_panel_path,
        ucc_exploded_path=ucc_exploded_path,
        shared_root=shared_root,
        top_k=innovation_top_k,
        quality_cap=innovation_quality_cap,
    )
    step_summaries["build_firm_year_innovation"] = repo_relative(innovation_path)
    logger.info("[5/6] build_firm_year_innovation 完成，用时 %.1fs", time.perf_counter() - step_start)

    logger.info("[6/6] 运行固定效应回归")
    step_start = time.perf_counter()
    regression_summary = run_regressions(
        experiment_id=experiment_id,
        output_root=output_root,
        firm_year_innovation_path=innovation_path,
        financial_panel_path=financial_panel_path,
        shared_root=shared_root,
        year_min=regression_year_min,
        year_max=regression_year_max,
    )
    step_summaries["run_regressions"] = regression_summary
    logger.info("[6/6] run_regressions 完成，用时 %.1fs", time.perf_counter() - step_start)

    summary = {
        "experiment_id": experiment_id,
        "stage1_dir": repo_relative(stage1_dir),
        "shared_root": repo_relative(shared_paths.root),
        "output_root": output_root,
        "config_path": repo_relative(paths.metadata_dir / "stage2_config.json"),
        "steps": step_summaries,
    }
    write_json(paths.metadata_dir / "run_stage2_pipeline.json", summary)
    logger.info("Stage2 完成: %s", experiment_id)
    return summary


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="运行单个实验的严格 stage2 流程")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--stage1-dir", required=True, help="stage1 结果目录")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--skip-diagnostics", action="store_true", help="跳过 diagnostics 步骤")
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
    parser.add_argument("--chunksize", type=int, default=100000, help="仅写入 metadata 的 patent_master 构造 chunksize")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    stage1_dir = resolve_repo_path(args.stage1_dir)
    assert stage1_dir is not None
    run_stage2(
        experiment_id=args.experiment_id,
        stage1_dir=stage1_dir,
        shared_root=args.shared_root,
        output_root=args.output_root,
        skip_diagnostics=args.skip_diagnostics,
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
