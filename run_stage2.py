from pathlib import Path
import sys

from patent_quality.project_paths import build_experiment_layout, get_project_root


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.config import (  # noqa: E402
    DiagnosticsConfig,
    ExperimentPatentPanelConfig,
    InnovationConfig,
    QualityBasicConfig,
    RegressionConfig,
    SpecialFirmsConfig,
    Stage2Config,
    Stage2InputsConfig,
    TopPatentsByYearConfig,
)
from analysis.run_stage2_pipeline import run_stage2  # noqa: E402


EXPERIMENT_ID = "标题_摘要_window3"
OUTPUT_ROOT = "outputs/experiments"
TOPK_VALUES = (10, 30, 50)
EXCLUDE_YEARS = (1985, 1986)
SKIP_DIAGNOSTICS = False
TOP_PATENTS_PER_YEAR = 100


def main():
    layout = build_experiment_layout(EXPERIMENT_ID, output_root=OUTPUT_ROOT)
    project_root = get_project_root()
    stage2_config = Stage2Config(
        experiment_id=EXPERIMENT_ID,
        output_root=OUTPUT_ROOT,
        inputs=Stage2InputsConfig(
            stage1_dir=str(layout.stage1_dir),
            shared_root=str(project_root / "outputs/shared"),
        ),
        diagnostics=DiagnosticsConfig(
            skip=SKIP_DIAGNOSTICS,
            topk_values=TOPK_VALUES,
            yearly_top_vocab_k=50,
            max_year_gap=5,
        ),
        build_experiment_patent_panel=ExperimentPatentPanelConfig(
            chunksize=100000,
        ),
        export_top_patents_by_year=TopPatentsByYearConfig(
            top_n=TOP_PATENTS_PER_YEAR,
        ),
        analyze_quality_basic=QualityBasicConfig(
            exclude_years=EXCLUDE_YEARS,
            quality_min=1e-5,
            bs_min=1e-6,
            quality_desc_threshold=5.0,
        ),
        analyze_special_firms=SpecialFirmsConfig(
            exclude_years=EXCLUDE_YEARS,
            quality_min=1e-5,
            bs_min=1e-6,
            quality_threshold=1.0,
            policy_start_year=2008,
            event_window=5,
        ),
        build_firm_year_innovation=InnovationConfig(
            top_k=10,
            quality_cap=1000.0,
        ),
        run_regressions=RegressionConfig(
            year_min=2000,
            year_max=2023,
        ),
    )

    print("=" * 60)
    print("开始执行完整 Stage2")
    print(f"项目根目录: {project_root}")
    print(f"实验目录: {layout.root}")
    print(f"Stage1 输入目录: {layout.stage1_dir}")
    print("=" * 60)

    summary = run_stage2(
        experiment_id=stage2_config.experiment_id,
        stage1_dir=layout.stage1_dir,
        shared_root=stage2_config.inputs.shared_root,
        output_root=stage2_config.output_root,
        skip_diagnostics=stage2_config.diagnostics.skip,
        topk_values=stage2_config.diagnostics.topk_values,
        yearly_top_vocab_k=stage2_config.diagnostics.yearly_top_vocab_k,
        max_year_gap=stage2_config.diagnostics.max_year_gap,
        exclude_years=stage2_config.analyze_quality_basic.exclude_years,
        top_patents_per_year=stage2_config.export_top_patents_by_year.top_n,
        quality_min=stage2_config.analyze_quality_basic.quality_min,
        bs_min=stage2_config.analyze_quality_basic.bs_min,
        analysis_quality_threshold=stage2_config.analyze_special_firms.quality_threshold,
        quality_desc_threshold=stage2_config.analyze_quality_basic.quality_desc_threshold,
        policy_start_year=stage2_config.analyze_special_firms.policy_start_year,
        event_window=stage2_config.analyze_special_firms.event_window,
        innovation_top_k=stage2_config.build_firm_year_innovation.top_k,
        innovation_quality_cap=stage2_config.build_firm_year_innovation.quality_cap,
        regression_year_min=stage2_config.run_regressions.year_min,
        regression_year_max=stage2_config.run_regressions.year_max,
        chunksize=stage2_config.build_experiment_patent_panel.chunksize,
    )

    print("\n" + "=" * 60)
    print("Stage2 执行完毕")
    print(f"结果目录: {layout.stage2_dir}")
    print(f"已记录步骤: {', '.join(summary['steps'].keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
