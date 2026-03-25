from pathlib import Path
import sys

from patent_quality.project_paths import build_experiment_layout, get_project_root


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.config import (  # noqa: E402
    DiagnosticsConfig,
    InnovationConfig,
    MainEnrichedConfig,
    QualityBasicConfig,
    RegressionConfig,
    SpecialFirmsConfig,
    Stage2Config,
    Stage2InputsConfig,
)
from analysis.run_stage2_pipeline import run_stage2  # noqa: E402


EXPERIMENT_ID = "标题_摘要_window5"
OUTPUT_ROOT = "outputs/experiments"
TOPK_VALUES = (10, 30, 50)
EXCLUDE_YEARS = (1985, 1986)


def main():
    layout = build_experiment_layout(EXPERIMENT_ID, output_root=OUTPUT_ROOT)
    project_root = get_project_root()
    stage2_config = Stage2Config(
        experiment_id=EXPERIMENT_ID,
        output_root=OUTPUT_ROOT,
        inputs=Stage2InputsConfig(
            stage1_dir=str(layout.stage1_dir),
            raw_patent_dir=str(project_root / "data/raw/中国专利分年份保存数据1985-2025"),
            special_list_path=str(project_root / "analysis/graph/科创企业名单2024.dta"),
            financial_data_path=str(project_root / "analysis/公司财务/数据/上市公司财务数据/上市公司财务数据.dta"),
            ucc_panel_path=str(project_root / "analysis/公司财务/数据/上市公司（包括所有子公司）各年度的统一社会信用代码列表.csv"),
            listedco_parent_path=str(project_root / "analysis/公司财务/数据/上市公司基本信息年度表/上市公司统一社会信用代码.csv"),
            subsidiary_mapping_path=str(project_root / "analysis/公司财务/数据/爱企查结果/上市公司子公司对应统一社会信用代码.csv"),
            subjoint_csv_path=str(project_root / "analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint_merged.csv"),
        ),
        diagnostics=DiagnosticsConfig(
            topk_values=TOPK_VALUES,
            yearly_top_vocab_k=50,
            max_year_gap=5,
        ),
        build_main_enriched=MainEnrichedConfig(
            chunksize=100000,
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
        raw_patent_dir=Path(stage2_config.inputs.raw_patent_dir),
        output_root=stage2_config.output_root,
        special_list_path=Path(stage2_config.inputs.special_list_path) if stage2_config.inputs.special_list_path else None,
        financial_data_path=Path(stage2_config.inputs.financial_data_path) if stage2_config.inputs.financial_data_path else None,
        ucc_panel_path=Path(stage2_config.inputs.ucc_panel_path) if stage2_config.inputs.ucc_panel_path else None,
        listedco_parent_path=Path(stage2_config.inputs.listedco_parent_path) if stage2_config.inputs.listedco_parent_path else None,
        subsidiary_mapping_path=Path(stage2_config.inputs.subsidiary_mapping_path) if stage2_config.inputs.subsidiary_mapping_path else None,
        subjoint_csv_path=Path(stage2_config.inputs.subjoint_csv_path) if stage2_config.inputs.subjoint_csv_path else None,
        topk_values=stage2_config.diagnostics.topk_values,
        yearly_top_vocab_k=stage2_config.diagnostics.yearly_top_vocab_k,
        max_year_gap=stage2_config.diagnostics.max_year_gap,
        exclude_years=stage2_config.analyze_quality_basic.exclude_years,
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
        chunksize=stage2_config.build_main_enriched.chunksize,
    )

    print("\n" + "=" * 60)
    print("Stage2 执行完毕")
    print(f"结果目录: {layout.stage2_dir}")
    print(f"已记录步骤: {', '.join(summary['steps'].keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
