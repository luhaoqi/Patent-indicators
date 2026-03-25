from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, load_manifest, repo_relative, resolve_repo_path  # noqa: E402
from run_stage2_pipeline import run_stage2  # noqa: E402


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="按 manifest 批量运行第二阶段已脚本化流程")
    parser.add_argument("--manifest", required=True, help="JSON/YAML manifest 路径")
    parser.add_argument("--output-root", help="覆盖 manifest.shared.output_root")
    parser.add_argument("--quality-threshold", type=float, help="覆盖 manifest/shared 中的高质量阈值")
    return parser


def _shared_value(shared: Dict[str, object], key: str, override):
    return override if override is not None else shared.get(key)


def main() -> None:
    args = parse_args().parse_args()
    manifest = load_manifest(args.manifest)
    shared = manifest.get("shared", {})
    experiments = manifest.get("experiments", [])
    if not experiments:
        raise ValueError("manifest 中没有 experiments")

    output_root = _shared_value(shared, "output_root", args.output_root) or "outputs/experiments"
    quality_threshold = float(_shared_value(shared, "analysis_quality_threshold", args.quality_threshold) or 1.0)
    default_raw_patent_dir = shared.get("raw_patent_dir", "data/raw/中国专利分年份保存数据1985-2025")
    default_ucc_panel_path = shared.get("ucc_panel_path")
    default_special_list_path = shared.get("special_list_path")
    default_financial_data_path = shared.get("financial_data_path")
    default_listedco_parent_path = shared.get("listedco_parent_path")
    default_subsidiary_mapping_path = shared.get("subsidiary_mapping_path")
    default_subjoint_csv_path = shared.get("subjoint_csv_path")

    batch_status: List[Dict[str, object]] = []
    for experiment in experiments:
        experiment_id = experiment["id"]
        paths = build_experiment_paths(experiment_id, output_root=output_root)
        paths.ensure_dirs()
        logger = build_logger(f"run_stage2_batch.{experiment_id}", paths.stage2_log_path())

        stage1_dir = resolve_repo_path(experiment["stage1_dir"])
        assert stage1_dir is not None
        raw_patent_dir = resolve_repo_path(experiment.get("raw_patent_dir", default_raw_patent_dir))
        ucc_panel_path = resolve_repo_path(experiment.get("ucc_panel_path", default_ucc_panel_path))
        assert raw_patent_dir is not None
        logger.info("开始运行实验: %s", experiment_id)
        summary = run_stage2(
            experiment_id=experiment_id,
            stage1_dir=stage1_dir,
            output_root=output_root,
            raw_patent_dir=raw_patent_dir,
            special_list_path=resolve_repo_path(experiment.get("special_list_path", default_special_list_path)) if experiment.get("special_list_path", default_special_list_path) else None,
            financial_data_path=resolve_repo_path(experiment.get("financial_data_path", default_financial_data_path)) if experiment.get("financial_data_path", default_financial_data_path) else None,
            ucc_panel_path=ucc_panel_path,
            listedco_parent_path=resolve_repo_path(experiment.get("listedco_parent_path", default_listedco_parent_path)) if experiment.get("listedco_parent_path", default_listedco_parent_path) else None,
            subsidiary_mapping_path=resolve_repo_path(experiment.get("subsidiary_mapping_path", default_subsidiary_mapping_path)) if experiment.get("subsidiary_mapping_path", default_subsidiary_mapping_path) else None,
            subjoint_csv_path=resolve_repo_path(experiment.get("subjoint_csv_path", default_subjoint_csv_path)) if experiment.get("subjoint_csv_path", default_subjoint_csv_path) else None,
            topk_values=experiment.get("topk_values", shared.get("topk_values", [10, 30, 50])),
            yearly_top_vocab_k=int(experiment.get("yearly_top_vocab_k", shared.get("yearly_top_vocab_k", 50))),
            max_year_gap=int(experiment.get("max_year_gap", shared.get("max_year_gap", 5))),
            analysis_quality_threshold=float(experiment.get("analysis_quality_threshold", quality_threshold)),
            quality_desc_threshold=float(experiment.get("quality_desc_threshold", shared.get("quality_desc_threshold", 5.0))),
            quality_min=float(experiment.get("quality_min", shared.get("quality_min", 1e-5))),
            bs_min=float(experiment.get("bs_min", shared.get("bs_min", 1e-6))),
            policy_start_year=int(experiment.get("policy_start_year", shared.get("policy_start_year", 2008))),
            event_window=int(experiment.get("event_window", shared.get("event_window", 5))),
            innovation_top_k=int(experiment.get("innovation_top_k", shared.get("innovation_top_k", 10))),
            innovation_quality_cap=float(experiment.get("innovation_quality_cap", shared.get("innovation_quality_cap", 1000.0))),
            regression_year_min=int(experiment.get("regression_year_min", shared.get("regression_year_min", 2000))),
            regression_year_max=int(experiment.get("regression_year_max", shared.get("regression_year_max", 2023))),
            chunksize=int(experiment.get("chunksize", shared.get("chunksize", 100000))),
        )
        batch_status.append(summary)
        logger.info("实验完成: %s", experiment_id)

    summary_root = resolve_repo_path(output_root)
    assert summary_root is not None
    summary_dir = summary_root / "_batch_summary"
    write_json(summary_dir / f"{Path(args.manifest).stem}.json", {"experiments": batch_status})


if __name__ == "__main__":
    main()
