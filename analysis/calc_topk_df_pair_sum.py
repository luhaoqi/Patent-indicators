from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.diagnostics import compute_topk_pair_sum  # noqa: E402
from common.paths import build_experiment_paths, resolve_repo_path  # noqa: E402
from patent_quality.project_paths import infer_experiment_id_from_stage1_dir  # noqa: E402


def _resolve_stage1_dir(args) -> Path:
    if args.stage1_dir:
        resolved = resolve_repo_path(args.stage1_dir)
        assert resolved is not None
        return resolved
    if args.dir:
        resolved = resolve_repo_path(args.dir)
        assert resolved is not None
        return resolved
    if args.experiment_id:
        return build_experiment_paths(args.experiment_id, output_root=args.output_root).stage1_dir
    raise ValueError("请提供 --stage1-dir、位置参数 dir 或 --experiment-id")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 TopK 过滤后的词汇 DF Pairwise Sum")
    parser.add_argument("dir", nargs="?", help="兼容旧用法：stage1 目录")
    parser.add_argument("--stage1-dir", help="第一阶段输出目录")
    parser.add_argument("--experiment-id", help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--k", type=int, default=10, help="每个文档保留的 TopK 词汇数")
    parser.add_argument("--output", help="显式指定 pairwise 输出 CSV")
    parser.add_argument("--weights-output", help="显式指定 yearly weight stats 输出 CSV")
    parser.add_argument("--max-year-gap", type=int, default=5, help="年份对最大间隔")
    args = parser.parse_args()

    stage1_dir = _resolve_stage1_dir(args)
    experiment_id = args.experiment_id or infer_experiment_id_from_stage1_dir(stage1_dir)
    diagnostics_dir = build_experiment_paths(experiment_id, output_root=args.output_root).diagnostics_dir
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    pairwise_path = resolve_repo_path(args.output) if args.output else diagnostics_dir / f"topk_df_pair_sum_k{args.k}.csv"
    weights_path = resolve_repo_path(args.weights_output) if args.weights_output else diagnostics_dir / f"topk_weight_stats_k{args.k}.csv"
    assert pairwise_path is not None
    assert weights_path is not None
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = compute_topk_pair_sum(stage1_dir, topk=args.k, max_year_gap=args.max_year_gap)
    outputs["pairwise"].to_csv(pairwise_path, index=False)
    outputs["yearly"].to_csv(weights_path, index=False)
    print(f"pairwise saved to: {pairwise_path}")
    print(f"weight stats saved to: {weights_path}")


if __name__ == "__main__":
    main()
