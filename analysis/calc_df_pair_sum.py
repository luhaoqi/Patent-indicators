from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.diagnostics import compute_df_pair_sum  # noqa: E402
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
    parser = argparse.ArgumentParser(description="统计任意两年词汇 DF 乘积之和")
    parser.add_argument("dir", nargs="?", help="兼容旧用法：stage1 目录")
    parser.add_argument("--stage1-dir", help="第一阶段输出目录")
    parser.add_argument("--experiment-id", help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--output", help="显式指定输出 CSV")
    parser.add_argument("--max-year-gap", type=int, default=5, help="年份对最大间隔")
    args = parser.parse_args()

    stage1_dir = _resolve_stage1_dir(args)
    experiment_id = args.experiment_id or infer_experiment_id_from_stage1_dir(stage1_dir)
    output_path = resolve_repo_path(args.output) if args.output else build_experiment_paths(experiment_id, output_root=args.output_root).diagnostics_dir / "df_pair_sum.csv"
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = compute_df_pair_sum(stage1_dir, max_year_gap=args.max_year_gap)
    frame.to_csv(output_path, index=False)
    print(f"df pair sum saved to: {output_path}")


if __name__ == "__main__":
    main()
