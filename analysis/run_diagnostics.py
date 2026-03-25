from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.diagnostics import run_diagnostics  # noqa: E402
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, repo_relative, resolve_repo_path  # noqa: E402


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="运行第一阶段 diagnostics 并标准化输出到 experiment 目录")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--stage1-dir", required=True, help="第一阶段结果目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--topk-values", nargs="*", type=int, default=[10, 30, 50], help="TopK diagnostics 列表")
    parser.add_argument("--yearly-top-vocab-k", type=int, default=50, help="每年高频词输出数量")
    parser.add_argument("--max-year-gap", type=int, default=5, help="年份对过滤窗口")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    stage1_dir = resolve_repo_path(args.stage1_dir)
    assert stage1_dir is not None
    paths = build_experiment_paths(args.experiment_id, output_root=args.output_root)
    paths.ensure_dirs()
    logger = build_logger(f"run_diagnostics.{args.experiment_id}", paths.logs_dir / "run_diagnostics.log")
    written_paths = run_diagnostics(
        stage1_dir=stage1_dir,
        diagnostics_dir=paths.diagnostics_dir,
        topk_values=args.topk_values,
        yearly_top_vocab_k=args.yearly_top_vocab_k,
        max_year_gap=args.max_year_gap,
        logger=logger,
    )
    write_json(
        paths.metadata_dir / "run_diagnostics.json",
        {
            "experiment_id": args.experiment_id,
            "stage1_dir": repo_relative(stage1_dir),
            "outputs": [repo_relative(path) for path in written_paths],
        },
    )
    logger.info("diagnostics 已输出 %s 个文件", len(written_paths))


if __name__ == "__main__":
    main()
