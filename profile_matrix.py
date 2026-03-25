import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from patent_quality.project_paths import (
    build_experiment_layout,
    infer_experiment_id_from_stage1_dir,
    resolve_project_path,
)


def profile_year(matrix_path: Path) -> dict:
    if not matrix_path.exists():
        raise FileNotFoundError(f"File not found: {matrix_path}")

    matrix = sparse.load_npz(matrix_path).tocsr()
    n_rows = matrix.shape[0]
    row_lens = np.diff(matrix.indptr)

    if n_rows == 0:
        return {"n_rows": 0, "message": "empty matrix"}

    histogram_bins = [0, 1, 5, 10, 20, 50, 100, 1000]
    counts, bins = np.histogram(row_lens, bins=histogram_bins)
    distribution = [
        {
            "bin_start": int(bins[idx]),
            "bin_end": int(bins[idx + 1]),
            "count": int(counts[idx]),
            "ratio": float(counts[idx] / n_rows),
        }
        for idx in range(len(counts))
    ]
    return {
        "n_rows": int(n_rows),
        "avg_terms": float(np.mean(row_lens)),
        "p50": float(np.percentile(row_lens, 50)),
        "p90": float(np.percentile(row_lens, 90)),
        "p99": float(np.percentile(row_lens, 99)),
        "max_terms": int(np.max(row_lens)),
        "distribution": distribution,
    }


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计单年向量矩阵稀疏度并输出到统一验证目录")
    parser.add_argument("year", type=int, nargs="?", default=2011, help="目标年份")
    parser.add_argument("--experiment-id", help="实验 ID")
    parser.add_argument("--stage1-dir", help="第一阶段输出目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--output-json", help="显式指定 JSON 输出路径")
    parser.add_argument(
        "--matrix-kind",
        choices=["vectors", "vectors_filtered"],
        default="vectors_filtered",
        help="选择统计原始向量还是剪枝后向量",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    stage1_dir = resolve_project_path(args.stage1_dir) if args.stage1_dir else None
    experiment_id = args.experiment_id or infer_experiment_id_from_stage1_dir(stage1_dir or "baseline_1985_2025_window5")
    layout = build_experiment_layout(experiment_id, output_root=args.output_root)
    layout.ensure_verification_dirs()
    source_stage1_dir = stage1_dir or layout.stage1_dir

    matrix_path = source_stage1_dir / args.matrix_kind / f"year={args.year}.npz"
    stats = profile_year(matrix_path)
    stats.update(
        {
            "experiment_id": experiment_id,
            "source_matrix": str(matrix_path),
            "matrix_kind": args.matrix_kind,
            "year": args.year,
        }
    )

    output_json = (
        resolve_project_path(args.output_json)
        if args.output_json
        else layout.verification_dir / "matrix" / f"matrix_profile_year={args.year}_{args.matrix_kind}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Year {args.year} Stats ({args.matrix_kind}):")
    print(f"  Rows: {stats['n_rows']}")
    if stats["n_rows"]:
        print(f"  Avg Terms: {stats['avg_terms']:.2f}")
        print(f"  Median (P50): {stats['p50']:.2f}")
        print(f"  P90: {stats['p90']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")
        print(f"  Max: {stats['max_terms']}")
    print(f"JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
