from __future__ import annotations

import argparse
import sys

from patent_quality.project_paths import resolve_project_path
from patent_quality.similarity_case_analysis import (
    DEFAULT_BOTTOM_N,
    DEFAULT_TOP_N,
    PatentSimilarityCaseError,
    resolve_stage1_dir,
    run_similarity_case_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按单个专利展开 exact-stage1 最终词项贡献，并导出前后窗口专利相似度表。"
    )
    parser.add_argument("--stage1-dir", help="推荐直接传 stage1_exact 输出目录")
    parser.add_argument("--experiment-id", help="exact 实验 ID；不传 stage1-dir 时使用")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--application-no", required=True, help="目标专利申请号")
    parser.add_argument("--year", required=True, type=int, help="公开公告年份")
    parser.add_argument("--date", help="公开公告日；只有同申请号同年份无法唯一定位时才需要")
    parser.add_argument("--title", help="专利名称；只有同申请号同年份无法唯一定位时才需要")
    parser.add_argument("--k", type=int, help="时间窗口大小；不传则优先从 exact 实验产物推断")
    parser.add_argument("--similarity-threshold", type=float, help="相似度阈值；不传则优先从 exact 实验产物推断")
    parser.add_argument("--output-dir", help="输出目录；默认写到实验 verification/patent_similarity_case 下")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="候选很多时保留前多少条")
    parser.add_argument("--bottom-n", type=int, default=DEFAULT_BOTTOM_N, help="候选很多时保留后多少条")
    return parser.parse_args()


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def main() -> int:
    _configure_stdio()
    args = parse_args()
    try:
        stage1_dir = resolve_stage1_dir(
            stage1_dir=args.stage1_dir,
            experiment_id=args.experiment_id,
            output_root=args.output_root,
        )
        output_dir = resolve_project_path(args.output_dir) if args.output_dir else None
        summary = run_similarity_case_analysis(
            stage1_dir=stage1_dir,
            application_no=args.application_no,
            year=args.year,
            date_value=args.date,
            title=args.title,
            window_size=args.k,
            similarity_threshold=args.similarity_threshold,
            output_dir=output_dir,
            top_n=args.top_n,
            bottom_n=args.bottom_n,
        )
    except PatentSimilarityCaseError as exc:
        print(f"[error] {exc}")
        return 1

    print(
        f"[target] application_no={summary['target']['application_no']} "
        f"year={summary['target']['year']} row={summary['target']['row']} "
        f"title={summary['target']['title']}"
    )
    if summary["target"].get("date_value"):
        print(f"[target] date={summary['target']['date_value']}")
    print(
        f"[config] window_size={summary['window_size']} "
        f"similarity_threshold={summary['similarity_threshold']:.10f}"
    )
    print(
        f"[tokens] stage1_token_count={summary['target_stage1_token_count']} "
        f"unique={summary['target_stage1_unique_token_count']} "
        f"final_vector_terms={summary['target_final_vector_term_count']}"
    )
    print(
        f"[window] backward_candidates={summary['backward_candidate_count']} "
        f"forward_candidates={summary['forward_candidate_count']}"
    )
    print(f"[output] term_contribution_csv={summary['term_contribution_csv']}")
    print(f"[output] backward_similarity_csv={summary['backward_similarity_csv']}")
    print(f"[output] forward_similarity_csv={summary['forward_similarity_csv']}")
    print(f"[output] summary_json={summary['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
