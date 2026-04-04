from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from analysis.common.paths import load_manifest
from patent_quality.case_analysis import (
    PatentCaseError,
    analyze_patent_case,
    default_output_path,
    format_console_summary,
    resolve_optional_path,
    search_stage1_index,
    search_stage1_tokens,
    write_analysis_json,
)
from patent_quality.project_paths import (
    build_experiment_layout,
    infer_experiment_id_from_stage1_dir,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按单个专利追踪 stage1 词项保留/舍弃情况")
    parser.add_argument("--stage1-dir", help="stage1 输出目录")
    parser.add_argument("--experiment-id", help="实验 ID，不传 stage1-dir 时使用")
    parser.add_argument("--cases-manifest", help="JSON/YAML 批量 case 配置路径")
    parser.add_argument("--output-root", default="outputs/experiments", help="实验输出根目录")
    parser.add_argument(
        "--raw-data-path",
        default="data/raw/中国专利分年份保存数据1985-2025",
        help="原始专利 CSV 目录或单个 CSV 路径",
    )
    parser.add_argument(
        "--config-script",
        default="run_full.py",
        help="用于解析 Config(...) 字面量参数的脚本路径；分析其他实验时请指向对应入口脚本",
    )
    parser.add_argument("--application-no", help="申请号，最推荐")
    parser.add_argument("--application-year", type=int, help="申请年份，强烈建议提供以减少扫描")
    parser.add_argument("--title", help="专利名称精确匹配")
    parser.add_argument("--title-contains", help="专利名称模糊匹配")
    parser.add_argument("--publication-date", help="公开（公告）日，支持 YYYY 或 YYYY-MM-DD")
    parser.add_argument("--output-json", help="显式指定 JSON 输出路径")
    parser.add_argument("--top-terms", type=int, default=15, help="终端摘要展示的词项数量")
    parser.add_argument("--skip-raw-cut", action="store_true", help="跳过原始文本 jieba.cut 结果统计")
    parser.add_argument(
        "--expand-year-search",
        action="store_true",
        help="当给定年份查不到时，继续扩展到 stage1 全部年份查找",
    )
    parser.add_argument("--yes-stage2", action="store_true", help="第 1 步失败后自动同意继续第 2 步")
    parser.add_argument("--yes-stage3", action="store_true", help="第 2 步失败后自动同意继续第 3 步")

    parser.add_argument("--min-term-count", type=int, help="覆盖 min_term_count")
    parser.add_argument("--max-doc-freq-ratio", type=float, help="覆盖 max_doc_freq_ratio")
    parser.add_argument("--manual-stopwords-path", help="覆盖 manual_stopwords_path")
    parser.add_argument("--df-ratio-threshold", type=float, help="覆盖 df_ratio_threshold")
    parser.add_argument("--top-df-percent", type=float, help="覆盖 top_df_percent")
    parser.add_argument("--topk-terms-per-doc", type=int, help="覆盖 topk_terms_per_doc")
    parser.add_argument("--user-dict-path", help="覆盖 user_dict_path")
    parser.add_argument("--stopword-path", action="append", dest="stopword_paths", help="覆盖 stopword_paths，可重复传入")
    parser.add_argument("--text-part", action="append", dest="col_text_parts", help="覆盖 col_text_parts，可重复传入")
    return parser.parse_args()


def resolve_stage1_dir(args: argparse.Namespace) -> tuple[str, Path]:
    if args.stage1_dir:
        stage1_dir = resolve_project_path(args.stage1_dir)
        experiment_id = args.experiment_id or infer_experiment_id_from_stage1_dir(stage1_dir)
        return experiment_id, stage1_dir
    if not args.experiment_id:
        raise PatentCaseError("必须提供 --stage1-dir 或 --experiment-id")
    layout = build_experiment_layout(args.experiment_id, output_root=args.output_root)
    return args.experiment_id, layout.stage1_dir


def build_overrides(args: argparse.Namespace) -> dict:
    return {
        "min_term_count": args.min_term_count,
        "max_doc_freq_ratio": args.max_doc_freq_ratio,
        "manual_stopwords_path": args.manual_stopwords_path,
        "df_ratio_threshold": args.df_ratio_threshold,
        "top_df_percent": args.top_df_percent,
        "topk_terms_per_doc": args.topk_terms_per_doc,
        "user_dict_path": args.user_dict_path,
        "stopword_paths": args.stopword_paths,
        "col_text_parts": args.col_text_parts,
    }


def _confirm_continue(question: str, *, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"[AUTO] {question} -> yes")
        return True
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _print_log_header(title: str) -> None:
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_lookup_log(result: dict) -> None:
    print(f"stage: {result['stage']}")
    print(f"restrict_to_year: {result['restrict_to_year']}")
    print(f"input_year: {result['input_year']}")
    print(f"scanned_file_count: {result['scanned_file_count']}")
    print(f"rows_scanned: {result['rows_scanned']}")
    print(f"match_count: {result['match_count']}")
    if result["scanned_files"]:
        print("scanned_files:")
        for path in result["scanned_files"]:
            print(f"- {path}")
    if result["matches"]:
        print("matches:")
        for match in result["matches"][:5]:
            if hasattr(match, "row_index"):
                print(
                    f"- application_no={match.application_no} year={match.application_year} "
                    f"row={match.row_index} title={match.title}"
                )
            else:
                print(
                    f"- application_no={match.application_no} year={match.application_year} "
                    f"line={match.line_index} token_count={match.token_count} title={match.title}"
                )


def run_staged_lookup(args: argparse.Namespace, stage1_dir: Path) -> tuple[int, int | None]:
    step1 = search_stage1_index(
        stage1_dir=stage1_dir,
        application_no=args.application_no,
        application_year=args.application_year,
        title=args.title,
        title_contains=args.title_contains,
        restrict_to_year=True,
    )
    _print_log_header("Step 1: Check stage1/index With Input Year")
    _print_lookup_log(step1)
    if step1["matches"]:
        return 1, step1["matches"][0].application_year

    if not _confirm_continue("Step 1 未命中，是否继续 Step 2: 检查同年份 stage1/tokens？", auto_yes=args.yes_stage2):
        raise PatentCaseError("用户在 Step 1 后停止。")

    step2 = search_stage1_tokens(
        stage1_dir=stage1_dir,
        application_no=args.application_no,
        application_year=args.application_year,
        title=args.title,
        title_contains=args.title_contains,
        restrict_to_year=True,
    )
    _print_log_header("Step 2: Check stage1/tokens With Input Year")
    _print_lookup_log(step2)
    if step2["matches"]:
        raise PatentCaseError(
            "Step 2 命中：该专利进入了分词阶段，但没有进入同年份 stage1/index。"
            "这通常意味着后续没有形成非空向量。"
        )

    if not _confirm_continue("Step 2 未命中，是否继续 Step 3: 扩展到全部年份查找？", auto_yes=args.yes_stage3):
        raise PatentCaseError("用户在 Step 2 后停止。")

    _print_log_header("Step 3A: Expand Search To All Years stage1/index")
    step3_index = search_stage1_index(
        stage1_dir=stage1_dir,
        application_no=args.application_no,
        application_year=args.application_year,
        title=args.title,
        title_contains=args.title_contains,
        restrict_to_year=False,
    )
    _print_lookup_log(step3_index)
    if step3_index["matches"]:
        return 3, step3_index["matches"][0].application_year

    _print_log_header("Step 3B: Expand Search To All Years stage1/tokens")
    step3_tokens = search_stage1_tokens(
        stage1_dir=stage1_dir,
        application_no=args.application_no,
        application_year=args.application_year,
        title=args.title,
        title_contains=args.title_contains,
        restrict_to_year=False,
    )
    _print_lookup_log(step3_tokens)
    if step3_tokens["matches"]:
        raise PatentCaseError(
            "Step 3B 命中：该专利在其他年份进入了分词阶段，但没有进入 stage1/index。"
            "说明它存在于 stage1 前处理中，但没有进入最终向量索引。"
        )

    raise PatentCaseError("Step 3 结束：stage1/index 和 stage1/tokens 全年份均未命中。")


def main() -> int:
    args = parse_args()
    return run_entry(args)


def run_entry(args: argparse.Namespace) -> int:
    if args.cases_manifest:
        return run_batch_cases(args)
    return run_single_case(args)


def run_single_case(args: argparse.Namespace) -> int:
    if not any([args.application_no, args.title, args.title_contains]):
        raise PatentCaseError("至少需要提供 --application-no、--title 或 --title-contains 之一")

    experiment_id, stage1_dir = resolve_stage1_dir(args)
    raw_data_path = resolve_project_path(args.raw_data_path)
    config_script = resolve_optional_path(args.config_script)
    lookup_stage, resolved_year = run_staged_lookup(args, stage1_dir)

    result = analyze_patent_case(
        stage1_dir=stage1_dir,
        raw_data_path=raw_data_path,
        application_no=args.application_no,
        application_year=resolved_year or args.application_year,
        title=args.title,
        title_contains=args.title_contains,
        publication_date=args.publication_date,
        config_script=config_script,
        config_overrides=build_overrides(args),
        include_raw_cut=not args.skip_raw_cut,
        expand_year_search=lookup_stage >= 3 or args.expand_year_search,
    )

    output_json = resolve_optional_path(args.output_json)
    if output_json is None:
        output_json = default_output_path(
            stage1_dir=stage1_dir,
            experiment_id=experiment_id,
            application_no=result["patent"]["application_no"],
        )
    write_analysis_json(result, output_json)

    _print_log_header("Analysis Summary")
    print(format_console_summary(result, top_terms=args.top_terms))
    print("")
    print(f"JSON saved to: {output_json}")
    return 0


def _merge_case_args(base_args: argparse.Namespace, shared: dict, case: dict) -> argparse.Namespace:
    data = vars(base_args).copy()
    data.update(shared)
    data.update(case)
    data["cases_manifest"] = None
    return SimpleNamespace(**data)


def run_batch_cases(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.cases_manifest)
    shared = manifest.get("shared", {})
    cases = manifest.get("cases", [])
    if not cases:
        raise PatentCaseError("cases_manifest 中没有 cases")

    failures = 0
    for index, case in enumerate(cases, start=1):
        case_args = _merge_case_args(args, shared, case)
        _print_log_header(
            f"Batch Case {index}/{len(cases)}: "
            f"{case.get('application_no') or case.get('title') or case.get('title_contains')}"
        )
        try:
            run_single_case(case_args)
        except PatentCaseError as exc:
            failures += 1
            print(f"ERROR: {exc}")

    if failures:
        raise PatentCaseError(f"批量分析完成，但有 {failures} 个 case 失败。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatentCaseError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
