from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re
import sys
from typing import Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common.analysis import (  # noqa: E402
    INVALID_UCC_VALUES,
    PATENT_UCC_COL,
    PATENT_YEAR_COL,
    QUALITY_COL,
    attach_special_year_labels,
    build_abc_summary_table,
    build_company_special_panel_from_ucc_set,
    build_company_year_abc_panel,
    build_company_year_special_panel,
    build_event_study_frame,
    build_group_comparison_table,
    filter_patents,
    normalize_string_series,
    resolve_patent_year_col,
    to_numeric,
)
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402
from common.plotting import save_figure, set_chinese_font  # noqa: E402
from common.tables import export_table  # noqa: E402
from special_firm_regressions import run_special_firm_regressions  # noqa: E402


ABC_GROUPS = ["A_treated_year", "B_same_firm_other_year", "C_never_treated"]
ABC_LABELS = {
    "A_treated_year": "A Treated-year",
    "B_same_firm_other_year": "B Same-firm other-year",
    "C_never_treated": "C Never-treated",
}
OUTPUT_CATEGORY = "特殊企业对比"
FILTERED_OUTPUT_CATEGORY = "特殊企业_过滤部分单位"
APPLICANT_COL = "申请人"
UCC_YEAR_TOP_N = 1000
DEFAULT_UNIT_FILTER_TERMS_PATH = "高校_研究所过滤词.txt"


def analyze_special_firms(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    experiment_patent_panel_path: Optional[Path] = None,
    firm_year_special_labels_path: Optional[Path] = None,
    special_ucc_set_path: Optional[Path] = None,
    shared_root: str = "outputs/shared",
    unit_filter_terms_path: Optional[Path] = None,
    exclude_years: Sequence[int] = (1985, 1986),
    quality_min: float = 1e-5,
    bs_min: float = 1e-6,
    quality_threshold: float = 1.0,
    regression_topk_share: float = 0.10,
    policy_start_year: int = 2008,
    event_window: int = 5,
    exact_date: bool = False,
) -> dict[str, object]:
    paths = build_experiment_paths(experiment_id, output_root=output_root, exact_date=exact_date)
    paths.ensure_dirs()
    logger = build_logger(f"analyze_special_firms.{experiment_id}", paths.logs_dir / "analyze_special_firms.log")
    set_chinese_font(logger=logger)

    patent_path = experiment_patent_panel_path or (paths.data_dir / "experiment_patent_panel.parquet")
    if not patent_path.exists():
        raise FileNotFoundError(f"找不到 experiment_patent_panel: {patent_path}")
    patent_df = pd.read_parquet(patent_path)
    year_col = resolve_patent_year_col(patent_df.columns, exact_date=exact_date)
    effective_firm_year_special_path = firm_year_special_labels_path
    effective_special_ucc_path = special_ucc_set_path
    if effective_firm_year_special_path is None or effective_special_ucc_path is None:
        shared_paths = build_shared_paths(shared_root)
        if effective_firm_year_special_path is None:
            candidate = shared_paths.special_firm_labels_dir / "firm_year_special_labels.parquet"
            if candidate.exists():
                effective_firm_year_special_path = candidate
        if effective_special_ucc_path is None:
            candidate = shared_paths.special_firm_labels_dir / "special_ucc_set.parquet"
            if candidate.exists():
                effective_special_ucc_path = candidate

    if effective_firm_year_special_path is None or effective_special_ucc_path is None:
        raise FileNotFoundError("缺少共享特殊企业标签，请先运行 run_shared_prep.py 生成 shared special_firm_labels")
    firm_year_special = pd.read_parquet(effective_firm_year_special_path)
    if year_col != PATENT_YEAR_COL and PATENT_YEAR_COL in firm_year_special.columns and year_col not in firm_year_special.columns:
        firm_year_special = firm_year_special.rename(columns={PATENT_YEAR_COL: year_col})
    special_uccs = (
        pd.read_parquet(effective_special_ucc_path)[PATENT_UCC_COL]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    special_source = repo_relative(effective_firm_year_special_path)
    special_ucc_source = repo_relative(effective_special_ucc_path)

    logger.info("读取 patent_df=%s 行, firm_year_special=%s 行", len(patent_df), len(firm_year_special))

    logger.info("开始过滤专利样本")
    filtered_patents = filter_patents(
        patent_df,
        year_col=year_col,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
    )
    logger.info("基础过滤后专利样本量: %s", len(filtered_patents))

    effective_unit_filter_terms_path = resolve_repo_path(unit_filter_terms_path or DEFAULT_UNIT_FILTER_TERMS_PATH)
    if effective_unit_filter_terms_path is None or not effective_unit_filter_terms_path.exists():
        raise FileNotFoundError(f"找不到过滤词文件: {unit_filter_terms_path or DEFAULT_UNIT_FILTER_TERMS_PATH}")
    unit_filter_terms = _load_unit_filter_terms(effective_unit_filter_terms_path)
    filtered_variant_patents, unit_filter_stats = _exclude_patents_by_applicant_terms(
        filtered_patents,
        terms=unit_filter_terms,
    )
    logger.info(
        "部分单位过滤完成: 基线样本=%s, 剔除=%s, 保留=%s, 过滤词文件=%s",
        unit_filter_stats["input_rows"],
        unit_filter_stats["excluded_rows"],
        unit_filter_stats["kept_rows"],
        repo_relative(effective_unit_filter_terms_path),
    )

    baseline_summary = _run_special_firm_variant(
        paths=paths,
        category=OUTPUT_CATEGORY,
        patent_df=filtered_patents,
        year_col=year_col,
        firm_year_special=firm_year_special,
        special_uccs=special_uccs,
        quality_threshold=quality_threshold,
        regression_topk_share=regression_topk_share,
        policy_start_year=policy_start_year,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        event_window=event_window,
        logger=logger,
        unit_filter_terms=[],
        unit_filter_terms_path=None,
        sample_context={
            "variant": "baseline",
            "input_rows": int(len(filtered_patents)),
            "excluded_rows_by_unit_filter": 0,
            "kept_rows_after_unit_filter": int(len(filtered_patents)),
        },
    )
    filtered_summary = _run_special_firm_variant(
        paths=paths,
        category=FILTERED_OUTPUT_CATEGORY,
        patent_df=filtered_variant_patents,
        year_col=year_col,
        firm_year_special=firm_year_special,
        special_uccs=special_uccs,
        quality_threshold=quality_threshold,
        regression_topk_share=regression_topk_share,
        policy_start_year=policy_start_year,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        event_window=event_window,
        logger=logger,
        unit_filter_terms=unit_filter_terms,
        unit_filter_terms_path=effective_unit_filter_terms_path,
        sample_context={
            "variant": "filtered_units",
            "input_rows": int(unit_filter_stats["input_rows"]),
            "excluded_rows_by_unit_filter": int(unit_filter_stats["excluded_rows"]),
            "kept_rows_after_unit_filter": int(unit_filter_stats["kept_rows"]),
        },
    )

    sample_compare = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "category": OUTPUT_CATEGORY,
                **baseline_summary["sample_stats"],
            },
            {
                "variant": "filtered_units",
                "category": FILTERED_OUTPUT_CATEGORY,
                **filtered_summary["sample_stats"],
            },
        ]
    )
    sample_compare_path = paths.table_subdir(FILTERED_OUTPUT_CATEGORY) / "tbl_special_analysis_sample_compare.csv"
    sample_compare.to_csv(sample_compare_path, index=False, encoding="utf-8-sig")
    logger.info("特殊企业样本对照表已输出: %s", repo_relative(sample_compare_path))

    summary = {
        "experiment_id": experiment_id,
        "experiment_patent_panel_path": repo_relative(patent_path),
        "special_label_source": special_source,
        "special_ucc_source": special_ucc_source,
        "unit_filter_terms_path": repo_relative(effective_unit_filter_terms_path),
        "unit_filter_terms": unit_filter_terms,
        "sample_compare_path": repo_relative(sample_compare_path),
        "data_outputs": baseline_summary["data_outputs"] + filtered_summary["data_outputs"],
        "figure_outputs": baseline_summary["figure_outputs"] + filtered_summary["figure_outputs"],
        "table_outputs": baseline_summary["table_outputs"] + filtered_summary["table_outputs"] + [repo_relative(sample_compare_path)],
        "variants": {
            "baseline": baseline_summary,
            "filtered_units": filtered_summary,
        },
        "year_col": year_col,
        "regression_topk_share": float(regression_topk_share),
        "exact_date": bool(exact_date),
    }
    write_json(paths.metadata_dir / "analyze_special_firms.json", summary)
    logger.info(
        "特殊企业相关图表、表格和中间数据已输出: baseline=%s, filtered=%s",
        baseline_summary["sample_stats"]["patents_after_quality_filter"],
        filtered_summary["sample_stats"]["patents_after_quality_filter"],
    )
    return summary


def _run_special_firm_variant(
    *,
    paths,
    category: str,
    patent_df: pd.DataFrame,
    year_col: str,
    firm_year_special: pd.DataFrame,
    special_uccs: pd.Series,
    quality_threshold: float,
    regression_topk_share: float,
    policy_start_year: int,
    exclude_years: Sequence[int],
    quality_min: float,
    bs_min: float,
    event_window: int,
    logger,
    unit_filter_terms: Sequence[str],
    unit_filter_terms_path: Optional[Path],
    sample_context: dict[str, int | str],
) -> dict[str, object]:
    table_dir = paths.table_subdir(category)
    figure_dir = paths.figure_subdir(category)
    data_dir = paths.data_subdir(category)
    ranking_table_dir = table_dir / "年度UCC质量排序"
    ranking_table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] 开始输出年度 UCC 质量排序表，top_n=%s", category, UCC_YEAR_TOP_N)
    ucc_year_by_mean, ucc_year_by_top5 = _build_ucc_year_quality_rankings(
        patent_df,
        year_col=year_col,
        firm_year_special=firm_year_special,
        special_uccs=special_uccs,
        top_n=UCC_YEAR_TOP_N,
    )
    ucc_year_mean_csv = ranking_table_dir / f"tbl_ucc_year_top{UCC_YEAR_TOP_N}_by_mean_quality.csv"
    ucc_year_top5_csv = ranking_table_dir / f"tbl_ucc_year_top{UCC_YEAR_TOP_N}_by_top5_mean_quality.csv"
    ucc_year_by_mean.to_csv(ucc_year_mean_csv, index=False, encoding="utf-8-sig")
    ucc_year_by_top5.to_csv(ucc_year_top5_csv, index=False, encoding="utf-8-sig")

    logger.info("[%s] 开始构造静态特殊企业面板", category)
    company_agg = build_company_special_panel_from_ucc_set(
        patent_df,
        special_uccs,
        quality_threshold=quality_threshold,
    )
    company_agg_path = data_dir / "company_special_panel.parquet"
    company_agg.to_parquet(company_agg_path, index=False)

    firm_compare = build_group_comparison_table(
        company_agg,
        group_col="is_special",
        var_specs=[
            (f"#Patents with Quality_q >= {quality_threshold:g}", "high_q_count", True),
            ("Total patents", "total_patents", True),
            ("log(1+Total patents)", "log_total_patents", False),
            ("Mean Quality_q", "mean_quality", False),
        ],
    )
    firm_compare_csv = table_dir / "tbl_firm_compare.csv"
    firm_compare_tex = table_dir / "tbl_firm_compare.tex"
    latex = export_table(
        firm_compare,
        csv_path=firm_compare_csv,
        tex_path=firm_compare_tex,
        caption="Firm-level Patent Outcomes: Special vs Other Firms",
        label="tab:firm_compare",
        digits=3,
        escape=False,
        index=False,
    )
    if latex is not None:
        (table_dir / "firm_compare.tex").write_text(latex, encoding="utf-8")

    p_dyn = attach_special_year_labels(
        patent_df,
        firm_year_special,
        policy_start_year=policy_start_year,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        year_col=year_col,
    )
    p_dyn_path = data_dir / "patents_special_year.parquet"
    p_dyn.to_parquet(p_dyn_path, index=False)

    pat_special_year = p_dyn[p_dyn["is_special_year"] == 1].copy()
    pat_other_year = p_dyn[p_dyn["is_special_year"] == 0].copy()
    quality_summary = pd.DataFrame(
        [
            _summarize_quality(pat_special_year, "Special (firm-year)"),
            _summarize_quality(pat_other_year, "Other (firm-year)"),
        ]
    )
    patent_quality_summary_csv = table_dir / "tbl_patent_special_year_quality_summary.csv"
    patent_quality_summary_tex = table_dir / "tbl_patent_special_year_quality_summary.tex"
    export_table(
        quality_summary,
        csv_path=patent_quality_summary_csv,
        tex_path=patent_quality_summary_tex,
        caption="Patent-level Quality Summary: Special-year vs Other-year",
        label="tab:patent_special_year_quality",
        digits=3,
        escape=False,
        index=False,
    )

    plt.figure(figsize=(8, 4.8))
    _plot_two_group_hist(
        pat_special_year[QUALITY_COL],
        pat_other_year[QUALITY_COL],
        label_a="Special (firm-year)",
        label_b="Other (firm-year)",
    )
    special_hist_path = figure_dir / "fig_special_vs_other_hist_log1p.png"
    save_figure(special_hist_path)

    company_year_agg = build_company_year_special_panel(
        p_dyn,
        quality_threshold=quality_threshold,
        year_col=year_col,
    )
    company_year_path = data_dir / "company_year_special.parquet"
    company_year_agg.to_parquet(company_year_path, index=False)

    firmyear_compare = build_group_comparison_table(
        company_year_agg,
        group_col="is_special_year",
        var_specs=[
            (f"#Patents with Quality_q >= {quality_threshold:g}", "high_q_count", True),
            ("Total patents", "total_patents", True),
            ("log(1+Total patents)", "log_total_patents", False),
            ("Mean Quality_q", "mean_quality", False),
            ("Max Quality_q", "max_quality", False),
        ],
    )
    firmyear_compare_csv = table_dir / "tbl_firmyear_compare.csv"
    firmyear_compare_tex = table_dir / "tbl_firmyear_compare.tex"
    latex_year = export_table(
        firmyear_compare,
        csv_path=firmyear_compare_csv,
        tex_path=firmyear_compare_tex,
        caption="Firm-year Patent Outcomes: Special-year vs Other-year",
        label="tab:firmyear_compare",
        digits=3,
        escape=False,
        index=False,
    )
    if latex_year is not None:
        (table_dir / "firmyear_compare.tex").write_text(latex_year, encoding="utf-8")

    trend = (
        p_dyn.assign(_q=to_numeric(p_dyn[QUALITY_COL]))
        .dropna(subset=[year_col])
        .groupby([year_col, "is_special_year"], as_index=False)
        .agg(mean_quality=("_q", "mean"), n=("_q", "size"))
        .sort_values([year_col, "is_special_year"])
    )
    trend_csv = table_dir / "tbl_special_year_trend.csv"
    trend.to_csv(trend_csv, index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    for group_value, group_df in trend.groupby("is_special_year"):
        label = "Special-year" if int(group_value) == 1 else "Other-year"
        plt.plot(group_df[year_col].astype(int), group_df["mean_quality"], marker="o", label=label)
    plt.xlabel("Year")
    plt.ylabel("Mean Quality_q")
    plt.title("Yearly mean Quality_q: Special-year vs Other-year")
    plt.legend()
    plt.tight_layout()
    trend_fig_path = figure_dir / "fig_special_year_vs_other_year_trend.png"
    save_figure(trend_fig_path)

    company_year_abc = build_company_year_abc_panel(
        p_dyn,
        quality_threshold=quality_threshold,
        year_col=year_col,
    )
    company_year_abc_path = data_dir / "company_year_abc.parquet"
    company_year_abc.to_parquet(company_year_abc_path, index=False)

    abc_summary = build_abc_summary_table(company_year_abc)
    abc_desc_csv = table_dir / "tbl_firm_year_abc_desc.csv"
    abc_desc_tex = table_dir / "tbl_firm_year_abc_desc.tex"
    export_table(
        abc_summary,
        csv_path=abc_desc_csv,
        tex_path=abc_desc_tex,
        caption="Descriptive statistics by firm-year treatment status (2008+)",
        label="tab:firm_year_abc_desc",
        digits=3,
        escape=False,
        index=False,
    )

    abc_source = p_dyn.copy()
    abc_source[PATENT_UCC_COL] = abc_source[PATENT_UCC_COL].astype("string").fillna("").str.strip()
    ever_special = company_year_abc[[PATENT_UCC_COL, "ever_special"]].drop_duplicates()
    abc_source = abc_source.merge(ever_special, on=PATENT_UCC_COL, how="left")
    abc_source["ever_special"] = to_numeric(abc_source["ever_special"]).fillna(0).astype(int)
    abc_source["firm_group_3"] = np.select(
        [
            (abc_source["ever_special"] == 1) & (abc_source["is_special_year"] == 1),
            (abc_source["ever_special"] == 1) & (abc_source["is_special_year"] == 0),
            (abc_source["ever_special"] == 0),
        ],
        ABC_GROUPS,
        default="C_never_treated",
    )
    abc_source[QUALITY_COL] = to_numeric(abc_source[QUALITY_COL])

    plt.figure(figsize=(9, 5))
    for group_name in ABC_GROUPS:
        values = np.log1p(abc_source.loc[abc_source["firm_group_3"] == group_name, QUALITY_COL].dropna())
        plt.hist(values, bins=80, density=True, alpha=0.35, label=ABC_LABELS[group_name])
    plt.xlabel("log(1 + Quality_q)")
    plt.ylabel("Density")
    plt.title("Patent-level distribution of Quality_q (A/B/C, 2008+)")
    plt.legend()
    plt.tight_layout()
    abc_patent_fig = figure_dir / "fig_abc_patent_quality_distribution.png"
    save_figure(abc_patent_fig)

    plt.figure(figsize=(9, 5))
    for group_name in ABC_GROUPS:
        values = np.log1p(company_year_abc.loc[company_year_abc["firm_group_3"] == group_name, "mean_quality"].dropna())
        plt.hist(values, bins=80, density=True, alpha=0.35, label=ABC_LABELS[group_name])
    plt.xlabel("log(1 + mean_quality)")
    plt.ylabel("Density")
    plt.title("Firm-year distribution of mean_quality (A/B/C, 2008+)")
    plt.legend()
    plt.tight_layout()
    abc_firm_year_hist = figure_dir / "fig_abc_firm_year_mean_quality_distribution.png"
    save_figure(abc_firm_year_hist)

    yearly_mean = (
        company_year_abc.groupby([year_col, "firm_group_3"], sort=False)["mean_quality"]
        .mean()
        .reset_index()
    )
    abc_yearly_mean_csv = table_dir / "tbl_abc_yearly_mean_quality.csv"
    yearly_mean.to_csv(abc_yearly_mean_csv, index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 5))
    for group_name in ABC_GROUPS:
        series = yearly_mean[yearly_mean["firm_group_3"] == group_name].sort_values(year_col)
        plt.plot(series[year_col], series["mean_quality"], label=ABC_LABELS[group_name])
    plt.xlabel("Year")
    plt.ylabel("Mean of mean_quality (firm-year)")
    plt.title("Yearly trend: mean_quality by group (2008+)")
    plt.legend()
    plt.tight_layout()
    abc_yearly_mean_fig = figure_dir / "fig_abc_yearly_mean_quality.png"
    save_figure(abc_yearly_mean_fig)

    yearly_high_q = (
        company_year_abc.groupby([year_col, "firm_group_3"], sort=False)[["high_q_count", "total_patents"]]
        .sum()
        .reset_index()
    )
    yearly_high_q["high_q_share"] = yearly_high_q["high_q_count"] / yearly_high_q["total_patents"].replace(0, np.nan)
    abc_yearly_share_csv = table_dir / "tbl_abc_yearly_high_q_share.csv"
    yearly_high_q.to_csv(abc_yearly_share_csv, index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 5))
    for group_name in ABC_GROUPS:
        series = yearly_high_q[yearly_high_q["firm_group_3"] == group_name].sort_values(year_col)
        plt.plot(series[year_col], series["high_q_share"], label=ABC_LABELS[group_name])
    plt.xlabel("Year")
    plt.ylabel(f"High-Q share (Quality_q >= {quality_threshold:g})")
    plt.title("Yearly trend: share of high-quality patents by group (2008+)")
    plt.legend()
    plt.tight_layout()
    abc_yearly_share_fig = figure_dir / "fig_abc_yearly_high_q_share.png"
    save_figure(abc_yearly_share_fig)

    ab = company_year_abc[company_year_abc["firm_group_3"].isin(["A_treated_year", "B_same_firm_other_year"])].copy()
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [
            ab.loc[ab["firm_group_3"] == "A_treated_year", "mean_quality"].dropna().to_numpy(),
            ab.loc[ab["firm_group_3"] == "B_same_firm_other_year", "mean_quality"].dropna().to_numpy(),
        ],
        labels=["A Treated-year", "B Same-firm other-year"],
        showfliers=False,
    )
    plt.ylabel("mean_quality (firm-year)")
    plt.title("Within-ever-special firms: A vs B")
    plt.tight_layout()
    ab_box_fig = figure_dir / "fig_abc_ab_boxplot.png"
    save_figure(ab_box_fig)

    overall = company_year_abc.groupby("firm_group_3")[["high_q_count", "total_patents"]].sum()
    overall["high_q_share"] = overall["high_q_count"] / overall["total_patents"].replace(0, np.nan)
    overall["mean_quality"] = company_year_abc.groupby("firm_group_3")["mean_quality"].mean()
    overall = overall.reset_index()
    abc_overall_csv = table_dir / "tbl_abc_overall_compare.csv"
    overall.to_csv(abc_overall_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 4.8))
    x_axis = np.arange(len(ABC_GROUPS))
    mean_values = [overall.loc[overall["firm_group_3"] == group_name, "mean_quality"].iloc[0] for group_name in ABC_GROUPS]
    share_values = [overall.loc[overall["firm_group_3"] == group_name, "high_q_share"].iloc[0] for group_name in ABC_GROUPS]
    plt.bar(x_axis - 0.2, mean_values, width=0.4, label="Avg mean_quality")
    plt.bar(x_axis + 0.2, share_values, width=0.4, label=f"High-Q share (>= {quality_threshold:g})")
    plt.xticks(x_axis, [ABC_LABELS[group_name] for group_name in ABC_GROUPS], rotation=0)
    plt.title("Overall comparison by group (2008+)")
    plt.legend()
    plt.tight_layout()
    abc_bar_fig = figure_dir / "fig_abc_overall_compare.png"
    save_figure(abc_bar_fig)

    event_study = build_event_study_frame(company_year_abc, year_col=year_col, window=event_window)
    event_study_csv = table_dir / "tbl_event_study_mean_quality.csv"
    event_study.to_csv(event_study_csv, index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    plt.plot(event_study["event_time"], event_study["mean_quality"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Event time (year - first treated year)")
    plt.ylabel("Mean mean_quality (firm-year)")
    plt.title(f"Event-study style trend around first treated year (+/-{event_window} years)")
    plt.tight_layout()
    event_fig = figure_dir / "fig_event_study_mean_quality.png"
    save_figure(event_fig)

    regression_summary = run_special_firm_regressions(
        paths=paths,
        category=category,
        patent_df=patent_df,
        dynamic_patent_df=p_dyn,
        year_col=year_col,
        firm_year_special=firm_year_special,
        special_uccs=special_uccs,
        topk_share=regression_topk_share,
        logger=logger,
    )
    logger.info(
        "[%s] 特殊企业回归完成: topk_share=%.3f, patent_static=%s, firm_year_dynamic=%s",
        category,
        float(regression_topk_share),
        regression_summary["panel_rows"]["patent_static"],
        regression_summary["panel_rows"]["firm_year_dynamic"],
    )

    sample_stats = {
        **sample_context,
        "patents_after_quality_filter": int(len(patent_df)),
        "dynamic_patents": int(len(p_dyn)),
        "pat_special_year_rows": int(len(pat_special_year)),
        "pat_other_year_rows": int(len(pat_other_year)),
        "company_special_rows": int(len(company_agg)),
        "company_year_rows": int(len(company_year_agg)),
        "company_year_abc_rows": int(len(company_year_abc)),
        "regression_patent_static_rows": int(regression_summary["panel_rows"]["patent_static"]),
        "regression_patent_dynamic_rows": int(regression_summary["panel_rows"]["patent_dynamic"]),
        "regression_firm_year_static_rows": int(regression_summary["panel_rows"]["firm_year_static"]),
        "regression_firm_year_dynamic_rows": int(regression_summary["panel_rows"]["firm_year_dynamic"]),
        "regression_firm_year_dynamic_within_rows": int(regression_summary["panel_rows"]["firm_year_dynamic_within"]),
        "regression_topk_share": float(regression_summary["topk_share"]),
    }
    sample_stats_csv = table_dir / "tbl_special_analysis_sample_sizes.csv"
    pd.DataFrame([sample_stats]).to_csv(sample_stats_csv, index=False, encoding="utf-8-sig")
    note_path = table_dir / "样本与过滤说明.md"
    _write_special_variant_note(
        note_path,
        category=category,
        year_col=year_col,
        sample_stats=sample_stats,
        unit_filter_terms=unit_filter_terms,
        unit_filter_terms_path=unit_filter_terms_path,
    )

    logger.info(
        "[%s] 样本统计: patents=%s, dynamic=%s, special_year=%s, other_year=%s",
        category,
        sample_stats["patents_after_quality_filter"],
        sample_stats["dynamic_patents"],
        sample_stats["pat_special_year_rows"],
        sample_stats["pat_other_year_rows"],
    )

    return {
        "category": category,
        "data_outputs": [
            repo_relative(company_agg_path),
            repo_relative(p_dyn_path),
            repo_relative(company_year_path),
            repo_relative(company_year_abc_path),
        ] + list(regression_summary["data_outputs"]),
        "figure_outputs": [
            repo_relative(special_hist_path),
            repo_relative(trend_fig_path),
            repo_relative(abc_patent_fig),
            repo_relative(abc_firm_year_hist),
            repo_relative(abc_yearly_mean_fig),
            repo_relative(abc_yearly_share_fig),
            repo_relative(ab_box_fig),
            repo_relative(abc_bar_fig),
            repo_relative(event_fig),
        ],
        "table_outputs": [
            repo_relative(firm_compare_csv),
            repo_relative(firm_compare_tex),
            repo_relative(firmyear_compare_csv),
            repo_relative(firmyear_compare_tex),
            repo_relative(patent_quality_summary_csv),
            repo_relative(patent_quality_summary_tex),
            repo_relative(abc_desc_csv),
            repo_relative(abc_desc_tex),
            repo_relative(trend_csv),
            repo_relative(abc_yearly_mean_csv),
            repo_relative(abc_yearly_share_csv),
            repo_relative(abc_overall_csv),
            repo_relative(event_study_csv),
            repo_relative(ucc_year_mean_csv),
            repo_relative(ucc_year_top5_csv),
            repo_relative(sample_stats_csv),
            repo_relative(note_path),
        ] + list(regression_summary["table_outputs"]),
        "sample_stats": sample_stats,
        "regression_summary": regression_summary,
    }


def _load_unit_filter_terms(path: Path) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _exclude_patents_by_applicant_terms(
    patent_df: pd.DataFrame,
    *,
    terms: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if APPLICANT_COL not in patent_df.columns or not terms:
        return patent_df.copy(), {
            "input_rows": int(len(patent_df)),
            "excluded_rows": 0,
            "kept_rows": int(len(patent_df)),
        }

    applicants = normalize_string_series(patent_df[APPLICANT_COL])
    pattern = "|".join(re.escape(term) for term in terms)
    mask = applicants.str.contains(pattern, regex=True, na=False)
    kept = patent_df.loc[~mask].copy()
    return kept, {
        "input_rows": int(len(patent_df)),
        "excluded_rows": int(mask.sum()),
        "kept_rows": int((~mask).sum()),
    }


def _write_special_variant_note(
    path: Path,
    *,
    category: str,
    year_col: str,
    sample_stats: dict[str, object],
    unit_filter_terms: Sequence[str],
    unit_filter_terms_path: Optional[Path],
) -> None:
    lines = [
        f"# {category}",
        "",
        f"- 年份口径: `{year_col}`",
        f"- 质量过滤后样本量: `{sample_stats['patents_after_quality_filter']}`",
        f"- 动态 special-year 样本量: `{sample_stats['dynamic_patents']}`",
        f"- special-year 专利数: `{sample_stats['pat_special_year_rows']}`",
        f"- other-year 专利数: `{sample_stats['pat_other_year_rows']}`",
    ]
    if unit_filter_terms_path is None:
        lines.extend(
            [
                "- 申请人过滤: 未执行",
                "- 本目录作为特殊企业分析的对照组保留。",
            ]
        )
    else:
        lines.extend(
            [
                f"- 申请人过滤词文件: `{repo_relative(unit_filter_terms_path)}`",
                f"- 过滤词: `{', '.join(unit_filter_terms)}`",
                "- 过滤规则: 若 `申请人` 字段中任一联合申请人名称包含任一过滤词，则整条专利记录剔除。",
                f"- 过滤前样本量: `{sample_stats['input_rows']}`",
                f"- 因过滤词剔除样本量: `{sample_stats['excluded_rows_by_unit_filter']}`",
                f"- 过滤后保留样本量: `{sample_stats['kept_rows_after_unit_filter']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_ucc_year_quality_rankings(
    patent_df: pd.DataFrame,
    *,
    year_col: str,
    firm_year_special: pd.DataFrame,
    special_uccs: pd.Series,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_source = patent_df.copy()
    rank_source[PATENT_UCC_COL] = normalize_string_series(rank_source[PATENT_UCC_COL])
    rank_source[year_col] = to_numeric(rank_source[year_col]).astype("Int64")
    rank_source[QUALITY_COL] = to_numeric(rank_source[QUALITY_COL])
    if APPLICANT_COL not in rank_source.columns:
        rank_source[APPLICANT_COL] = ""
    rank_source[APPLICANT_COL] = normalize_string_series(rank_source[APPLICANT_COL])

    label_panel = firm_year_special[[PATENT_UCC_COL, year_col, "is_special_year"]].copy()
    label_panel[PATENT_UCC_COL] = normalize_string_series(label_panel[PATENT_UCC_COL])
    label_panel[year_col] = to_numeric(label_panel[year_col]).astype("Int64")
    label_panel["is_special_year"] = to_numeric(label_panel["is_special_year"]).fillna(0).astype(int)

    rank_source = rank_source.merge(label_panel, on=[PATENT_UCC_COL, year_col], how="left")
    rank_source["is_special_year"] = to_numeric(rank_source["is_special_year"]).fillna(0).astype(int)
    rank_source = rank_source[
        rank_source[year_col].notna()
        & rank_source[QUALITY_COL].notna()
        & rank_source[PATENT_UCC_COL].notna()
        & (~rank_source[PATENT_UCC_COL].isin(INVALID_UCC_VALUES))
    ].copy()

    name_lookup = (
        rank_source.loc[rank_source[APPLICANT_COL] != "", [year_col, PATENT_UCC_COL, APPLICANT_COL]]
        .drop_duplicates([year_col, PATENT_UCC_COL])
        .rename(columns={APPLICANT_COL: "申请人示例"})
    )
    special_ucc_set = sorted(
        {
            value
            for value in special_uccs.astype("string").fillna("").str.strip().tolist()
            if value and value not in INVALID_UCC_VALUES
        }
    )

    grouped = (
        pl.from_pandas(
            rank_source[[year_col, PATENT_UCC_COL, QUALITY_COL, "is_special_year"]],
            include_index=False,
        )
        .with_columns(
            pl.col(year_col).cast(pl.Int64, strict=False).alias(year_col),
            pl.col(PATENT_UCC_COL).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().alias(PATENT_UCC_COL),
            pl.col(QUALITY_COL).cast(pl.Float64, strict=False).alias(QUALITY_COL),
            pl.col("is_special_year").cast(pl.Int8, strict=False).fill_null(0).alias("is_special_year"),
        )
        .group_by([year_col, PATENT_UCC_COL])
        .agg(
            pl.len().alias("patent_count"),
            pl.col(QUALITY_COL).mean().alias("mean_quality"),
            pl.col(QUALITY_COL).sort(descending=True).head(5).mean().alias("top5_mean_quality"),
            pl.col(QUALITY_COL).max().alias("max_quality"),
            pl.col("is_special_year").max().cast(pl.Int8).alias("is_special_year"),
        )
        .with_columns(pl.col(PATENT_UCC_COL).is_in(special_ucc_set).cast(pl.Int8).alias("is_special"))
        .to_pandas()
    )
    grouped = grouped.merge(name_lookup, on=[year_col, PATENT_UCC_COL], how="left")
    grouped["申请人示例"] = grouped["申请人示例"].fillna("")

    by_mean = _format_ucc_year_rank_table(
        grouped,
        year_col=year_col,
        top_n=top_n,
        sort_columns=["mean_quality", "top5_mean_quality", "patent_count", PATENT_UCC_COL],
        ascending=[False, False, False, True],
    )
    by_top5 = _format_ucc_year_rank_table(
        grouped,
        year_col=year_col,
        top_n=top_n,
        sort_columns=["top5_mean_quality", "mean_quality", "patent_count", PATENT_UCC_COL],
        ascending=[False, False, False, True],
    )
    return by_mean, by_top5


def _format_ucc_year_rank_table(
    grouped: pd.DataFrame,
    *,
    year_col: str,
    top_n: int,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for year, year_df in grouped.groupby(year_col, sort=True):
        ranked = year_df.sort_values(sort_columns, ascending=ascending, kind="mergesort").head(top_n).copy()
        ranked.insert(0, "年内排名", np.arange(1, len(ranked) + 1))
        outputs.append(ranked)

    if not outputs:
        return pd.DataFrame(
            columns=[
                "年内排名",
                year_col,
                PATENT_UCC_COL,
                "申请人示例",
                "patent_count",
                "mean_quality",
                "top5_mean_quality",
                "max_quality",
                "is_special",
                "is_special_year",
            ]
        )

    combined = pd.concat(outputs, ignore_index=True)
    return combined[
        [
            "年内排名",
            year_col,
            PATENT_UCC_COL,
            "申请人示例",
            "patent_count",
            "mean_quality",
            "top5_mean_quality",
            "max_quality",
            "is_special",
            "is_special_year",
        ]
    ]


def _summarize_quality(frame: pd.DataFrame, group_name: str) -> dict[str, object]:
    values = to_numeric(frame[QUALITY_COL])
    return {
        "group": group_name,
        "N": int(values.notna().sum()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _plot_two_group_hist(series_a: pd.Series, series_b: pd.Series, *, label_a: str, label_b: str) -> None:
    values_a = np.log1p(to_numeric(series_a).dropna())
    values_b = np.log1p(to_numeric(series_b).dropna())
    plt.hist(values_b, bins=80, alpha=0.45, label=label_b)
    plt.hist(values_a, bins=80, alpha=0.45, label=label_a)
    plt.xlabel("log(1 + Quality_q)")
    plt.ylabel("Count")
    plt.title("Quality_q distribution: Special firm-year vs Other firm-year")
    plt.legend()
    plt.tight_layout()


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="输出专精特新/特殊企业相关图表、对比表和公司年面板")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--experiment-patent-panel-path", help="experiment_patent_panel.parquet 路径")
    parser.add_argument("--firm-year-special-labels-path", help="共享 firm_year_special_labels.parquet 路径")
    parser.add_argument("--special-ucc-set-path", help="共享 special_ucc_set.parquet 路径")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--unit-filter-terms-path", default=DEFAULT_UNIT_FILTER_TERMS_PATH, help="高校/研究所等单位过滤词文件")
    parser.add_argument("--exclude-years", nargs="*", type=int, default=[1985, 1986], help="排除年份")
    parser.add_argument("--quality-min", type=float, default=1e-5, help="Quality_q 最小阈值")
    parser.add_argument("--bs-min", type=float, default=1e-6, help="BS 最小阈值")
    parser.add_argument("--quality-threshold", type=float, default=1.0, help="高质量阈值")
    parser.add_argument("--regression-topk-share", type=float, default=0.10, help="特殊企业回归的年度前 k%% 阈值")
    parser.add_argument("--policy-start-year", type=int, default=2008, help="特殊企业政策生效年份")
    parser.add_argument("--event-window", type=int, default=5, help="事件研究窗口")
    parser.add_argument("--exact-date", action="store_true", help="使用 exact_date 模式，读取/输出 stage2_exact")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    analyze_special_firms(
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        experiment_patent_panel_path=resolve_repo_path(args.experiment_patent_panel_path) if args.experiment_patent_panel_path else None,
        firm_year_special_labels_path=resolve_repo_path(args.firm_year_special_labels_path) if args.firm_year_special_labels_path else None,
        special_ucc_set_path=resolve_repo_path(args.special_ucc_set_path) if args.special_ucc_set_path else None,
        shared_root=args.shared_root,
        unit_filter_terms_path=resolve_repo_path(args.unit_filter_terms_path) if args.unit_filter_terms_path else None,
        exclude_years=args.exclude_years,
        quality_min=args.quality_min,
        bs_min=args.bs_min,
        quality_threshold=args.quality_threshold,
        regression_topk_share=args.regression_topk_share,
        policy_start_year=args.policy_start_year,
        event_window=args.event_window,
        exact_date=args.exact_date,
    )


if __name__ == "__main__":
    main()
