from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.analysis import (  # noqa: E402
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
    resolve_patent_year_col,
    to_numeric,
)
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402
from common.plotting import save_figure, set_chinese_font  # noqa: E402
from common.tables import export_table  # noqa: E402


ABC_GROUPS = ["A_treated_year", "B_same_firm_other_year", "C_never_treated"]
ABC_LABELS = {
    "A_treated_year": "A Treated-year",
    "B_same_firm_other_year": "B Same-firm other-year",
    "C_never_treated": "C Never-treated",
}


def analyze_special_firms(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    experiment_patent_panel_path: Optional[Path] = None,
    firm_year_special_labels_path: Optional[Path] = None,
    special_ucc_set_path: Optional[Path] = None,
    shared_root: str = "outputs/shared",
    exclude_years: Sequence[int] = (1985, 1986),
    quality_min: float = 1e-5,
    bs_min: float = 1e-6,
    quality_threshold: float = 1.0,
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
    logger.info("过滤后专利样本量: %s", len(filtered_patents))

    logger.info("开始构造静态特殊企业面板")
    company_agg = build_company_special_panel_from_ucc_set(
        filtered_patents,
        special_uccs,
        quality_threshold=quality_threshold,
    )
    company_agg_path = paths.data_dir / "company_special_panel.parquet"
    company_agg.to_parquet(company_agg_path, index=False)
    logger.info("静态企业面板输出: %s", repo_relative(company_agg_path))

    logger.info("开始输出静态企业对比表")
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
    firm_compare_csv = paths.tables_dir / "tbl_firm_compare.csv"
    firm_compare_tex = paths.tables_dir / "tbl_firm_compare.tex"
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
        (paths.tables_dir / "firm_compare.tex").write_text(latex, encoding="utf-8")

    p_dyn = attach_special_year_labels(
        patent_df,
        firm_year_special,
        policy_start_year=policy_start_year,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
        year_col=year_col,
    )
    p_dyn_path = paths.data_dir / "patents_special_year.parquet"
    p_dyn.to_parquet(p_dyn_path, index=False)
    logger.info("动态 special-year 专利表输出: %s", repo_relative(p_dyn_path))

    pat_special_year = p_dyn[p_dyn["is_special_year"] == 1].copy()
    pat_other_year = p_dyn[p_dyn["is_special_year"] == 0].copy()
    quality_summary = pd.DataFrame(
        [
            _summarize_quality(pat_special_year, "Special (firm-year)"),
            _summarize_quality(pat_other_year, "Other (firm-year)"),
        ]
    )
    export_table(
        quality_summary,
        csv_path=paths.tables_dir / "tbl_patent_special_year_quality_summary.csv",
        tex_path=paths.tables_dir / "tbl_patent_special_year_quality_summary.tex",
        caption="Patent-level Quality Summary: Special-year vs Other-year",
        label="tab:patent_special_year_quality",
        digits=3,
        escape=False,
        index=False,
    )
    logger.info("专利层动态特殊企业摘要表已输出")

    plt.figure(figsize=(8, 4.8))
    _plot_two_group_hist(
        pat_special_year[QUALITY_COL],
        pat_other_year[QUALITY_COL],
        label_a="Special (firm-year)",
        label_b="Other (firm-year)",
    )
    special_hist_path = paths.figures_dir / "fig_special_vs_other_hist_log1p.png"
    save_figure(special_hist_path)
    logger.info("动态特殊企业直方图已输出: %s", repo_relative(special_hist_path))

    logger.info("开始构造 company_year special 面板")
    company_year_agg = build_company_year_special_panel(
        p_dyn,
        quality_threshold=quality_threshold,
        year_col=year_col,
    )
    company_year_path = paths.data_dir / "company_year_special.parquet"
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
    firmyear_compare_csv = paths.tables_dir / "tbl_firmyear_compare.csv"
    firmyear_compare_tex = paths.tables_dir / "tbl_firmyear_compare.tex"
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
        (paths.tables_dir / "firmyear_compare.tex").write_text(latex_year, encoding="utf-8")

    logger.info("开始汇总 special-year 年度趋势")
    trend = (
        p_dyn.assign(_q=to_numeric(p_dyn[QUALITY_COL]))
        .dropna(subset=[year_col])
        .groupby([year_col, "is_special_year"], as_index=False)
        .agg(mean_quality=("_q", "mean"), n=("_q", "size"))
        .sort_values([year_col, "is_special_year"])
    )
    trend.to_csv(paths.tables_dir / "tbl_special_year_trend.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    for group_value, group_df in trend.groupby("is_special_year"):
        label = "Special-year" if int(group_value) == 1 else "Other-year"
        plt.plot(group_df[year_col].astype(int), group_df["mean_quality"], marker="o", label=label)
    plt.xlabel("Year")
    plt.ylabel("Mean Quality_q")
    plt.title("Yearly mean Quality_q: Special-year vs Other-year")
    plt.legend()
    plt.tight_layout()
    trend_fig_path = paths.figures_dir / "fig_special_year_vs_other_year_trend.png"
    save_figure(trend_fig_path)
    logger.info("special-year 年度趋势图已输出: %s", repo_relative(trend_fig_path))

    logger.info("开始构造 A/B/C firm-year 面板")
    company_year_abc = build_company_year_abc_panel(
        p_dyn,
        quality_threshold=quality_threshold,
        year_col=year_col,
    )
    company_year_abc_path = paths.data_dir / "company_year_abc.parquet"
    company_year_abc.to_parquet(company_year_abc_path, index=False)

    abc_summary = build_abc_summary_table(company_year_abc)
    export_table(
        abc_summary,
        csv_path=paths.tables_dir / "tbl_firm_year_abc_desc.csv",
        tex_path=paths.tables_dir / "tbl_firm_year_abc_desc.tex",
        caption="Descriptive statistics by firm-year treatment status (2008+)",
        label="tab:firm_year_abc_desc",
        digits=3,
        escape=False,
        index=False,
    )
    logger.info("A/B/C 描述统计表已输出")

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
    abc_patent_fig = paths.figures_dir / "fig_abc_patent_quality_distribution.png"
    save_figure(abc_patent_fig)
    logger.info("A/B/C 专利层分布图已输出: %s", repo_relative(abc_patent_fig))

    plt.figure(figsize=(9, 5))
    for group_name in ABC_GROUPS:
        values = np.log1p(company_year_abc.loc[company_year_abc["firm_group_3"] == group_name, "mean_quality"].dropna())
        plt.hist(values, bins=80, density=True, alpha=0.35, label=ABC_LABELS[group_name])
    plt.xlabel("log(1 + mean_quality)")
    plt.ylabel("Density")
    plt.title("Firm-year distribution of mean_quality (A/B/C, 2008+)")
    plt.legend()
    plt.tight_layout()
    abc_firm_year_hist = paths.figures_dir / "fig_abc_firm_year_mean_quality_distribution.png"
    save_figure(abc_firm_year_hist)

    yearly_mean = (
        company_year_abc.groupby([year_col, "firm_group_3"], sort=False)["mean_quality"]
        .mean()
        .reset_index()
    )
    yearly_mean.to_csv(paths.tables_dir / "tbl_abc_yearly_mean_quality.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 5))
    for group_name in ABC_GROUPS:
        series = yearly_mean[yearly_mean["firm_group_3"] == group_name].sort_values(year_col)
        plt.plot(series[year_col], series["mean_quality"], label=ABC_LABELS[group_name])
    plt.xlabel("Year")
    plt.ylabel("Mean of mean_quality (firm-year)")
    plt.title("Yearly trend: mean_quality by group (2008+)")
    plt.legend()
    plt.tight_layout()
    abc_yearly_mean_fig = paths.figures_dir / "fig_abc_yearly_mean_quality.png"
    save_figure(abc_yearly_mean_fig)

    yearly_high_q = (
        company_year_abc.groupby([year_col, "firm_group_3"], sort=False)[["high_q_count", "total_patents"]]
        .sum()
        .reset_index()
    )
    yearly_high_q["high_q_share"] = yearly_high_q["high_q_count"] / yearly_high_q["total_patents"].replace(0, np.nan)
    yearly_high_q.to_csv(paths.tables_dir / "tbl_abc_yearly_high_q_share.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 5))
    for group_name in ABC_GROUPS:
        series = yearly_high_q[yearly_high_q["firm_group_3"] == group_name].sort_values(year_col)
        plt.plot(series[year_col], series["high_q_share"], label=ABC_LABELS[group_name])
    plt.xlabel("Year")
    plt.ylabel(f"High-Q share (Quality_q >= {quality_threshold:g})")
    plt.title("Yearly trend: share of high-quality patents by group (2008+)")
    plt.legend()
    plt.tight_layout()
    abc_yearly_share_fig = paths.figures_dir / "fig_abc_yearly_high_q_share.png"
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
    ab_box_fig = paths.figures_dir / "fig_abc_ab_boxplot.png"
    save_figure(ab_box_fig)

    overall = company_year_abc.groupby("firm_group_3")[["high_q_count", "total_patents"]].sum()
    overall["high_q_share"] = overall["high_q_count"] / overall["total_patents"].replace(0, np.nan)
    overall["mean_quality"] = company_year_abc.groupby("firm_group_3")["mean_quality"].mean()
    overall = overall.reset_index()
    overall.to_csv(paths.tables_dir / "tbl_abc_overall_compare.csv", index=False, encoding="utf-8-sig")

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
    abc_bar_fig = paths.figures_dir / "fig_abc_overall_compare.png"
    save_figure(abc_bar_fig)

    event_study = build_event_study_frame(company_year_abc, year_col=year_col, window=event_window)
    event_study.to_csv(paths.tables_dir / "tbl_event_study_mean_quality.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    plt.plot(event_study["event_time"], event_study["mean_quality"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Event time (year - first treated year)")
    plt.ylabel("Mean mean_quality (firm-year)")
    plt.title(f"Event-study style trend around first treated year (+/-{event_window} years)")
    plt.tight_layout()
    event_fig = paths.figures_dir / "fig_event_study_mean_quality.png"
    save_figure(event_fig)

    summary = {
        "experiment_id": experiment_id,
        "experiment_patent_panel_path": repo_relative(patent_path),
        "special_label_source": special_source,
        "special_ucc_source": special_ucc_source,
        "data_outputs": [
            repo_relative(company_agg_path),
            repo_relative(p_dyn_path),
            repo_relative(company_year_path),
            repo_relative(company_year_abc_path),
        ],
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
            repo_relative(paths.tables_dir / "tbl_patent_special_year_quality_summary.csv"),
            repo_relative(paths.tables_dir / "tbl_firm_year_abc_desc.csv"),
        ],
        "year_col": year_col,
        "exact_date": bool(exact_date),
    }
    write_json(paths.metadata_dir / "analyze_special_firms.json", summary)
    logger.info("特殊企业相关图表、表格和中间数据已输出")
    return summary


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
    parser.add_argument("--exclude-years", nargs="*", type=int, default=[1985, 1986], help="排除年份")
    parser.add_argument("--quality-min", type=float, default=1e-5, help="Quality_q 最小阈值")
    parser.add_argument("--bs-min", type=float, default=1e-6, help="BS 最小阈值")
    parser.add_argument("--quality-threshold", type=float, default=1.0, help="高质量阈值")
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
        exclude_years=args.exclude_years,
        quality_min=args.quality_min,
        bs_min=args.bs_min,
        quality_threshold=args.quality_threshold,
        policy_start_year=args.policy_start_year,
        event_window=args.event_window,
        exact_date=args.exact_date,
    )


if __name__ == "__main__":
    main()
