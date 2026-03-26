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
from scipy import stats  # noqa: E402

from common.analysis import build_descriptive_table, filter_patents, to_numeric  # noqa: E402
from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, repo_relative, resolve_repo_path  # noqa: E402
from common.plotting import save_figure, set_chinese_font  # noqa: E402
from common.tables import export_table  # noqa: E402


def analyze_quality_basic(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    experiment_patent_panel_path: Optional[Path] = None,
    exclude_years: Sequence[int] = (1985, 1986),
    quality_min: float = 1e-5,
    bs_min: float = 1e-6,
    quality_desc_threshold: float = 5.0,
    yearly_count_thresholds: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
) -> dict[str, object]:
    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(f"analyze_quality_basic.{experiment_id}", paths.logs_dir / "analyze_quality_basic.log")
    set_chinese_font(logger=logger)

    patent_path = experiment_patent_panel_path or (paths.data_dir / "experiment_patent_panel.parquet")
    if not patent_path.exists():
        raise FileNotFoundError(f"找不到 experiment_patent_panel: {patent_path}")
    logger.info("读取专利实验面板: %s", repo_relative(patent_path))
    patent_df = pd.read_parquet(patent_path)
    logger.info("开始按阈值过滤专利样本")
    filtered = filter_patents(
        patent_df,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
    )
    filtered["Quality_q"] = to_numeric(filtered["Quality_q"])
    if "被引证次数" not in filtered.columns:
        filtered["被引证次数"] = 0
        logger.warning("main_enriched 中缺少 被引证次数 列，已按 0 处理")
    filtered["被引证次数"] = to_numeric(filtered["被引证次数"]).fillna(0)
    filtered = filtered.dropna(subset=["Quality_q"]).copy()
    logger.info("基础图表样本量: %s", len(filtered))

    q_all = filtered["Quality_q"].dropna()
    q_high = filtered.loc[filtered["Quality_q"] >= quality_desc_threshold, "Quality_q"]
    cites = filtered["被引证次数"]
    logger.info("开始生成描述统计表")
    desc_table = build_descriptive_table(
        {
            "Quality_q (All)": q_all,
            f"Quality_q >= {quality_desc_threshold:g}": q_high,
            "Citations": cites,
        }
    )
    desc_csv = paths.tables_dir / "tbl_desc_patent_quality.csv"
    desc_tex = paths.tables_dir / "tbl_desc_patent_quality.tex"
    export_table(
        desc_table,
        csv_path=desc_csv,
        tex_path=desc_tex,
        caption="Descriptive Statistics of Patent Innovation and Citations",
        label="tab:desc_stats",
        digits=3,
        escape=False,
    )
    logger.info("描述统计表已输出: %s", repo_relative(desc_csv))

    x = np.log1p(filtered["Quality_q"].to_numpy())
    y = np.log1p(filtered["被引证次数"].to_numpy())
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    logger.info("散点/回归图有效样本量: %s", len(x))

    plt.figure(figsize=(8, 4.8))
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xlabel("log(1 + Quality_q)")
    plt.ylabel("log(1 + 被引证次数)")
    plt.title("Quality_q vs Citations")
    plt.grid(True, alpha=0.3)
    scatter_path = paths.figures_dir / "fig_quality_vs_citations_logq_logcite.png"
    save_figure(scatter_path)
    logger.info("散点图已输出: %s", repo_relative(scatter_path))

    if len(x) >= 2:
        logger.info("开始估计 Quality_q 与被引证次数线性关系")
        reg = stats.linregress(x, y)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = reg.intercept + reg.slope * x_line
        plt.figure(figsize=(8, 4.8))
        plt.scatter(x, y, s=8, alpha=0.3)
        plt.plot(x_line, y_line, linewidth=2)
        plt.xlabel("log(1 + Quality_q)")
        plt.ylabel("log(1 + 被引证次数)")
        plt.title("Quality_q vs Citations (linear fit)")
        plt.grid(True, alpha=0.3)
        plt.text(
            0.02,
            0.98,
            f"R² = {reg.rvalue ** 2:.4f}\nSlope = {reg.slope:.4g}\nP-value = {reg.pvalue:.4g}\nN = {len(x)}",
            transform=plt.gca().transAxes,
            va="top",
            bbox={"alpha": 0.2},
        )
        ols_fig_path = paths.figures_dir / "fig_quality_vs_citations_fit_logq_logcite.png"
        save_figure(ols_fig_path)
        logger.info("回归拟合图已输出: %s", repo_relative(ols_fig_path))
        ols_table = pd.DataFrame(
            [
                {
                    "slope": reg.slope,
                    "intercept": reg.intercept,
                    "r_value": reg.rvalue,
                    "r_squared": reg.rvalue ** 2,
                    "p_value": reg.pvalue,
                    "std_err": reg.stderr,
                    "nobs": len(x),
                }
            ]
        )
    else:
        ols_fig_path = None
        ols_table = pd.DataFrame(
            [{"slope": np.nan, "intercept": np.nan, "r_value": np.nan, "r_squared": np.nan, "p_value": np.nan, "std_err": np.nan, "nobs": len(x)}]
        )
    export_table(
        ols_table,
        csv_path=paths.tables_dir / "tbl_quality_citation_ols.csv",
        tex_path=paths.tables_dir / "tbl_quality_citation_ols.tex",
        caption="Patent Quality and Citations Regression",
        label="tab:quality_citation_ols",
        digits=4,
        escape=False,
        index=False,
    )
    logger.info("回归结果表已输出")

    plt.figure(figsize=(8, 4.8))
    plt.hist(np.log1p(q_all.to_numpy()), bins=60)
    plt.xlabel("log(1 + Quality_q)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Quality_q")
    plt.grid(True, alpha=0.3)
    dist_path = paths.figures_dir / "fig_quality_distribution_log1p_logy.png"
    save_figure(dist_path)
    logger.info("质量分布图已输出: %s", repo_relative(dist_path))

    logger.info("开始生成年度均值图表")
    yearly_mean = filtered.groupby("申请年份")["Quality_q"].mean().sort_index().reset_index()
    yearly_mean.columns = ["申请年份", "mean_quality"]
    yearly_mean.to_csv(paths.tables_dir / "tbl_yearly_mean_quality.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    plt.plot(yearly_mean["申请年份"], yearly_mean["mean_quality"], marker="o")
    plt.xlabel("申请年份")
    plt.ylabel("Mean(Quality_q)")
    plt.title("Yearly Mean of Quality_q")
    plt.grid(True, alpha=0.3)
    yearly_mean_fig = paths.figures_dir / "fig_yearly_mean_quality.png"
    save_figure(yearly_mean_fig)
    logger.info("年度均值图已输出: %s", repo_relative(yearly_mean_fig))

    yearly_rows: list[dict[str, float]] = []
    plt.figure(figsize=(10, 5))
    years = np.sort(filtered["申请年份"].dropna().unique())
    logger.info("开始按阈值生成年度高质量专利计数，阈值数=%s", len(yearly_count_thresholds))
    for threshold in yearly_count_thresholds:
        temp = filtered[filtered["Quality_q"].fillna(0) >= threshold]
        counts = temp.groupby("申请年份").size().reindex(years, fill_value=0)
        plt.plot(counts.index, counts.values, marker="o", label=f">= {threshold:g}")
        logger.info("阈值 %.3g 的年度统计完成", threshold)
        for year, count in counts.items():
            yearly_rows.append({"申请年份": int(year), "threshold": float(threshold), "count": int(count)})
    plt.xlabel("申请年份")
    plt.ylabel("Count")
    plt.title("Yearly counts by Quality_q thresholds")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    yearly_count_fig = paths.figures_dir / "fig_yearly_high_q_counts.png"
    save_figure(yearly_count_fig)
    pd.DataFrame(yearly_rows).to_csv(paths.tables_dir / "tbl_yearly_high_q_counts.csv", index=False, encoding="utf-8-sig")
    logger.info("年度高质量计数图表已输出: %s", repo_relative(yearly_count_fig))

    summary = {
        "experiment_id": experiment_id,
        "experiment_patent_panel_path": repo_relative(patent_path),
        "figure_paths": [
            repo_relative(scatter_path),
            repo_relative(dist_path),
            repo_relative(yearly_mean_fig),
            repo_relative(yearly_count_fig),
        ]
        + ([repo_relative(ols_fig_path)] if ols_fig_path is not None else []),
        "table_paths": [
            repo_relative(desc_csv),
            repo_relative(desc_tex),
            repo_relative(paths.tables_dir / "tbl_quality_citation_ols.csv"),
            repo_relative(paths.tables_dir / "tbl_quality_citation_ols.tex"),
            repo_relative(paths.tables_dir / "tbl_yearly_mean_quality.csv"),
            repo_relative(paths.tables_dir / "tbl_yearly_high_q_counts.csv"),
        ],
        "rows_used": int(len(filtered)),
    }
    write_json(paths.metadata_dir / "analyze_quality_basic.json", summary)
    logger.info("基础图表与表格已输出")
    return summary


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="输出专利质量的基础图表和描述统计")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--experiment-patent-panel-path", help="experiment_patent_panel.parquet 路径")
    parser.add_argument("--exclude-years", nargs="*", type=int, default=[1985, 1986], help="排除年份")
    parser.add_argument("--quality-min", type=float, default=1e-5, help="Quality_q 最小阈值")
    parser.add_argument("--bs-min", type=float, default=1e-6, help="BS 最小阈值")
    parser.add_argument("--quality-desc-threshold", type=float, default=5.0, help="描述统计中的高质量阈值")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    analyze_quality_basic(
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        experiment_patent_panel_path=resolve_repo_path(args.experiment_patent_panel_path) if args.experiment_patent_panel_path else None,
        exclude_years=args.exclude_years,
        quality_min=args.quality_min,
        bs_min=args.bs_min,
        quality_desc_threshold=args.quality_desc_threshold,
    )


if __name__ == "__main__":
    main()
