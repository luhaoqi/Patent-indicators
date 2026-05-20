"""按公开公告年份统计 stage1 专利数量并绘制折线图。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plotting import save_figure, set_chinese_font


DEFAULT_INPUT = Path(
    "outputs/experiments/标题_摘要_ExactTime_window_1/stage1_exact/patent_quality_output.csv"
)
DEFAULT_OUTPUT = Path("analysis/graph/patents_by_publish_year.png")


def build_year_counts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["公开公告年份"])
    df = df.dropna(subset=["公开公告年份"])
    df["公开公告年份"] = df["公开公告年份"].astype(int)
    counts = (
        df["公开公告年份"]
        .value_counts()
        .sort_index()
        .rename_axis("公开公告年份")
        .reset_index(name="专利数量")
    )
    return counts


def plot(counts: pd.DataFrame, output_path: Path, total: int) -> None:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        counts["公开公告年份"],
        counts["专利数量"],
        marker="o",
        markersize=4,
        linewidth=1.6,
        color="#1f77b4",
    )

    ax.set_xlabel("公开公告年份")
    ax.set_ylabel("当年专利数量（件）")
    ax.set_title(f"各年份专利公开数量（按公开公告年份口径，合计 {total:,} 件）")

    years = counts["公开公告年份"].tolist()
    ax.set_xticks(years[::2])
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)

    last_year = int(counts["公开公告年份"].max())
    last_count = int(counts.loc[counts["公开公告年份"] == last_year, "专利数量"].iloc[0])
    ax.annotate(
        f"{last_year} 年为不完整数据\n（{last_count:,} 件）",
        xy=(last_year, last_count),
        xytext=(-95, -85),
        textcoords="offset points",
        fontsize=9,
        color="#555555",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    peak_year = int(counts.loc[counts["专利数量"].idxmax(), "公开公告年份"])
    peak_count = int(counts["专利数量"].max())
    ax.annotate(
        f"峰值：{peak_year} 年\n{peak_count:,} 件",
        xy=(peak_year, peak_count),
        xytext=(-110, -30),
        textcoords="offset points",
        fontsize=9,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    save_figure(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("analysis/graph/patents_by_publish_year.csv"),
    )
    args = parser.parse_args()

    counts = build_year_counts(args.input)
    total = int(counts["专利数量"].sum())
    print(f"覆盖年份: {int(counts['公开公告年份'].min())} - {int(counts['公开公告年份'].max())}")
    print(f"专利总数: {total:,}")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.csv_output, index=False, encoding="utf-8-sig")
    plot(counts, args.output, total)
    print(f"已写入图: {args.output}")
    print(f"已写入数据: {args.csv_output}")


if __name__ == "__main__":
    main()
