from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.io import build_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402
from common.plotting import save_figure, set_chinese_font  # noqa: E402
from common.tables import export_table  # noqa: E402


def run_regressions(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    firm_year_innovation_path: Optional[Path] = None,
    financial_panel_path: Optional[Path] = None,
    shared_root: str = "outputs/shared",
    year_min: int = 2000,
    year_max: int = 2023,
) -> dict[str, object]:
    from linearmodels.panel import PanelOLS

    paths = build_experiment_paths(experiment_id, output_root=output_root)
    paths.ensure_dirs()
    logger = build_logger(f"run_regressions.{experiment_id}", paths.logs_dir / "run_regressions.log")
    set_chinese_font(logger=logger)

    innovation_path = firm_year_innovation_path or (paths.data_dir / "firm_year_innovation.parquet")
    logger.info("读取 firm_year_innovation: %s", repo_relative(innovation_path))
    innov = pd.read_parquet(innovation_path)
    innov = _normalize_innovation_frame(innov)

    effective_financial_panel_path = financial_panel_path
    if effective_financial_panel_path is None:
        shared_paths = build_shared_paths(shared_root)
        candidate = shared_paths.financial_panel_dir / "financial_annual_clean.parquet"
        if candidate.exists():
            effective_financial_panel_path = candidate

    if effective_financial_panel_path is None or not effective_financial_panel_path.exists():
        raise FileNotFoundError("缺少共享财务年报面板，请先运行 run_shared_prep.py 生成 shared financial_panel")
    logger.info("读取共享财务年报面板: %s", repo_relative(effective_financial_panel_path))
    fin_annual = pd.read_parquet(effective_financial_panel_path)
    if "year" not in fin_annual.columns and "Accper" in fin_annual.columns:
        fin_annual["Accper"] = pd.to_datetime(fin_annual["Accper"], errors="coerce")
        fin_annual["year"] = fin_annual["Accper"].dt.year
    if "year" not in fin_annual.columns:
        raise KeyError("共享财务年报面板缺少 year 列")
    fin_annual["year"] = pd.to_numeric(fin_annual["year"], errors="coerce")
    fin_annual = fin_annual[(fin_annual["year"] >= year_min) & (fin_annual["year"] <= year_max)].copy()
    logger.info("财务年报样本整理后 rows=%s", len(fin_annual))

    logger.info("开始合并财务数据与创新指标")
    df = fin_annual.merge(innov, on=["stkcd", "year"], how="inner")
    logger.info("财务与创新指标合并后 rows=%s", len(df))

    df = df[df["asset"].notna() & (df["asset"] > 0)].copy()
    df["ln_asset"] = np.log(df["asset"])
    df["lev_ratio"] = df["liability"] / df["asset"]
    df["soe"] = pd.to_numeric(df["soe"], errors="coerce").fillna(0).astype(int)
    if "研发费用" in df.columns:
        df["rd_intensity"] = (df["研发费用"] / df["asset"]).clip(upper=0.5)
    else:
        df["rd_intensity"] = np.nan

    df = df.sort_values(["stkcd", "year"]).copy()
    df = df.set_index(["stkcd", "year"])
    df["Innovation_z_lag1"] = df.groupby(level=0)["Innovation_z"].shift(1)
    df["Innovation_z_lag2"] = df.groupby(level=0)["Innovation_z"].shift(2)
    df["PatentCount_lag1"] = df.groupby(level=0)["PatentCount"].shift(1)
    logger.info("滞后变量构造完成，面板行数=%s", len(df))

    regression_panel_path = paths.data_dir / "regression_panel.parquet"
    df.reset_index().to_parquet(regression_panel_path, index=False)
    logger.info("回归面板已写出: %s", repo_relative(regression_panel_path))

    model_specs = [
        {
            "name": "ROA Baseline",
            "formula": "roa ~ 1 + Innovation_z + EntityEffects + TimeEffects",
            "var": "Innovation_z",
            "columns": ["roa", "Innovation_z"],
        },
        {
            "name": "ROA + Controls",
            "formula": "roa ~ 1 + Innovation_z + ln_asset + finlev + gassets + EntityEffects + TimeEffects",
            "var": "Innovation_z",
            "columns": ["roa", "Innovation_z", "ln_asset", "finlev", "gassets"],
        },
        {
            "name": "ROA Lag1",
            "formula": "roa ~ 1 + Innovation_z_lag1 + ln_asset + lev_ratio + gassets + EntityEffects + TimeEffects",
            "var": "Innovation_z_lag1",
            "columns": ["roa", "Innovation_z_lag1", "ln_asset", "lev_ratio", "gassets"],
        },
        {
            "name": "ROE + Controls",
            "formula": "roe ~ 1 + Innovation_z + ln_asset + lev_ratio + gassets + EntityEffects + TimeEffects",
            "var": "Innovation_z",
            "columns": ["roe", "Innovation_z", "ln_asset", "lev_ratio", "gassets"],
        },
    ]
    if df["rd_intensity"].notna().any():
        model_specs.append(
            {
                "name": "ROA + RD",
                "formula": "roa ~ 1 + Innovation_z + rd_intensity + ln_asset + lev_ratio + gassets + EntityEffects + TimeEffects",
                "var": "Innovation_z",
                "columns": ["roa", "Innovation_z", "rd_intensity", "ln_asset", "lev_ratio", "gassets"],
            }
        )

    summary_rows: list[dict[str, Any]] = []
    text_outputs: list[str] = []
    for spec in model_specs:
        reg_df = df[spec["columns"]].dropna().copy()
        if reg_df.empty:
            logger.warning("模型 %s 因缺少有效样本被跳过", spec["name"])
            continue
        logger.info("开始回归: %s，样本量=%s，公式=%s", spec["name"], len(reg_df), spec["formula"])
        model = PanelOLS.from_formula(spec["formula"], data=reg_df)
        result = model.fit(cov_type="clustered", cluster_entity=True)
        logger.info("完成回归: %s N=%s", spec["name"], int(result.nobs))
        summary_rows.append(
            {
                "model": spec["name"],
                "variable": spec["var"],
                "coef": result.params.get(spec["var"], np.nan),
                "se": result.std_errors.get(spec["var"], np.nan),
                "t": result.tstats.get(spec["var"], np.nan),
                "p": result.pvalues.get(spec["var"], np.nan),
                "nobs": int(result.nobs),
                "rsq_within": float(result.rsquared_within),
                "formula": spec["formula"],
            }
        )
        text_path = paths.tables_dir / f"reg_{_slugify(spec['name'])}.txt"
        text_path.write_text(str(result.summary), encoding="utf-8")
        text_outputs.append(repo_relative(text_path))

    summary_table = pd.DataFrame(summary_rows)
    summary_csv = paths.tables_dir / "tbl_regression_summary.csv"
    summary_tex = paths.tables_dir / "tbl_regression_summary.tex"
    export_table(
        summary_table,
        csv_path=summary_csv,
        tex_path=summary_tex,
        caption="Panel Regression Summary",
        label="tab:reg_summary",
        digits=4,
        escape=False,
        index=False,
    )
    logger.info("回归汇总表已输出: %s", repo_relative(summary_csv))

    coefficient_fig = None
    if not summary_table.empty:
        plt.figure(figsize=(9, 4.8))
        x_axis = np.arange(len(summary_table))
        y = summary_table["coef"].to_numpy(dtype=float)
        se = summary_table["se"].to_numpy(dtype=float)
        plt.errorbar(x_axis, y, yerr=1.96 * se, fmt="o", capsize=4)
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.xticks(x_axis, summary_table["model"], rotation=20, ha="right")
        plt.ylabel("Coefficient")
        plt.title("Innovation coefficient by regression specification")
        plt.tight_layout()
        coefficient_fig = paths.figures_dir / "fig_regression_coefficients.png"
        save_figure(coefficient_fig)

    summary = {
        "experiment_id": experiment_id,
        "firm_year_innovation_path": repo_relative(innovation_path),
        "financial_panel_path": repo_relative(effective_financial_panel_path) if effective_financial_panel_path is not None else None,
        "regression_panel_path": repo_relative(regression_panel_path),
        "table_outputs": [repo_relative(summary_csv), repo_relative(summary_tex)] + text_outputs,
        "figure_outputs": [repo_relative(coefficient_fig)] if coefficient_fig is not None else [],
        "models_run": [row["model"] for row in summary_rows],
    }
    write_json(paths.metadata_dir / "run_regressions.json", summary)
    return summary


def _normalize_innovation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {}
    if "Stkid" in df.columns:
        rename_map["Stkid"] = "stkcd"
    if "证券ID" in df.columns:
        rename_map["证券ID"] = "stkcd"
    if "ShortName" in df.columns:
        rename_map["ShortName"] = "shortname"
    if "公司简称" in df.columns:
        rename_map["公司简称"] = "shortname"
    if "PatentCount" not in df.columns and "patent_count" in df.columns:
        rename_map["patent_count"] = "PatentCount"
    df = df.rename(columns=rename_map)
    required = ["stkcd", "year", "Innovation_raw", "Innovation_z", "PatentCount"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"firm_year_innovation 缺少列: {missing}")
    df["stkcd"] = pd.to_numeric(df["stkcd"], errors="coerce").astype("Int64").astype("string").str.zfill(6)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    return df


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("+", "plus")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="根据 firm_year_innovation 和财务面板运行固定效应回归")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--firm-year-innovation-path", help="firm_year_innovation.parquet 路径")
    parser.add_argument("--financial-panel-path", help="共享 financial_annual_clean.parquet 路径")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--year-min", type=int, default=2000, help="财务样本最小年份")
    parser.add_argument("--year-max", type=int, default=2023, help="财务样本最大年份")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    run_regressions(
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        firm_year_innovation_path=resolve_repo_path(args.firm_year_innovation_path) if args.firm_year_innovation_path else None,
        financial_panel_path=resolve_repo_path(args.financial_panel_path) if args.financial_panel_path else None,
        shared_root=args.shared_root,
        year_min=args.year_min,
        year_max=args.year_max,
    )


if __name__ == "__main__":
    main()
