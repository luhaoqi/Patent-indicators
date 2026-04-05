from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.io import build_logger, close_logger, write_json  # noqa: E402
from common.paths import build_experiment_paths, build_shared_paths, repo_relative, resolve_repo_path  # noqa: E402
from common.plotting import save_figure, set_chinese_font  # noqa: E402
from common.tables import export_table  # noqa: E402


OUTPUT_CATEGORY = "回归分析"
CURRENT_SCHEMA_VERSION = 4
DEFAULT_SAMPLE_THRESHOLDS = (10, 5, 1)
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99
DEFAULT_RD_YEAR_MIN = 2019
DEFAULT_RD_YEAR_MAX = 2023
DEFAULT_FUTURE_HORIZONS = (1, 2, 3, 4, 5)
MAIN_QUALITY_VARS = ("mean_z_q_ft", "highq_share_ft", "log_highq_count_ft")
FUTURE_QUALITY_VARS = ("mean_z_q_ft", "highq_share_ft")
QUALITY_VAR_LABELS = {
    "mean_z_q_ft": "mean_z",
    "highq_share_ft": "highq_share",
    "log_highq_count_ft": "log_highq_count",
    "mean_raw_q_w_ft": "mean_raw_q_w",
    "rd_intensity_asset": "rd_asset",
}
REGRESSION_TEXT_DIRNAME = "regressions"


def run_regressions(
    *,
    experiment_id: str,
    output_root: str = "outputs/experiments",
    firm_year_innovation_path: Optional[Path] = None,
    financial_panel_path: Optional[Path] = None,
    shared_root: str = "outputs/shared",
    year_min: int = 2000,
    year_max: int = 2023,
    sample_thresholds: Sequence[int] = DEFAULT_SAMPLE_THRESHOLDS,
    winsor_lower: float = DEFAULT_WINSOR_LOWER,
    winsor_upper: float = DEFAULT_WINSOR_UPPER,
    rd_year_min: int = DEFAULT_RD_YEAR_MIN,
    rd_year_max: int = DEFAULT_RD_YEAR_MAX,
    future_horizons: Sequence[int] = DEFAULT_FUTURE_HORIZONS,
    exact_date: bool = False,
) -> dict[str, object]:
    from linearmodels.panel import PanelOLS

    thresholds = _normalize_thresholds(sample_thresholds)
    horizons = _normalize_future_horizons(future_horizons)
    if not (0.0 <= winsor_lower < winsor_upper <= 1.0):
        raise ValueError("winsor_lower / winsor_upper 必须满足 0 <= lower < upper <= 1")
    if rd_year_min > rd_year_max:
        raise ValueError("rd_year_min 不能大于 rd_year_max")

    paths = build_experiment_paths(experiment_id, output_root=output_root, exact_date=exact_date)
    paths.ensure_dirs()
    table_dir = paths.table_subdir(OUTPUT_CATEGORY)
    figure_dir = paths.figure_subdir(OUTPUT_CATEGORY)
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
    fin_annual = _normalize_financial_panel(fin_annual, year_min=year_min, year_max=year_max)
    logger.info("财务年报 universe 整理后 rows=%s firms=%s", len(fin_annual), fin_annual["stkcd"].nunique())

    logger.info("开始合并财务数据与创新指标")
    df = fin_annual.merge(innov, on=["stkcd", "year"], how="left")
    logger.info("财务与创新指标合并后 rows=%s", len(df))

    df = _prepare_regression_panel(df, winsor_lower=winsor_lower, winsor_upper=winsor_upper)
    dep_configs = _build_dep_configs(df)
    df = _add_future_outcome_columns(
        df,
        dep_sources=[config["dep_source"] for config in dep_configs],
        future_horizons=horizons,
    )
    regression_panel_path = paths.data_dir / "regression_panel.parquet"
    df.to_parquet(regression_panel_path, index=False)
    logger.info("回归面板已写出: %s", repo_relative(regression_panel_path))

    specs = _build_main_specs(dep_configs=dep_configs, sample_thresholds=thresholds)
    specs.extend(
        _build_future_specs(
            dep_configs=dep_configs,
            sample_thresholds=thresholds,
            future_horizons=horizons,
        )
    )
    specs.extend(
        _build_rd_specs(
            dep_configs=dep_configs,
            rd_year_min=rd_year_min,
            rd_year_max=rd_year_max,
            sample_thresholds=thresholds,
        )
    )
    logger.info("待执行回归规格数=%s", len(specs))

    universe_counts = {
        "financial_universe_rows": int(len(df)),
        "financial_universe_firms": int(df["stkcd"].nunique()),
        "patent_matched_rows": int((df["PatentCount"] >= 1).sum()),
        "patent_matched_firms": int(df.loc[df["PatentCount"] >= 1, "stkcd"].nunique()),
    }

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    text_outputs: list[str] = []
    success_count = 0
    failed_count = 0

    for spec in specs:
        summary_row, sample_row, text_path = _run_single_spec(
            df=df,
            spec=spec,
            table_dir=table_dir,
            logger=logger,
            universe_counts=universe_counts,
            panel_ols_cls=PanelOLS,
        )
        summary_rows.append(summary_row)
        sample_rows.append(sample_row)
        if text_path is not None:
            text_outputs.append(repo_relative(text_path))
        if summary_row["status"] == "success":
            success_count += 1
        else:
            failed_count += 1

    summary_table = pd.DataFrame(summary_rows)
    sample_table = pd.DataFrame(sample_rows)

    summary_csv = table_dir / "tbl_regression_summary.csv"
    summary_tex = table_dir / "tbl_regression_summary.tex"
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

    sample_csv = table_dir / "tbl_regression_sample_summary.csv"
    sample_tex = table_dir / "tbl_regression_sample_summary.tex"
    export_table(
        sample_table,
        csv_path=sample_csv,
        tex_path=sample_tex,
        caption="Regression Sample Summary",
        label="tab:reg_sample_summary",
        digits=4,
        escape=False,
        index=False,
    )
    logger.info("回归样本说明表已输出: %s", repo_relative(sample_csv))

    coefficient_fig = _plot_regression_coefficients(summary_table=summary_table, figure_dir=figure_dir)

    summary = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "firm_year_innovation_path": repo_relative(innovation_path),
        "financial_panel_path": repo_relative(effective_financial_panel_path) if effective_financial_panel_path is not None else None,
        "regression_panel_path": repo_relative(regression_panel_path),
        "table_outputs": [repo_relative(summary_csv), repo_relative(summary_tex)] + text_outputs,
        "sample_summary_outputs": [repo_relative(sample_csv), repo_relative(sample_tex)],
        "figure_outputs": [repo_relative(coefficient_fig)] if coefficient_fig is not None else [],
        "models_run": summary_table.loc[summary_table["status"] == "success", "spec_id"].tolist(),
        "success_count": int(success_count),
        "failed_count": int(failed_count),
        "sample_thresholds": list(thresholds),
        "winsor_lower": float(winsor_lower),
        "winsor_upper": float(winsor_upper),
        "rd_year_min": int(rd_year_min),
        "rd_year_max": int(rd_year_max),
        "future_horizons": list(horizons),
        "exact_date": bool(exact_date),
    }
    write_json(paths.metadata_dir / "run_regressions.json", summary)
    close_logger(logger)
    return summary


def _normalize_thresholds(sample_thresholds: Sequence[int]) -> tuple[int, ...]:
    values = sorted({int(value) for value in sample_thresholds}, reverse=True)
    if not values:
        raise ValueError("sample_thresholds 不能为空")
    if values[-1] < 1:
        raise ValueError("sample_thresholds 至少应包含一个 >=1 的门槛")
    return tuple(values)


def _normalize_future_horizons(future_horizons: Sequence[int]) -> tuple[int, ...]:
    values = sorted({int(value) for value in future_horizons})
    if not values:
        raise ValueError("future_horizons 不能为空")
    if values[0] < 1:
        raise ValueError("future_horizons 必须全部 >= 1")
    return tuple(values)


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

    if "mean_raw_q_w_ft" not in df.columns and "Innovation_raw" in df.columns:
        df["mean_raw_q_w_ft"] = df["Innovation_raw"]
    if "mean_z_q_ft" not in df.columns and "Innovation_z" in df.columns:
        df["mean_z_q_ft"] = df["Innovation_z"]
    if "Innovation_raw" not in df.columns and "mean_raw_q_w_ft" in df.columns:
        df["Innovation_raw"] = df["mean_raw_q_w_ft"]
    if "Innovation_z" not in df.columns and "mean_z_q_ft" in df.columns:
        df["Innovation_z"] = df["mean_z_q_ft"]

    required = ["stkcd", "year", "PatentCount"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"firm_year_innovation 缺少列: {missing}")

    df["stkcd"] = pd.to_numeric(df["stkcd"], errors="coerce").astype("Int64").astype("string").str.zfill(6)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["stkcd"].notna() & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    numeric_columns = [
        "PatentCount",
        "Innovation_raw",
        "Innovation_z",
        "mean_z_q_ft",
        "highq_share_ft",
        "highq_count_ft",
        "log_highq_count_ft",
        "mean_raw_q_w_ft",
        "mean_raw_q_ft",
        "legacy_topk_mean_q_w_ft",
        "log_patent_count_ft",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "log_patent_count_ft" not in df.columns:
        df["log_patent_count_ft"] = np.log1p(df["PatentCount"].clip(lower=0))

    return df.sort_values(["stkcd", "year"]).drop_duplicates(["stkcd", "year"], keep="last").reset_index(drop=True)


def _normalize_financial_panel(
    frame: pd.DataFrame,
    *,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {}
    if "Stkid" in df.columns and "stkcd" not in df.columns:
        rename_map["Stkid"] = "stkcd"
    if "证券ID" in df.columns and "stkcd" not in df.columns:
        rename_map["证券ID"] = "stkcd"
    df = df.rename(columns=rename_map)

    if "stkcd" not in df.columns:
        raise KeyError("共享财务年报面板缺少 stkcd 列")
    if "year" not in df.columns:
        if "Accper" not in df.columns:
            raise KeyError("共享财务年报面板缺少 year / Accper 列")
        df["Accper"] = pd.to_datetime(df["Accper"], errors="coerce")
        df["year"] = df["Accper"].dt.year
    elif "Accper" in df.columns:
        df["Accper"] = pd.to_datetime(df["Accper"], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["stkcd"] = pd.to_numeric(df["stkcd"], errors="coerce").astype("Int64").astype("string").str.zfill(6)
    df = df[df["stkcd"].notna() & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= int(year_min)) & (df["year"] <= int(year_max))].copy()

    numeric_columns = [
        "roa",
        "roe",
        "asset",
        "liability",
        "finlev",
        "gassets",
        "gfa",
        "ebit",
        "ebitda",
        "profit",
        "sales",
        "soe",
        "研发费用",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "asset" not in df.columns:
        raise KeyError("共享财务年报面板缺少 asset 列")

    df = df[df["asset"].notna() & (df["asset"] > 0)].copy()
    sort_columns = ["stkcd", "year"]
    if "Accper" in df.columns:
        sort_columns.append("Accper")
    df = df.sort_values(sort_columns).drop_duplicates(["stkcd", "year"], keep="last")
    return df.reset_index(drop=True)


def _prepare_regression_panel(
    frame: pd.DataFrame,
    *,
    winsor_lower: float,
    winsor_upper: float,
) -> pd.DataFrame:
    df = frame.copy()
    df["PatentCount"] = pd.to_numeric(df.get("PatentCount"), errors="coerce").fillna(0)
    if "log_patent_count_ft" in df.columns:
        df["log_patent_count_ft"] = pd.to_numeric(df["log_patent_count_ft"], errors="coerce")
    else:
        df["log_patent_count_ft"] = np.nan
    df["log_patent_count_ft"] = df["log_patent_count_ft"].where(df["log_patent_count_ft"].notna(), np.log1p(df["PatentCount"]))

    for column in ["mean_z_q_ft", "highq_share_ft", "highq_count_ft", "log_highq_count_ft", "mean_raw_q_w_ft", "mean_raw_q_ft", "Innovation_raw", "Innovation_z"]:
        if column not in df.columns:
            df[column] = np.nan
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ln_asset"] = np.log(df["asset"])
    df["lev_ratio"] = _safe_ratio(df.get("liability"), df.get("asset"))
    if "soe" in df.columns:
        df["soe"] = pd.to_numeric(df["soe"], errors="coerce").fillna(0).astype(int)
    else:
        df["soe"] = 0

    df["ebit_asset"] = _safe_ratio(df.get("ebit"), df.get("asset"))
    df["ebitda_asset"] = _safe_ratio(df.get("ebitda"), df.get("asset"))
    df["profit_asset"] = _safe_ratio(df.get("profit"), df.get("asset"))
    df["profit_margin"] = _safe_ratio(df.get("profit"), df.get("sales"))
    df["ebit_margin"] = _safe_ratio(df.get("ebit"), df.get("sales"))
    df["ebitda_margin"] = _safe_ratio(df.get("ebitda"), df.get("sales"))
    df["log_sales"] = _safe_log1p(df.get("sales"))

    rd_source = pd.to_numeric(df["研发费用"], errors="coerce") if "研发费用" in df.columns else pd.Series(np.nan, index=df.index)
    df["rd_intensity_asset"] = _safe_ratio(rd_source, df.get("asset"))
    df["rd_intensity_sales"] = _safe_ratio(rd_source, df.get("sales"))
    df["ln_rd"] = _safe_log1p(rd_source)

    df = df.sort_values(["stkcd", "year"]).reset_index(drop=True)
    df["sales_growth"] = df.groupby("stkcd", sort=False)["log_sales"].diff()

    winsor_targets = [
        "roa",
        "roe",
        "ebit_asset",
        "ebitda_asset",
        "profit_asset",
        "profit_margin",
        "ebit_margin",
        "ebitda_margin",
        "sales_growth",
        "gassets",
        "gfa",
    ]
    df = _winsorize_by_year(df, columns=winsor_targets, lower=winsor_lower, upper=winsor_upper)
    return df


def _future_dep_source(dep_source: str, horizon: int) -> str:
    return f"{dep_source}_lead{int(horizon)}"


def _add_future_outcome_columns(
    frame: pd.DataFrame,
    *,
    dep_sources: Sequence[str],
    future_horizons: Sequence[int],
) -> pd.DataFrame:
    df = frame.copy()
    base_columns = [column for column in _unique_in_order(dep_sources) if column in df.columns]
    if not base_columns:
        return df

    base = df[["stkcd", "year", *base_columns]].copy()
    for horizon in future_horizons:
        lead_frame = base.copy()
        lead_frame["year"] = lead_frame["year"] - int(horizon)
        lead_frame = lead_frame.rename(
            columns={column: _future_dep_source(column, int(horizon)) for column in base_columns}
        )
        df = df.merge(lead_frame, on=["stkcd", "year"], how="left")
    return df


def _safe_ratio(numerator: Any, denominator: Any) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series(dtype="float64")
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    mask = num.notna() & den.notna() & (den > 0)
    result.loc[mask] = num.loc[mask] / den.loc[mask]
    return result


def _safe_log1p(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    mask = values.notna() & (values >= 0)
    result.loc[mask] = np.log1p(values.loc[mask])
    return result


def _winsorize_by_year(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    lower: float,
    upper: float,
) -> pd.DataFrame:
    df = frame.copy()
    for column in columns:
        if column not in df.columns:
            continue
        low = df.groupby("year")[column].transform(lambda series: series.quantile(lower))
        high = df.groupby("year")[column].transform(lambda series: series.quantile(upper))
        df[f"{column}_w"] = pd.to_numeric(df[column], errors="coerce").clip(lower=low, upper=high)
    return df


def _build_dep_configs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    configs = [
        {"dep_var": "roa", "dep_source": "roa_w", "use_gassets_control": True, "priority_group": "main", "include_rd": True, "include_future": True},
        {"dep_var": "roe", "dep_source": "roe_w", "use_gassets_control": True, "priority_group": "main", "include_rd": True, "include_future": True},
        {"dep_var": "ebit_asset", "dep_source": "ebit_asset_w", "use_gassets_control": True, "priority_group": "main", "include_rd": True, "include_future": True},
        {"dep_var": "ebitda_asset", "dep_source": "ebitda_asset_w", "use_gassets_control": True, "priority_group": "main", "include_rd": True, "include_future": False},
        {"dep_var": "profit_asset", "dep_source": "profit_asset_w", "use_gassets_control": True, "priority_group": "main", "include_rd": True, "include_future": False},
        {"dep_var": "profit_margin", "dep_source": "profit_margin_w", "use_gassets_control": False, "priority_group": "margin", "include_rd": True, "include_future": False},
        {"dep_var": "ebit_margin", "dep_source": "ebit_margin_w", "use_gassets_control": False, "priority_group": "margin", "include_rd": True, "include_future": False},
        {"dep_var": "ebitda_margin", "dep_source": "ebitda_margin_w", "use_gassets_control": False, "priority_group": "margin", "include_rd": True, "include_future": False},
        {"dep_var": "log_sales", "dep_source": "log_sales", "use_gassets_control": False, "priority_group": "margin", "include_rd": True, "include_future": True},
        {"dep_var": "log_asset", "dep_source": "ln_asset", "use_gassets_control": False, "priority_group": "growth", "include_rd": True, "include_future": True},
        {"dep_var": "sales_growth", "dep_source": "sales_growth_w", "use_gassets_control": False, "priority_group": "growth", "include_rd": True, "include_future": False},
        {"dep_var": "gassets", "dep_source": "gassets_w", "use_gassets_control": False, "priority_group": "growth", "include_rd": True, "include_future": False},
        {"dep_var": "gfa", "dep_source": "gfa_w", "use_gassets_control": False, "priority_group": "growth", "include_rd": True, "include_future": False},
    ]
    available: list[dict[str, Any]] = []
    for config in configs:
        dep_source = config["dep_source"]
        if dep_source in frame.columns and frame[dep_source].notna().any():
            available.append(config)
    return available


def _build_controls_for_dep(
    dep_config: dict[str, Any],
    *,
    dep_source_override: Optional[str] = None,
) -> list[str]:
    controls: list[str] = []
    dep_source = dep_source_override or dep_config["dep_source"]
    if dep_source != "ln_asset":
        controls.append("ln_asset")
    if dep_source != "lev_ratio":
        controls.append("lev_ratio")
    if dep_config["use_gassets_control"] and dep_source != "gassets":
        controls.append("gassets")
    return controls


def _build_main_specs(
    *,
    dep_configs: Sequence[dict[str, Any]],
    sample_thresholds: Sequence[int],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    quality_pairs = [
        ("mean_z_q_ft", False),
        ("mean_z_q_ft", True),
        ("highq_share_ft", False),
        ("highq_share_ft", True),
        ("log_highq_count_ft", False),
        ("log_highq_count_ft", True),
        ("mean_raw_q_w_ft", True),
    ]
    for threshold in sample_thresholds:
        for dep_config in dep_configs:
            controls = _build_controls_for_dep(dep_config)
            for quality_var, add_count_control in quality_pairs:
                spec_id = _build_spec_id(
                    dep_var=dep_config["dep_var"],
                    key_regressor=quality_var,
                    sample_threshold=threshold,
                    suffix="cnt1" if add_count_control else "cnt0",
                )
                specs.append(
                    {
                        "spec_id": spec_id,
                        "model_family": "main",
                        "dep_var": dep_config["dep_var"],
                        "dep_source": dep_config["dep_source"],
                        "key_regressor": quality_var,
                        "regressor_vars": [quality_var],
                        "sample_threshold": int(threshold),
                        "sample_rule": f"PatentCount >= {int(threshold)}",
                        "year_range": None,
                        "controls": controls,
                        "add_count_control": bool(add_count_control),
                        "future_horizon": 0,
                        "rd_var": None,
                        "rd_same_sample": False,
                        "rd_year_min": None,
                        "rd_year_max": None,
                    }
                )
    return specs


def _build_future_specs(
    *,
    dep_configs: Sequence[dict[str, Any]],
    sample_thresholds: Sequence[int],
    future_horizons: Sequence[int],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    future_dep_configs = [config for config in dep_configs if config.get("include_future")]
    for threshold in sample_thresholds:
        for horizon in future_horizons:
            for dep_config in future_dep_configs:
                future_dep_source = _future_dep_source(dep_config["dep_source"], int(horizon))
                controls = _build_controls_for_dep(dep_config, dep_source_override=future_dep_source)
                for quality_var in FUTURE_QUALITY_VARS:
                    spec_id = _build_spec_id(
                        dep_var=dep_config["dep_var"],
                        key_regressor=quality_var,
                        sample_threshold=threshold,
                        suffix=f"h{int(horizon)}_cnt1",
                    )
                    specs.append(
                        {
                            "spec_id": spec_id,
                            "model_family": "future_main",
                            "dep_var": dep_config["dep_var"],
                            "dep_source": future_dep_source,
                            "key_regressor": quality_var,
                            "regressor_vars": [quality_var],
                            "sample_threshold": int(threshold),
                            "sample_rule": f"PatentCount >= {int(threshold)}",
                            "year_range": None,
                            "controls": controls,
                            "add_count_control": True,
                            "future_horizon": int(horizon),
                            "rd_var": None,
                            "rd_same_sample": False,
                            "rd_year_min": None,
                            "rd_year_max": None,
                        }
                    )
    return specs


def _build_rd_specs(
    *,
    dep_configs: Sequence[dict[str, Any]],
    rd_year_min: int,
    rd_year_max: int,
    sample_thresholds: Sequence[int],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    rd_var = "rd_intensity_asset"
    for sample_threshold in sample_thresholds:
        for dep_config in dep_configs:
            if not dep_config["include_rd"]:
                continue
            controls = _build_controls_for_dep(dep_config)
            for quality_var in MAIN_QUALITY_VARS:
                baseline_id = _build_spec_id(
                    dep_var=dep_config["dep_var"],
                    key_regressor=quality_var,
                    sample_threshold=sample_threshold,
                    suffix="rdsame",
                )
                specs.append(
                    {
                        "spec_id": baseline_id,
                        "model_family": "rd_same_sample",
                        "dep_var": dep_config["dep_var"],
                        "dep_source": dep_config["dep_source"],
                        "key_regressor": quality_var,
                        "regressor_vars": [quality_var],
                        "sample_threshold": int(sample_threshold),
                        "sample_rule": f"PatentCount >= {int(sample_threshold)}",
                        "year_range": f"{rd_year_min}-{rd_year_max}",
                        "controls": controls,
                        "add_count_control": True,
                        "future_horizon": 0,
                        "rd_var": rd_var,
                        "rd_same_sample": True,
                        "rd_year_min": int(rd_year_min),
                        "rd_year_max": int(rd_year_max),
                    }
                )

                horse_race_id = _build_spec_id(
                    dep_var=dep_config["dep_var"],
                    key_regressor=quality_var,
                    sample_threshold=sample_threshold,
                    suffix="rdhorse",
                )
                specs.append(
                    {
                        "spec_id": horse_race_id,
                        "model_family": "rd_horse_race",
                        "dep_var": dep_config["dep_var"],
                        "dep_source": dep_config["dep_source"],
                        "key_regressor": quality_var,
                        "regressor_vars": [quality_var, rd_var],
                        "sample_threshold": int(sample_threshold),
                        "sample_rule": f"PatentCount >= {int(sample_threshold)}",
                        "year_range": f"{rd_year_min}-{rd_year_max}",
                        "controls": controls,
                        "add_count_control": True,
                        "future_horizon": 0,
                        "rd_var": rd_var,
                        "rd_same_sample": True,
                        "rd_year_min": int(rd_year_min),
                        "rd_year_max": int(rd_year_max),
                    }
                )

            rd_only_id = _build_spec_id(
                dep_var=dep_config["dep_var"],
                key_regressor=rd_var,
                sample_threshold=sample_threshold,
                suffix="rdonly",
            )
            specs.append(
                {
                    "spec_id": rd_only_id,
                    "model_family": "rd_only",
                    "dep_var": dep_config["dep_var"],
                    "dep_source": dep_config["dep_source"],
                    "key_regressor": rd_var,
                    "regressor_vars": [rd_var],
                    "sample_threshold": int(sample_threshold),
                    "sample_rule": f"PatentCount >= {int(sample_threshold)}",
                    "year_range": f"{rd_year_min}-{rd_year_max}",
                    "controls": controls,
                    "add_count_control": True,
                    "future_horizon": 0,
                    "rd_var": rd_var,
                    "rd_same_sample": True,
                    "rd_year_min": int(rd_year_min),
                    "rd_year_max": int(rd_year_max),
                }
            )
    return specs


def _build_spec_id(
    *,
    dep_var: str,
    key_regressor: str,
    sample_threshold: int,
    suffix: str,
) -> str:
    dep_part = dep_var.replace("_", "")
    reg_part = QUALITY_VAR_LABELS.get(key_regressor, key_regressor.replace("_", ""))
    return f"{dep_part}_{reg_part}_pc{sample_threshold}_{suffix}"


def _build_regression_text_path(table_dir: Path, spec: dict[str, Any]) -> Path:
    family_dir = "future" if int(spec["future_horizon"]) > 0 else "current"
    dep_part = spec["dep_var"].replace("_", "")
    reg_part = QUALITY_VAR_LABELS.get(spec["key_regressor"], spec["key_regressor"].replace("_", ""))
    group_dir = f"{dep_part}_{reg_part}"
    text_dir = table_dir / REGRESSION_TEXT_DIRNAME / family_dir / group_dir
    text_dir.mkdir(parents=True, exist_ok=True)
    return text_dir / f"reg_{spec['spec_id']}.txt"


def _run_single_spec(
    *,
    df: pd.DataFrame,
    spec: dict[str, Any],
    table_dir: Path,
    logger,
    universe_counts: dict[str, int],
    panel_ols_cls,
) -> tuple[dict[str, Any], dict[str, Any], Optional[Path]]:
    sample_df = df[df["PatentCount"] >= spec["sample_threshold"]].copy()
    threshold_rows = int(len(sample_df))
    threshold_firms = int(sample_df["stkcd"].nunique())

    rd_available_rows: Optional[int] = None
    rd_available_firms: Optional[int] = None
    if spec["rd_same_sample"]:
        sample_df = sample_df[
            (sample_df["year"] >= int(spec["rd_year_min"])) & (sample_df["year"] <= int(spec["rd_year_max"]))
        ].copy()
        rd_available_rows = int(sample_df.loc[sample_df[spec["rd_var"]].notna()].shape[0])
        rd_available_firms = int(sample_df.loc[sample_df[spec["rd_var"]].notna(), "stkcd"].nunique())
        sample_df = sample_df[sample_df[spec["rd_var"]].notna()].copy()

    logger.info(
        "[REG_START] spec_id=%s dep=%s x=%s sample_rule=%s horizon=%s add_count=%s add_rd=%s",
        spec["spec_id"],
        spec["dep_var"],
        spec["key_regressor"],
        spec["sample_rule"],
        int(spec["future_horizon"]),
        int(spec["add_count_control"]),
        int(spec["rd_var"] is not None and spec["key_regressor"] != spec["rd_var"] and spec["rd_var"] in spec["regressor_vars"]),
    )

    dep_source = spec["dep_source"]
    y_df = sample_df[sample_df[dep_source].notna()].copy()
    dropped_by_y_missing = int(len(sample_df) - len(y_df))

    x_df = y_df[y_df[spec["key_regressor"]].notna()].copy()
    dropped_by_x_missing = int(len(y_df) - len(x_df))

    controls = list(spec["controls"])
    if spec["add_count_control"]:
        controls.append("log_patent_count_ft")
    other_regressors = [column for column in spec["regressor_vars"] if column != spec["key_regressor"]]
    controls.extend(other_regressors)
    controls = _unique_in_order(controls)

    if controls:
        control_mask = x_df[controls].notna().all(axis=1)
        reg_df = x_df.loc[control_mask].copy()
    else:
        reg_df = x_df.copy()
    dropped_by_controls_missing = int(len(x_df) - len(reg_df))

    final_reg_firms = int(reg_df["stkcd"].nunique())
    final_reg_nobs_pre_fit = int(len(reg_df))
    logger.info(
        "[REG_SAMPLE] spec_id=%s financial_universe=%s patent_matched=%s threshold_sample=%s final_nobs=%s firms=%s",
        spec["spec_id"],
        universe_counts["financial_universe_rows"],
        universe_counts["patent_matched_rows"],
        threshold_rows,
        final_reg_nobs_pre_fit,
        final_reg_firms,
    )

    summary_row = {
        "spec_id": spec["spec_id"],
        "status": "failed",
        "model_family": spec["model_family"],
        "dep_var": spec["dep_var"],
        "dep_source": dep_source,
        "key_regressor": spec["key_regressor"],
        "rd_var": spec["rd_var"],
        "sample_threshold": int(spec["sample_threshold"]),
        "sample_rule": spec["sample_rule"],
        "year_range": spec["year_range"],
        "future_horizon": int(spec["future_horizon"]),
        "add_log_patent_count": bool(spec["add_count_control"]),
        "controls": " + ".join(controls),
        "coef": np.nan,
        "se": np.nan,
        "t": np.nan,
        "p": np.nan,
        "rd_coef": np.nan,
        "rd_se": np.nan,
        "rd_p": np.nan,
        "nobs": final_reg_nobs_pre_fit,
        "nfirms": final_reg_firms,
        "rsq_within": np.nan,
        "formula": "",
        "output_txt": None,
        "error": "",
    }
    sample_row = {
        "spec_id": spec["spec_id"],
        "dep_var": spec["dep_var"],
        "key_regressor": spec["key_regressor"],
        "sample_rule": spec["sample_rule"],
        "year_range": spec["year_range"] or f"{int(df['year'].min())}-{int(df['year'].max())}",
        "future_horizon": int(spec["future_horizon"]),
        "financial_universe_rows": universe_counts["financial_universe_rows"],
        "financial_universe_firms": universe_counts["financial_universe_firms"],
        "patent_matched_rows": universe_counts["patent_matched_rows"],
        "patent_matched_firms": universe_counts["patent_matched_firms"],
        "threshold_rows": threshold_rows,
        "threshold_firms": threshold_firms,
        "rd_available_rows": rd_available_rows,
        "rd_available_firms": rd_available_firms,
        "final_reg_nobs": final_reg_nobs_pre_fit,
        "final_reg_firms": final_reg_firms,
        "dropped_by_y_missing": dropped_by_y_missing,
        "dropped_by_x_missing": dropped_by_x_missing,
        "dropped_by_controls_missing": dropped_by_controls_missing,
    }

    if reg_df.empty:
        summary_row["error"] = "empty_sample_after_filters"
        logger.warning("[REG_FAIL] spec_id=%s error=%s", spec["spec_id"], summary_row["error"])
        return summary_row, sample_row, None

    if reg_df["stkcd"].nunique() < 2 or reg_df["year"].nunique() < 2:
        summary_row["error"] = "insufficient_firm_or_year_variation"
        logger.warning("[REG_FAIL] spec_id=%s error=%s", spec["spec_id"], summary_row["error"])
        return summary_row, sample_row, None

    model_columns = _unique_in_order([dep_source, *spec["regressor_vars"], *controls])
    panel_df = reg_df[["stkcd", "year", *model_columns]].set_index(["stkcd", "year"]).sort_index()
    rhs = " + ".join(_unique_in_order([*spec["regressor_vars"], *controls]))
    formula = f"{dep_source} ~ 1 + {rhs} + EntityEffects + TimeEffects"
    summary_row["formula"] = formula

    try:
        model = panel_ols_cls.from_formula(formula, data=panel_df, drop_absorbed=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)
        text_path = _build_regression_text_path(table_dir, spec)
        text_path.write_text(str(result.summary), encoding="utf-8")

        summary_row.update(
            {
                "status": "success",
                "coef": float(result.params.get(spec["key_regressor"], np.nan)),
                "se": float(result.std_errors.get(spec["key_regressor"], np.nan)),
                "t": float(result.tstats.get(spec["key_regressor"], np.nan)),
                "p": float(result.pvalues.get(spec["key_regressor"], np.nan)),
                "rd_coef": float(result.params.get(spec["rd_var"], np.nan)) if spec["rd_var"] is not None else np.nan,
                "rd_se": float(result.std_errors.get(spec["rd_var"], np.nan)) if spec["rd_var"] is not None else np.nan,
                "rd_p": float(result.pvalues.get(spec["rd_var"], np.nan)) if spec["rd_var"] is not None else np.nan,
                "nobs": int(result.nobs),
                "nfirms": int(panel_df.index.get_level_values(0).nunique()),
                "rsq_within": float(result.rsquared_within),
                "output_txt": repo_relative(text_path),
            }
        )
        sample_row["final_reg_nobs"] = int(result.nobs)
        sample_row["final_reg_firms"] = int(panel_df.index.get_level_values(0).nunique())
        logger.info(
            "[REG_DONE] spec_id=%s coef=%.6f se=%.6f p=%.4f output=%s",
            spec["spec_id"],
            summary_row["coef"],
            summary_row["se"],
            summary_row["p"],
            repo_relative(text_path),
        )
        return summary_row, sample_row, text_path
    except Exception as exc:  # pragma: no cover - exact exceptions depend on linearmodels version
        summary_row["error"] = str(exc)
        logger.warning("[REG_FAIL] spec_id=%s error=%s", spec["spec_id"], exc)
        return summary_row, sample_row, None


def _unique_in_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _plot_regression_coefficients(
    *,
    summary_table: pd.DataFrame,
    figure_dir: Path,
) -> Optional[Path]:
    if summary_table.empty:
        return None

    subset = summary_table[
        (summary_table["status"] == "success")
        & (summary_table["model_family"] == "main")
        & (summary_table["sample_threshold"] == 10)
        & (summary_table["add_log_patent_count"])
        & (summary_table["key_regressor"].isin(MAIN_QUALITY_VARS))
        & (summary_table["dep_var"].isin(["roa", "roe", "ebit_asset", "ebitda_asset", "profit_asset"]))
    ].copy()
    if subset.empty:
        return None

    subset["label"] = subset["dep_var"] + "\n" + subset["key_regressor"]
    subset = subset.reset_index(drop=True)

    plt.figure(figsize=(max(10, len(subset) * 0.6), 5.2))
    x_axis = np.arange(len(subset))
    y = subset["coef"].to_numpy(dtype=float)
    se = subset["se"].to_numpy(dtype=float)
    plt.errorbar(x_axis, y, yerr=1.96 * se, fmt="o", capsize=4)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xticks(x_axis, subset["label"], rotation=30, ha="right")
    plt.ylabel("Coefficient")
    plt.title("Main Patent Quality Coefficients (PatentCount >= 10)")
    plt.tight_layout()
    coefficient_fig = figure_dir / "fig_regression_coefficients.png"
    save_figure(coefficient_fig)
    return coefficient_fig


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="根据 firm_year_innovation 和财务面板运行固定效应回归")
    parser.add_argument("--experiment-id", required=True, help="实验 ID")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--firm-year-innovation-path", help="firm_year_innovation.parquet 路径")
    parser.add_argument("--financial-panel-path", help="共享 financial_annual_clean.parquet 路径")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    parser.add_argument("--year-min", type=int, default=2000, help="财务样本最小年份")
    parser.add_argument("--year-max", type=int, default=2023, help="财务样本最大年份")
    parser.add_argument("--sample-thresholds", nargs="+", type=int, default=list(DEFAULT_SAMPLE_THRESHOLDS), help="PatentCount 样本门槛列表")
    parser.add_argument("--winsor-lower", type=float, default=DEFAULT_WINSOR_LOWER, help="财务因变量按年 winsorize 下分位数")
    parser.add_argument("--winsor-upper", type=float, default=DEFAULT_WINSOR_UPPER, help="财务因变量按年 winsorize 上分位数")
    parser.add_argument("--rd-year-min", type=int, default=DEFAULT_RD_YEAR_MIN, help="RD 对照回归起始年份")
    parser.add_argument("--rd-year-max", type=int, default=DEFAULT_RD_YEAR_MAX, help="RD 对照回归终止年份")
    parser.add_argument("--future-horizons", nargs="+", type=int, default=list(DEFAULT_FUTURE_HORIZONS), help="未来财务回归 horizon 列表")
    parser.add_argument("--exact-date", action="store_true", help="使用 exact_date 模式，读取/输出 stage2_exact")
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
        sample_thresholds=args.sample_thresholds,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        rd_year_min=args.rd_year_min,
        rd_year_max=args.rd_year_max,
        future_horizons=args.future_horizons,
        exact_date=args.exact_date,
    )


if __name__ == "__main__":
    main()
