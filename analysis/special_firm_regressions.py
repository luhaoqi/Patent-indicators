from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Iterable, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from common.analysis import (
    INVALID_UCC_VALUES,
    PATENT_UCC_COL,
    PATENT_YEAR_COL,
    QUALITY_COL,
    normalize_string_series,
    to_numeric,
)
from common.paths import repo_relative


PATENT_ID_COL = "申请号"
REGRESSION_OUTPUT_CATEGORY = "回归分析"
STATIC_OUTPUT_GROUP = "静态横截面"
ABC_OUTPUT_GROUP = "ABC分解"
DYNAMIC_OUTPUT_GROUP = "动态企业内"
STD_ERROR_VERSIONS = ("cluster_firm", "no_cluster")
ABC_EFFECT_ROWS = (
    ("B-C", "ever_special_f", {"ever_special_f": 1.0}),
    ("A-B", "special_year_ft", {"special_year_ft": 1.0}),
    ("A-C", "ever_special_f + special_year_ft", {"ever_special_f": 1.0, "special_year_ft": 1.0}),
)


@dataclass(frozen=True)
class RegressionSpec:
    code: str
    output_group: str
    source_frame: str
    dependent_var: str
    sample_unit: str
    formula_label: str
    focal_terms: tuple[str, ...]
    weight_col: Optional[str] = None
    firm_fe: bool = False


@dataclass
class ModelResultBundle:
    params: pd.Series
    std_errors: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    covariance: pd.DataFrame
    nobs: int
    rsquared: float
    df_resid: float
    summary_text: str
    estimator: str
    formula_used: str
    cluster_applied: bool
    fit_warnings: tuple[str, ...]


REGRESSION_SPECS: tuple[RegressionSpec, ...] = (
    RegressionSpec(
        code="S1",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="patent_static",
        dependent_var="raw_q_pft",
        sample_unit="专利层",
        formula_label="raw_q_pft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="S2",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="patent_static",
        dependent_var="z_q_pft",
        sample_unit="专利层",
        formula_label="z_q_pft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="S3",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="patent_static",
        dependent_var="topk_q_pft",
        sample_unit="专利层",
        formula_label="topk_q_pft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="S4",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="firm_year_static",
        dependent_var="patent_count_ft",
        sample_unit="公司-年份层",
        formula_label="patent_count_ft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="S5",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="firm_year_static",
        dependent_var="log_patent_count_ft",
        sample_unit="公司-年份层",
        formula_label="log_patent_count_ft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="S6",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="firm_year_static",
        dependent_var="highq_count_ft",
        sample_unit="公司-年份层",
        formula_label="highq_count_ft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="S7",
        output_group=STATIC_OUTPUT_GROUP,
        source_frame="firm_year_static",
        dependent_var="log_highq_count_ft",
        sample_unit="公司-年份层",
        formula_label="log_highq_count_ft ~ ever_special_f + year FE",
        focal_terms=("ever_special_f",),
    ),
    RegressionSpec(
        code="G1",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="mean_raw_q_ft",
        sample_unit="公司-年份层",
        formula_label="mean_raw_q_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
        weight_col="patent_count_ft",
    ),
    RegressionSpec(
        code="G2",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="mean_z_q_ft",
        sample_unit="公司-年份层",
        formula_label="mean_z_q_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
        weight_col="patent_count_ft",
    ),
    RegressionSpec(
        code="G3",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="highq_share_ft",
        sample_unit="公司-年份层",
        formula_label="highq_share_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
        weight_col="patent_count_ft",
    ),
    RegressionSpec(
        code="G4",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="patent_count_ft",
        sample_unit="公司-年份层",
        formula_label="patent_count_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
    ),
    RegressionSpec(
        code="G5",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="log_patent_count_ft",
        sample_unit="公司-年份层",
        formula_label="log_patent_count_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
    ),
    RegressionSpec(
        code="G6",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="highq_count_ft",
        sample_unit="公司-年份层",
        formula_label="highq_count_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
    ),
    RegressionSpec(
        code="G7",
        output_group=ABC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="log_highq_count_ft",
        sample_unit="公司-年份层",
        formula_label="log_highq_count_ft ~ ever_special_f + special_year_ft + year FE",
        focal_terms=("ever_special_f", "special_year_ft"),
    ),
    RegressionSpec(
        code="D1",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="mean_raw_q_ft",
        sample_unit="公司-年份层",
        formula_label="mean_raw_q_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        weight_col="patent_count_ft",
        firm_fe=True,
    ),
    RegressionSpec(
        code="D2",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="mean_z_q_ft",
        sample_unit="公司-年份层",
        formula_label="mean_z_q_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        weight_col="patent_count_ft",
        firm_fe=True,
    ),
    RegressionSpec(
        code="D3",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="highq_share_ft",
        sample_unit="公司-年份层",
        formula_label="highq_share_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        weight_col="patent_count_ft",
        firm_fe=True,
    ),
    RegressionSpec(
        code="D4",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="patent_count_ft",
        sample_unit="公司-年份层",
        formula_label="patent_count_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        firm_fe=True,
    ),
    RegressionSpec(
        code="D5",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="log_patent_count_ft",
        sample_unit="公司-年份层",
        formula_label="log_patent_count_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        firm_fe=True,
    ),
    RegressionSpec(
        code="D6",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="highq_count_ft",
        sample_unit="公司-年份层",
        formula_label="highq_count_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        firm_fe=True,
    ),
    RegressionSpec(
        code="D7",
        output_group=DYNAMIC_OUTPUT_GROUP,
        source_frame="firm_year_dynamic",
        dependent_var="log_highq_count_ft",
        sample_unit="公司-年份层",
        formula_label="log_highq_count_ft ~ special_year_ft + firm FE + year FE",
        focal_terms=("special_year_ft",),
        firm_fe=True,
    ),
)


def prepare_special_regression_patent_frame(
    patent_df: pd.DataFrame,
    *,
    year_col: str,
    special_uccs: Iterable[str],
    firm_year_special: pd.DataFrame,
    topk_share: float = 0.10,
) -> pd.DataFrame:
    if not 0 < float(topk_share) <= 1:
        raise ValueError(f"topk_share 必须在 (0, 1] 区间内，当前={topk_share}")

    df = patent_df.copy()
    if PATENT_ID_COL not in df.columns:
        df[PATENT_ID_COL] = pd.RangeIndex(start=0, stop=len(df)).map(lambda value: f"row_{value:08d}")

    df["firm_id"] = normalize_string_series(df[PATENT_UCC_COL])
    df["year"] = to_numeric(df[year_col]).astype("Int64")
    df["patent_id"] = normalize_string_series(df[PATENT_ID_COL])
    df["raw_q_pft"] = to_numeric(df[QUALITY_COL])
    df["__rowid"] = np.arange(len(df), dtype=np.int64)

    label_panel = firm_year_special.copy()
    if year_col not in label_panel.columns and PATENT_YEAR_COL in label_panel.columns:
        label_panel = label_panel.rename(columns={PATENT_YEAR_COL: year_col})
    label_panel["firm_id"] = normalize_string_series(label_panel[PATENT_UCC_COL])
    label_panel["year"] = to_numeric(label_panel[year_col]).astype("Int64")
    label_panel["special_year_ft"] = to_numeric(label_panel["is_special_year"]).fillna(0).astype(int)
    label_panel = (
        label_panel[["firm_id", "year", "special_year_ft"]]
        .dropna(subset=["year"])
        .drop_duplicates(["firm_id", "year"], keep="last")
    )

    special_ucc_set = {
        value
        for value in pd.Series(list(special_uccs), dtype="string").fillna("").str.strip().tolist()
        if value and value not in INVALID_UCC_VALUES
    }

    df = df.merge(label_panel, on=["firm_id", "year"], how="left")
    df["special_year_ft"] = to_numeric(df["special_year_ft"]).fillna(0).astype(int)
    df["ever_special_f"] = df["firm_id"].isin(special_ucc_set).astype(int)

    df = df[
        df["year"].notna()
        & df["raw_q_pft"].notna()
        & df["firm_id"].notna()
        & (~df["firm_id"].isin(INVALID_UCC_VALUES))
    ].copy()

    df = df.sort_values(
        ["year", "raw_q_pft", "patent_id", "__rowid"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    year_counts = df.groupby("year")["patent_id"].transform("size")
    rank_index = df.groupby("year").cumcount().add(1)
    year_mean = df.groupby("year")["raw_q_pft"].transform("mean")
    year_sd = df.groupby("year")["raw_q_pft"].transform(lambda series: series.std(ddof=0))

    df["rank_q_pft"] = rank_index / year_counts
    df["z_q_pft"] = np.where(
        year_sd.fillna(0).to_numpy() == 0,
        0.0,
        ((df["raw_q_pft"] - year_mean) / year_sd).to_numpy(),
    )
    df["topk_q_pft"] = (df["rank_q_pft"] <= float(topk_share)).astype(int)
    df["abc_group"] = np.select(
        [
            (df["ever_special_f"] == 1) & (df["special_year_ft"] == 1),
            (df["ever_special_f"] == 1) & (df["special_year_ft"] == 0),
            (df["ever_special_f"] == 0),
        ],
        ["A", "B", "C"],
        default="C",
    )

    return (
        df[
            [
                "patent_id",
                "firm_id",
                "year",
                "raw_q_pft",
                "z_q_pft",
                "rank_q_pft",
                "topk_q_pft",
                "ever_special_f",
                "special_year_ft",
                "abc_group",
            ]
        ]
        .reset_index(drop=True)
    )


def build_special_regression_firm_year_panel(
    patent_frame: pd.DataFrame,
) -> pd.DataFrame:
    if patent_frame.empty:
        return pd.DataFrame(
            columns=[
                "firm_id",
                "year",
                "patent_count_ft",
                "highq_count_ft",
                "mean_raw_q_ft",
                "mean_z_q_ft",
                "highq_share_ft",
                "log_patent_count_ft",
                "log_highq_count_ft",
                "ever_special_f",
                "special_year_ft",
                "abc_group",
            ]
        )

    panel = (
        patent_frame.groupby(["firm_id", "year"], sort=False)
        .agg(
            patent_count_ft=("patent_id", "size"),
            highq_count_ft=("topk_q_pft", "sum"),
            mean_raw_q_ft=("raw_q_pft", "mean"),
            mean_z_q_ft=("z_q_pft", "mean"),
            ever_special_f=("ever_special_f", "max"),
            special_year_ft=("special_year_ft", "max"),
        )
        .reset_index()
    )
    panel["highq_share_ft"] = panel["highq_count_ft"] / panel["patent_count_ft"].replace(0, np.nan)
    panel["log_patent_count_ft"] = np.log1p(panel["patent_count_ft"])
    panel["log_highq_count_ft"] = np.log1p(panel["highq_count_ft"])
    panel["abc_group"] = np.select(
        [
            (panel["ever_special_f"] == 1) & (panel["special_year_ft"] == 1),
            (panel["ever_special_f"] == 1) & (panel["special_year_ft"] == 0),
            (panel["ever_special_f"] == 0),
        ],
        ["A", "B", "C"],
        default="C",
    )
    return panel


def run_special_firm_regressions(
    *,
    paths,
    category: str,
    patent_df: pd.DataFrame,
    dynamic_patent_df: pd.DataFrame,
    year_col: str,
    firm_year_special: pd.DataFrame,
    special_uccs: Iterable[str],
    topk_share: float = 0.10,
    logger,
) -> dict[str, object]:
    regression_table_dir = paths.table_subdir(category) / REGRESSION_OUTPUT_CATEGORY
    regression_table_dir.mkdir(parents=True, exist_ok=True)
    regression_data_dir = paths.data_subdir(category) / REGRESSION_OUTPUT_CATEGORY
    regression_data_dir.mkdir(parents=True, exist_ok=True)
    suite_start = time.perf_counter()

    patent_static = prepare_special_regression_patent_frame(
        patent_df,
        year_col=year_col,
        special_uccs=special_uccs,
        firm_year_special=firm_year_special,
        topk_share=topk_share,
    )
    patent_dynamic = prepare_special_regression_patent_frame(
        dynamic_patent_df,
        year_col=year_col,
        special_uccs=special_uccs,
        firm_year_special=firm_year_special,
        topk_share=topk_share,
    )
    firm_year_static = build_special_regression_firm_year_panel(patent_static)
    firm_year_dynamic = build_special_regression_firm_year_panel(patent_dynamic)

    dynamic_within = firm_year_dynamic.copy()
    if not dynamic_within.empty:
        dynamic_variation = dynamic_within.groupby("firm_id")["special_year_ft"].transform("nunique")
        dynamic_within = dynamic_within[dynamic_variation > 1].copy()

    panel_outputs = {
        "patent_static_panel_path": regression_data_dir / "patent_static_regression_panel.parquet",
        "patent_dynamic_panel_path": regression_data_dir / "patent_dynamic_regression_panel.parquet",
        "firm_year_static_panel_path": regression_data_dir / "firm_year_static_regression_panel.parquet",
        "firm_year_dynamic_panel_path": regression_data_dir / "firm_year_dynamic_regression_panel.parquet",
        "firm_year_dynamic_within_panel_path": regression_data_dir / "firm_year_dynamic_within_regression_panel.parquet",
    }
    patent_static.to_parquet(panel_outputs["patent_static_panel_path"], index=False)
    patent_dynamic.to_parquet(panel_outputs["patent_dynamic_panel_path"], index=False)
    firm_year_static.to_parquet(panel_outputs["firm_year_static_panel_path"], index=False)
    firm_year_dynamic.to_parquet(panel_outputs["firm_year_dynamic_panel_path"], index=False)
    dynamic_within.to_parquet(panel_outputs["firm_year_dynamic_within_panel_path"], index=False)

    frame_lookup = {
        "patent_static": patent_static,
        "firm_year_static": firm_year_static,
        "firm_year_dynamic": firm_year_dynamic,
    }

    logger.info(
        "[%s][回归分析] 面板准备完成: patent_static=%s, patent_dynamic=%s, firm_year_static=%s, firm_year_dynamic=%s, dynamic_within=%s, topk_share=%.3f",
        category,
        len(patent_static),
        len(patent_dynamic),
        len(firm_year_static),
        len(firm_year_dynamic),
        len(dynamic_within),
        float(topk_share),
    )

    all_rows: list[dict[str, Any]] = []
    table_outputs: list[str] = []
    total_runs = len(REGRESSION_SPECS) * len(STD_ERROR_VERSIONS)
    completed_runs = 0
    warning_runs = 0
    for spec in REGRESSION_SPECS:
        source_df = dynamic_within if spec.firm_fe else frame_lookup[spec.source_frame]
        for se_version in STD_ERROR_VERSIONS:
            run_start = time.perf_counter()
            completed_runs += 1
            output_dir = regression_table_dir / spec.output_group / se_version
            output_dir.mkdir(parents=True, exist_ok=True)
            reg_csv_path = output_dir / f"reg_{spec.code}.csv"
            reg_txt_path = output_dir / f"reg_{spec.code}.txt"

            logger.info(
                "[%s][回归分析][%s][%s] 开始 (%s/%s) 样本=%s, 公式=%s",
                category,
                spec.code,
                se_version,
                completed_runs,
                total_runs,
                len(source_df),
                spec.formula_label,
            )

            try:
                result_bundle, sample_df = _fit_regression(
                    spec=spec,
                    source_df=source_df,
                    se_version=se_version,
                )
                rows = _build_regression_rows(
                    spec=spec,
                    result_bundle=result_bundle,
                    sample_df=sample_df,
                    se_version=se_version,
                    topk_share=topk_share,
                )
                warning_message = _build_model_warning_message(
                    spec=spec,
                    result_bundle=result_bundle,
                    rows=rows,
                )
                _attach_warning_columns(
                    rows,
                    warning_message=warning_message,
                )
                summary_text = _render_regression_text(
                    spec=spec,
                    result_bundle=result_bundle,
                    sample_df=sample_df,
                    rows=rows,
                    se_version=se_version,
                    topk_share=topk_share,
                    warning_message=warning_message,
                )
                elapsed = time.perf_counter() - run_start
                cumulative = time.perf_counter() - suite_start
                logger.info(
                    "[%s][回归分析][%s][%s] 完成 用时 %.1fs, 累计 %.1fs, nobs=%s",
                    category,
                    spec.code,
                    se_version,
                    elapsed,
                    cumulative,
                    int(result_bundle.nobs),
                )
                if warning_message:
                    warning_runs += 1
                    logger.warning(
                        "[%s][回归分析][%s][%s] 推断警告: %s",
                        category,
                        spec.code,
                        se_version,
                        warning_message,
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - run_start
                cumulative = time.perf_counter() - suite_start
                logger.warning(
                    "[%s][回归分析][%s][%s] 失败 用时 %.1fs, 累计 %.1fs: %s",
                    category,
                    spec.code,
                    se_version,
                    elapsed,
                    cumulative,
                    exc,
                )
                rows = _build_skipped_rows(
                    spec=spec,
                    source_df=source_df,
                    se_version=se_version,
                    topk_share=topk_share,
                    reason=str(exc),
                )
                summary_text = f"{spec.code} ({se_version}) skipped\nreason: {exc}\n"
                warning_runs += 1

            rows.to_csv(reg_csv_path, index=False, encoding="utf-8-sig")
            reg_txt_path.write_text(summary_text, encoding="utf-8")
            table_outputs.extend([repo_relative(reg_csv_path), repo_relative(reg_txt_path)])
            all_rows.extend(rows.to_dict(orient="records"))

    summary_table = pd.DataFrame(all_rows)
    overall_summary_path = regression_table_dir / "tbl_regression_summary.csv"
    summary_table.to_csv(overall_summary_path, index=False, encoding="utf-8-sig")
    table_outputs.append(repo_relative(overall_summary_path))

    for output_group in (STATIC_OUTPUT_GROUP, ABC_OUTPUT_GROUP, DYNAMIC_OUTPUT_GROUP):
        group_summary = summary_table[summary_table["analysis_type"] == output_group].copy()
        group_summary_path = regression_table_dir / output_group / "tbl_regression_summary.csv"
        group_summary_path.parent.mkdir(parents=True, exist_ok=True)
        group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")
        table_outputs.append(repo_relative(group_summary_path))

        for se_version in STD_ERROR_VERSIONS:
            version_summary = group_summary[group_summary["se_version"] == se_version].copy()
            version_summary_path = regression_table_dir / output_group / se_version / "tbl_regression_summary.csv"
            version_summary_path.parent.mkdir(parents=True, exist_ok=True)
            version_summary.to_csv(version_summary_path, index=False, encoding="utf-8-sig")
            table_outputs.append(repo_relative(version_summary_path))

    total_elapsed = time.perf_counter() - suite_start
    logger.info(
        "[%s][回归分析] 全部完成: %s 个回归版本, 其中警告/失败 %s 个, 总用时 %.1fs",
        category,
        total_runs,
        warning_runs,
        total_elapsed,
    )

    return {
        "topk_share": float(topk_share),
        "data_outputs": [repo_relative(path) for path in panel_outputs.values()],
        "table_outputs": table_outputs,
        "panel_rows": {
            "patent_static": int(len(patent_static)),
            "patent_dynamic": int(len(patent_dynamic)),
            "firm_year_static": int(len(firm_year_static)),
            "firm_year_dynamic": int(len(firm_year_dynamic)),
            "firm_year_dynamic_within": int(len(dynamic_within)),
        },
        "timing_seconds": {
            "total": float(total_elapsed),
        },
        "warning_runs": int(warning_runs),
    }


def _fit_regression(
    *,
    spec: RegressionSpec,
    source_df: pd.DataFrame,
    se_version: str,
) -> tuple[ModelResultBundle, pd.DataFrame]:
    if spec.firm_fe:
        return _fit_dynamic_fe_regression(spec=spec, source_df=source_df, se_version=se_version)
    return _fit_cross_sectional_regression(spec=spec, source_df=source_df, se_version=se_version)


def _fit_cross_sectional_regression(
    *,
    spec: RegressionSpec,
    source_df: pd.DataFrame,
    se_version: str,
) -> tuple[ModelResultBundle, pd.DataFrame]:
    formula_used = _build_formula(spec, include_firm_fe=spec.firm_fe)
    required_columns = ["firm_id", "year", spec.dependent_var, *spec.focal_terms]
    if spec.weight_col:
        required_columns.append(spec.weight_col)
    working = (
        source_df[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    if working.empty:
        raise ValueError("无有效样本")
    working = _coerce_model_frame(
        working,
        numeric_columns=[spec.dependent_var, *spec.focal_terms, *( [spec.weight_col] if spec.weight_col else [] )],
    )

    if spec.weight_col:
        model = smf.wls(formula_used, data=working, weights=working[spec.weight_col])
    else:
        model = smf.ols(formula_used, data=working)

    cluster_applied = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if se_version == "cluster_firm" and working["firm_id"].nunique() >= 2:
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": working["firm_id"], "use_correction": True},
            )
            cluster_applied = True
        else:
            result = model.fit()

        bundle = ModelResultBundle(
            params=result.params,
            std_errors=result.bse,
            tstats=result.tvalues,
            pvalues=result.pvalues,
            covariance=result.cov_params(),
            nobs=int(result.nobs),
            rsquared=float(getattr(result, "rsquared", np.nan)),
            df_resid=float(getattr(result, "df_resid", np.nan)),
            summary_text=str(result.summary()),
            estimator="statsmodels",
            formula_used=formula_used,
            cluster_applied=cluster_applied,
            fit_warnings=_format_caught_warnings(caught),
        )
    return bundle, working


def _fit_dynamic_fe_regression(
    *,
    spec: RegressionSpec,
    source_df: pd.DataFrame,
    se_version: str,
) -> tuple[ModelResultBundle, pd.DataFrame]:
    required_columns = ["firm_id", "year", spec.dependent_var, *spec.focal_terms]
    if spec.weight_col:
        required_columns.append(spec.weight_col)
    working = (
        source_df[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    if working.empty:
        raise ValueError("无有效样本")
    working = _coerce_model_frame(
        working,
        numeric_columns=[spec.dependent_var, *spec.focal_terms, *( [spec.weight_col] if spec.weight_col else [] )],
    )

    try:
        from linearmodels.panel import PanelOLS

        panel = working.sort_values(["firm_id", "year"]).set_index(["firm_id", "year"])
        exog = panel[list(spec.focal_terms)]
        weights = panel[spec.weight_col] if spec.weight_col else None
        model = PanelOLS(
            panel[spec.dependent_var],
            exog,
            weights=weights,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
        )
        cluster_applied = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if se_version == "cluster_firm" and working["firm_id"].nunique() >= 2:
                result = model.fit(cov_type="clustered", cluster_entity=True)
                cluster_applied = True
            else:
                result = model.fit(cov_type="unadjusted")

            bundle = ModelResultBundle(
                params=result.params,
                std_errors=result.std_errors,
                tstats=result.tstats,
                pvalues=result.pvalues,
                covariance=result.cov,
                nobs=int(result.nobs),
                rsquared=float(getattr(result, "rsquared_within", np.nan)),
                df_resid=float(getattr(result, "df_resid", np.nan)),
                summary_text=str(result.summary),
                estimator="linearmodels.PanelOLS",
                formula_used=spec.formula_label,
                cluster_applied=cluster_applied,
                fit_warnings=_format_caught_warnings(caught),
            )
        return bundle, working
    except Exception:
        return _fit_cross_sectional_regression(
            spec=spec,
            source_df=source_df,
            se_version=se_version,
        )


def _build_regression_rows(
    *,
    spec: RegressionSpec,
    result_bundle: ModelResultBundle,
    sample_df: pd.DataFrame,
    se_version: str,
    topk_share: float,
) -> pd.DataFrame:
    base = {
        "regression_id": spec.code,
        "analysis_type": spec.output_group,
        "sample_unit": spec.sample_unit,
        "dependent_var": spec.dependent_var,
        "formula_label": spec.formula_label,
        "formula_used": result_bundle.formula_used,
        "se_version": se_version,
        "cluster_applied": int(result_bundle.cluster_applied),
        "cluster_dimension": "firm_id" if se_version == "cluster_firm" else "",
        "weighted": int(spec.weight_col is not None),
        "weight_var": spec.weight_col or "",
        "year_fe": 1,
        "firm_fe": int(spec.firm_fe),
        "nobs": int(result_bundle.nobs),
        "n_firms": int(sample_df["firm_id"].nunique()) if "firm_id" in sample_df.columns else np.nan,
        "n_years": int(sample_df["year"].nunique()) if "year" in sample_df.columns else np.nan,
        "rsquared": result_bundle.rsquared,
        "topk_share": float(topk_share),
        "estimator": result_bundle.estimator,
        "status": "ok",
        "message": "",
        "warning_flag": 0,
        "warning_message": "",
    }

    rows: list[dict[str, Any]] = []
    if spec.output_group == ABC_OUTPUT_GROUP:
        for effect_label, variable_label, weights in ABC_EFFECT_ROWS:
            coef, se, t_stat, p_value = _linear_combination(result_bundle, weights)
            rows.append(
                {
                    **base,
                    "effect_label": effect_label,
                    "effect_variable": variable_label,
                    "coef": coef,
                    "se": se,
                    "t": t_stat,
                    "p": p_value,
                }
            )
        return pd.DataFrame(rows)

    term = spec.focal_terms[0]
    rows.append(
        {
            **base,
            "effect_label": term,
            "effect_variable": term,
            "coef": _safe_get(result_bundle.params, term),
            "se": _safe_get(result_bundle.std_errors, term),
            "t": _safe_get(result_bundle.tstats, term),
            "p": _safe_get(result_bundle.pvalues, term),
        }
    )
    return pd.DataFrame(rows)


def _build_skipped_rows(
    *,
    spec: RegressionSpec,
    source_df: pd.DataFrame,
    se_version: str,
    topk_share: float,
    reason: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "regression_id": spec.code,
                "analysis_type": spec.output_group,
                "sample_unit": spec.sample_unit,
                "dependent_var": spec.dependent_var,
                "formula_label": spec.formula_label,
                "formula_used": _build_formula(spec, include_firm_fe=spec.firm_fe),
                "se_version": se_version,
                "cluster_applied": 0,
                "cluster_dimension": "firm_id" if se_version == "cluster_firm" else "",
                "weighted": int(spec.weight_col is not None),
                "weight_var": spec.weight_col or "",
                "year_fe": 1,
                "firm_fe": int(spec.firm_fe),
                "nobs": int(len(source_df)),
                "n_firms": int(source_df["firm_id"].nunique()) if "firm_id" in source_df.columns else np.nan,
                "n_years": int(source_df["year"].nunique()) if "year" in source_df.columns else np.nan,
                "rsquared": np.nan,
                "topk_share": float(topk_share),
                "estimator": "",
                "status": "skipped",
                "message": reason,
                "warning_flag": 1,
                "warning_message": reason,
                "effect_label": "",
                "effect_variable": "",
                "coef": np.nan,
                "se": np.nan,
                "t": np.nan,
                "p": np.nan,
            }
        ]
    )


def _render_regression_text(
    *,
    spec: RegressionSpec,
    result_bundle: ModelResultBundle,
    sample_df: pd.DataFrame,
    rows: pd.DataFrame,
    se_version: str,
    topk_share: float,
    warning_message: str,
) -> str:
    header = [
        f"regression_id: {spec.code}",
        f"analysis_type: {spec.output_group}",
        f"sample_unit: {spec.sample_unit}",
        f"formula_label: {spec.formula_label}",
        f"formula_used: {result_bundle.formula_used}",
        f"se_version: {se_version}",
        f"cluster_applied: {int(result_bundle.cluster_applied)}",
        f"weighted: {int(spec.weight_col is not None)}",
        f"weight_var: {spec.weight_col or ''}",
        f"nobs: {int(result_bundle.nobs)}",
        f"n_firms: {int(sample_df['firm_id'].nunique()) if 'firm_id' in sample_df.columns else 0}",
        f"n_years: {int(sample_df['year'].nunique()) if 'year' in sample_df.columns else 0}",
        f"rsquared: {_format_float(result_bundle.rsquared)}",
        f"topk_share: {_format_float(topk_share)}",
        f"warning_message: {warning_message}",
        "",
        "key_effects:",
        rows.to_string(index=False),
        "",
        "captured_warnings:",
        "\n".join(result_bundle.fit_warnings) if result_bundle.fit_warnings else "",
        "",
        "model_summary:",
        result_bundle.summary_text,
    ]
    return "\n".join(header) + "\n"


def _build_formula(spec: RegressionSpec, *, include_firm_fe: bool) -> str:
    rhs_terms = list(spec.focal_terms)
    rhs_terms.append("C(year)")
    if include_firm_fe:
        rhs_terms.append("C(firm_id)")
    return f"{spec.dependent_var} ~ {' + '.join(rhs_terms)}"


def _coerce_model_frame(frame: pd.DataFrame, *, numeric_columns: list[str]) -> pd.DataFrame:
    working = frame.copy()
    working["firm_id"] = working["firm_id"].astype("string").fillna("").astype(str)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype(int)
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").astype(float)
    return working


def _format_caught_warnings(caught: list[warnings.WarningMessage]) -> tuple[str, ...]:
    messages: list[str] = []
    seen: set[str] = set()
    for item in caught:
        category = getattr(item.category, "__name__", "Warning")
        text = str(item.message).strip()
        payload = f"{category}: {text}" if text else category
        if payload and payload not in seen:
            seen.add(payload)
            messages.append(payload)
    return tuple(messages)


def _build_model_warning_message(
    *,
    spec: RegressionSpec,
    result_bundle: ModelResultBundle,
    rows: pd.DataFrame,
) -> str:
    messages: list[str] = list(result_bundle.fit_warnings)

    invalid_cols: list[str] = []
    for column in ("coef", "se", "t", "p"):
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            if values.isna().any() or np.isinf(values).any():
                invalid_cols.append(column)
    if invalid_cols:
        messages.append(f"non-finite regression outputs detected in columns: {', '.join(sorted(set(invalid_cols)))}")

    if spec.output_group == ABC_OUTPUT_GROUP:
        bad_effects = rows.loc[
            pd.to_numeric(rows["se"], errors="coerce").isna()
            | pd.to_numeric(rows["t"], errors="coerce").isna()
            | pd.to_numeric(rows["p"], errors="coerce").isna(),
            "effect_label",
        ].astype(str).tolist()
        if bad_effects:
            messages.append(f"ABC decomposition has invalid inference for effects: {', '.join(bad_effects)}")
    else:
        if pd.to_numeric(rows["se"], errors="coerce").isna().any():
            messages.append(f"invalid standard error for effect: {rows['effect_label'].astype(str).iloc[0]}")

    unique_messages: list[str] = []
    seen: set[str] = set()
    for message in messages:
        message = message.strip()
        if message and message not in seen:
            seen.add(message)
            unique_messages.append(message)
    return " | ".join(unique_messages)


def _attach_warning_columns(
    rows: pd.DataFrame,
    *,
    warning_message: str,
) -> None:
    if warning_message:
        rows["warning_flag"] = 1
        rows["warning_message"] = warning_message
        rows["status"] = rows["status"].where(rows["status"] != "ok", "warning")
    else:
        rows["warning_flag"] = rows["warning_flag"].fillna(0).astype(int)
        rows["warning_message"] = rows["warning_message"].fillna("")


def _linear_combination(
    result_bundle: ModelResultBundle,
    weights: dict[str, float],
) -> tuple[float, float, float, float]:
    if any(parameter not in result_bundle.params.index for parameter in weights):
        return (np.nan, np.nan, np.nan, np.nan)

    coef = sum(float(result_bundle.params[parameter]) * weight for parameter, weight in weights.items())
    variance = 0.0
    for left_name, left_weight in weights.items():
        for right_name, right_weight in weights.items():
            variance += (
                float(result_bundle.covariance.loc[left_name, right_name]) * float(left_weight) * float(right_weight)
            )
    variance = max(float(variance), 0.0)
    se = float(np.sqrt(variance))
    if se == 0:
        return coef, 0.0, np.nan, np.nan

    t_stat = coef / se
    if result_bundle.cluster_applied or pd.isna(result_bundle.df_resid):
        p_value = float(2 * stats.norm.sf(abs(t_stat)))
    else:
        p_value = float(2 * stats.t.sf(abs(t_stat), df=max(result_bundle.df_resid, 1)))
    return coef, se, t_stat, p_value


def _safe_get(series: pd.Series, key: str) -> float:
    value = series.get(key, np.nan)
    if pd.isna(value):
        return np.nan
    return float(value)


def _format_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"
