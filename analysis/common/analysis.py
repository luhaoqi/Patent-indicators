from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl


PATENT_UCC_COL = "统一社会信用代码"
PATENT_YEAR_COL = "申请年份"
PUBLIC_YEAR_COL = "公开公告年份"
QUALITY_COL = "Quality_q"
BS_COL = "BS"
SPECIAL_YEAR_COL = "年份"
SPECIAL_UCC_COL = "统一社会信用代码"

DEFAULT_SPECIAL_FLAG_COLS: tuple[str, ...] = (
    "高新技术企业",
    "专精特新企业",
    "制造业单项冠军企业",
    "国家技术创新示范企业",
    "科创企业称号总数",
)

INVALID_UCC_VALUES = {"", "-", "nan", "NaN", "None", "NULL", "null"}


def resolve_patent_year_col(columns: Sequence[str], *, exact_date: bool = False) -> str:
    names = set(columns)
    if exact_date and PUBLIC_YEAR_COL in names:
        return PUBLIC_YEAR_COL
    if PATENT_YEAR_COL in names:
        return PATENT_YEAR_COL
    if PUBLIC_YEAR_COL in names:
        return PUBLIC_YEAR_COL
    raise KeyError("未找到可用的专利年份列")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def filter_patents(
    df: pd.DataFrame,
    *,
    year_col: str = PATENT_YEAR_COL,
    quality_col: str = QUALITY_COL,
    bs_col: str = BS_COL,
    exclude_years: Optional[Sequence[int]] = None,
    quality_min: Optional[float] = None,
    bs_min: Optional[float] = 1e-6,
) -> pd.DataFrame:
    out = df.copy()
    if year_col in out.columns:
        out[year_col] = to_numeric(out[year_col]).astype("Int64")
        if exclude_years:
            out = out[~out[year_col].isin(exclude_years)]

    if quality_col in out.columns:
        out[quality_col] = to_numeric(out[quality_col])
        if quality_min is not None:
            out = out[out[quality_col].fillna(0) >= quality_min]

    if bs_min is not None and bs_col in out.columns:
        out[bs_col] = to_numeric(out[bs_col])
        out = out[out[bs_col].fillna(0) >= bs_min]

    return out


def describe_series(
    series: pd.Series,
    *,
    percentiles: Sequence[float] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
) -> dict[str, float]:
    numeric = to_numeric(series).dropna()
    if numeric.empty:
        result: dict[str, float] = {"N": 0, "Mean": np.nan, "Std. Dev.": np.nan, "Min": np.nan}
        for percentile in percentiles:
            result[_percentile_label(percentile)] = np.nan
        result["Max"] = np.nan
        return result

    result = {
        "N": int(numeric.count()),
        "Mean": float(numeric.mean()),
        "Std. Dev.": float(numeric.std()),
        "Min": float(numeric.min()),
    }
    for percentile in percentiles:
        result[_percentile_label(percentile)] = float(numeric.quantile(percentile))
    result["Max"] = float(numeric.max())
    return result


def build_descriptive_table(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({name: describe_series(series) for name, series in series_map.items()})


def load_special_panel(
    special_df: pd.DataFrame,
    *,
    ucc_col: str = SPECIAL_UCC_COL,
    year_col: str = SPECIAL_YEAR_COL,
) -> pd.DataFrame:
    out = special_df.copy()
    out[ucc_col] = normalize_string_series(out[ucc_col])
    if year_col in out.columns:
        out[year_col] = to_numeric(out[year_col]).astype("Int64")
    return out


def compute_special_ucc_set(
    special_df: pd.DataFrame,
    *,
    ucc_col: str = SPECIAL_UCC_COL,
) -> set[str]:
    ucc = normalize_string_series(special_df[ucc_col])
    return {value for value in ucc.tolist() if value and value not in INVALID_UCC_VALUES}


def build_firm_year_special_panel(
    special_df: pd.DataFrame,
    *,
    ucc_col: str = SPECIAL_UCC_COL,
    year_col: str = SPECIAL_YEAR_COL,
    flag_candidates: Sequence[str] = DEFAULT_SPECIAL_FLAG_COLS,
) -> pd.DataFrame:
    df = load_special_panel(special_df, ucc_col=ucc_col, year_col=year_col)
    available = [column for column in flag_candidates if column in df.columns]
    if not available:
        raise ValueError("special_df 中未找到可用的特殊企业标识列")

    if "科创企业称号总数" in available:
        is_special_year = to_numeric(df["科创企业称号总数"]).fillna(0) > 0
    else:
        total = pd.Series(0.0, index=df.index)
        for column in available:
            total = total.add(to_numeric(df[column]).fillna(0), fill_value=0)
        is_special_year = total > 0

    panel = (
        df.loc[df[year_col].notna(), [ucc_col, year_col]]
        .assign(is_special_year=is_special_year.loc[df[year_col].notna()].astype(int))
        .drop_duplicates()
    )
    return (
        panel.groupby([ucc_col, year_col], as_index=False)
        .agg(is_special_year=("is_special_year", "max"))
        .rename(columns={ucc_col: PATENT_UCC_COL, year_col: PATENT_YEAR_COL})
    )


def prepare_valid_ucc_patents(
    patent_df: pd.DataFrame,
    *,
    ucc_col: str = PATENT_UCC_COL,
    quality_col: str = QUALITY_COL,
) -> pd.DataFrame:
    out = patent_df.copy()
    out[ucc_col] = normalize_string_series(out[ucc_col])
    out[quality_col] = to_numeric(out[quality_col])
    mask = out[ucc_col].notna() & (~out[ucc_col].isin(INVALID_UCC_VALUES))
    return out[mask & out[quality_col].notna()].copy()


def _invalid_ucc_expr(column: str) -> pl.Expr:
    return pl.col(column).is_in(sorted(INVALID_UCC_VALUES))


def _prepare_valid_ucc_patents_polars(
    patent_df: pd.DataFrame,
    *,
    ucc_col: str = PATENT_UCC_COL,
    quality_col: str = QUALITY_COL,
    extra_columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    columns = [ucc_col, quality_col]
    if extra_columns:
        for column in extra_columns:
            if column not in columns:
                columns.append(column)

    df = pl.from_pandas(patent_df[columns], include_index=False)
    exprs: list[pl.Expr] = [
        pl.col(ucc_col).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().alias(ucc_col),
        pl.col(quality_col).cast(pl.Float64, strict=False).alias(quality_col),
    ]
    for column in columns:
        if column in (ucc_col, quality_col):
            continue
        if column == PATENT_YEAR_COL or column == SPECIAL_YEAR_COL:
            exprs.append(pl.col(column).cast(pl.Int64, strict=False).alias(column))
        elif column == "is_special_year":
            exprs.append(pl.col(column).cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int8).alias(column))
        else:
            exprs.append(pl.col(column))

    return (
        df.with_columns(exprs)
        .filter(~_invalid_ucc_expr(ucc_col))
        .filter(pl.col(quality_col).is_not_null())
    )


def build_company_special_panel(
    patent_df: pd.DataFrame,
    special_df: pd.DataFrame,
    *,
    quality_threshold: float,
    ucc_col: str = PATENT_UCC_COL,
    quality_col: str = QUALITY_COL,
) -> pd.DataFrame:
    df = prepare_valid_ucc_patents(patent_df, ucc_col=ucc_col, quality_col=quality_col)
    special_uccs = compute_special_ucc_set(special_df, ucc_col=SPECIAL_UCC_COL)
    company_agg = (
        df.groupby(ucc_col, dropna=False)
        .agg(
            total_patents=(quality_col, "size"),
            high_q_count=(quality_col, lambda series: int((to_numeric(series).fillna(-np.inf) >= quality_threshold).sum())),
            mean_quality=(quality_col, "mean"),
            max_quality=(quality_col, "max"),
        )
        .reset_index()
    )
    company_agg["log_total_patents"] = np.log1p(company_agg["total_patents"])
    company_agg["is_special"] = company_agg[ucc_col].isin(special_uccs).astype(int)
    return company_agg


def build_company_special_panel_from_ucc_set(
    patent_df: pd.DataFrame,
    special_uccs: Iterable[str],
    *,
    quality_threshold: float,
    ucc_col: str = PATENT_UCC_COL,
    quality_col: str = QUALITY_COL,
) -> pd.DataFrame:
    special_ucc_set = {value for value in special_uccs if value and value not in INVALID_UCC_VALUES}
    df = _prepare_valid_ucc_patents_polars(
        patent_df,
        ucc_col=ucc_col,
        quality_col=quality_col,
    )
    company_agg = (
        df.group_by(ucc_col)
        .agg(
            pl.len().alias("total_patents"),
            (pl.col(quality_col) >= quality_threshold).sum().cast(pl.Int64).alias("high_q_count"),
            pl.col(quality_col).mean().alias("mean_quality"),
            pl.col(quality_col).max().alias("max_quality"),
        )
        .with_columns(
            pl.col("total_patents").cast(pl.Float64).log1p().alias("log_total_patents"),
            pl.col(ucc_col).is_in(sorted(special_ucc_set)).cast(pl.Int8).alias("is_special"),
        )
        .to_pandas()
    )
    company_agg["log_total_patents"] = np.log1p(company_agg["total_patents"])
    company_agg["is_special"] = company_agg["is_special"].astype(int)
    return company_agg


def attach_special_year_labels(
    patent_df: pd.DataFrame,
    firm_year_special: pd.DataFrame,
    *,
    policy_start_year: Optional[int] = 2008,
    exclude_years: Optional[Sequence[int]] = None,
    quality_min: Optional[float] = 1e-5,
    bs_min: Optional[float] = 1e-6,
    ucc_col: str = PATENT_UCC_COL,
    year_col: str = PATENT_YEAR_COL,
) -> pd.DataFrame:
    out = patent_df.copy()
    out[ucc_col] = normalize_string_series(out[ucc_col])
    out[year_col] = to_numeric(out[year_col]).astype("Int64")
    if policy_start_year is not None:
        out = out[out[year_col] >= policy_start_year]
    out = filter_patents(
        out,
        year_col=year_col,
        exclude_years=exclude_years,
        quality_min=quality_min,
        bs_min=bs_min,
    )
    merged = out.merge(firm_year_special, on=[ucc_col, year_col], how="left")
    merged["is_special_year"] = to_numeric(merged["is_special_year"]).fillna(0).astype(int)
    return merged


def build_company_year_special_panel(
    p_dyn: pd.DataFrame,
    *,
    quality_threshold: float,
    ucc_col: str = PATENT_UCC_COL,
    year_col: str = PATENT_YEAR_COL,
    quality_col: str = QUALITY_COL,
) -> pd.DataFrame:
    df = _prepare_valid_ucc_patents_polars(
        p_dyn,
        ucc_col=ucc_col,
        quality_col=quality_col,
        extra_columns=[year_col, "is_special_year"],
    )
    agg = (
        df.group_by([ucc_col, year_col])
        .agg(
            pl.len().alias("total_patents"),
            (pl.col(quality_col) >= quality_threshold).sum().cast(pl.Int64).alias("high_q_count"),
            pl.col(quality_col).mean().alias("mean_quality"),
            pl.col(quality_col).max().alias("max_quality"),
            pl.col("is_special_year").max().cast(pl.Int8).alias("is_special_year"),
        )
        .with_columns(pl.col("total_patents").cast(pl.Float64).log1p().alias("log_total_patents"))
        .to_pandas()
    )
    agg["log_total_patents"] = np.log1p(agg["total_patents"])
    return agg


def build_company_year_abc_panel(
    p_dyn: pd.DataFrame,
    *,
    quality_threshold: float,
    ucc_col: str = PATENT_UCC_COL,
    year_col: str = PATENT_YEAR_COL,
    quality_col: str = QUALITY_COL,
) -> pd.DataFrame:
    df = _prepare_valid_ucc_patents_polars(
        p_dyn,
        ucc_col=ucc_col,
        quality_col=quality_col,
        extra_columns=[year_col, "is_special_year"],
    )
    df = df.filter(pl.col(year_col).is_not_null())
    ever_special = (
        df.group_by(ucc_col)
        .agg(pl.col("is_special_year").max().cast(pl.Int8).alias("ever_special"))
    )
    df = (
        df.join(ever_special, on=ucc_col, how="left")
        .with_columns(
            pl.when((pl.col("ever_special") == 1) & (pl.col("is_special_year") == 1))
            .then(pl.lit("A_treated_year"))
            .when((pl.col("ever_special") == 1) & (pl.col("is_special_year") == 0))
            .then(pl.lit("B_same_firm_other_year"))
            .otherwise(pl.lit("C_never_treated"))
            .alias("firm_group_3"),
            (pl.col(quality_col) >= quality_threshold).cast(pl.Int8).alias("_high_q"),
        )
    )
    agg = (
        df.group_by([ucc_col, year_col])
        .agg(
            pl.len().alias("total_patents"),
            pl.col("_high_q").sum().cast(pl.Int64).alias("high_q_count"),
            pl.col(quality_col).mean().alias("mean_quality"),
            pl.col(quality_col).max().alias("max_quality"),
            pl.col("is_special_year").max().cast(pl.Int8).alias("is_special_year"),
            pl.col("ever_special").max().cast(pl.Int8).alias("ever_special"),
            pl.col("firm_group_3").first().alias("firm_group_3"),
        )
        .with_columns(pl.col("total_patents").cast(pl.Float64).log1p().alias("log_total_patents"))
        .to_pandas()
    )
    agg["log_total_patents"] = np.log1p(agg["total_patents"])
    return agg


def build_abc_summary_table(company_year_abc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group_df in company_year_abc.groupby("firm_group_3", dropna=False):
        rows.append(
            {
                "firm_group_3": group_name,
                "n_rows": int(len(group_df)),
                "total_patents_sum": int(group_df["total_patents"].sum()),
                "high_q_share": float(
                    group_df["high_q_count"].sum() / group_df["total_patents"].replace(0, np.nan).sum()
                )
                if group_df["total_patents"].sum()
                else np.nan,
                "mean_quality_mean": float(group_df["mean_quality"].mean()),
                "mean_quality_p50": float(group_df["mean_quality"].quantile(0.50)),
                "mean_quality_p90": float(group_df["mean_quality"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def build_group_comparison_table(
    frame: pd.DataFrame,
    *,
    group_col: str,
    var_specs: Optional[Sequence[tuple[str, str, bool]]] = None,
    percentiles: Sequence[float] = (0.50, 0.90),
    digits: int = 3,
    use_mannwhitney: bool = False,
) -> pd.DataFrame:
    from scipy import stats

    if var_specs is None:
        var_specs = (
            ("Total patents", "total_patents", True),
            ("Mean Quality_q", "mean_quality", False),
        )

    df = frame.copy()
    true_values = df[df[group_col].astype(int) == 1]
    false_values = df[df[group_col].astype(int) == 0]

    rows: list[dict[str, Any]] = [
        {
            "Statistic": "N",
            "Special": int(len(true_values)),
            "Other": int(len(false_values)),
            "Diff": "",
            "t-stat": "",
            "p-value": "",
        }
    ]

    for display_name, column, _is_int in var_specs:
        left = to_numeric(true_values[column]).dropna()
        right = to_numeric(false_values[column]).dropna()
        if len(left) >= 2 and len(right) >= 2:
            t_stat, p_value = stats.ttest_ind(left, right, equal_var=False, nan_policy="omit")
        else:
            t_stat, p_value = np.nan, np.nan

        rows.append(
            {
                "Statistic": display_name,
                "Special": "",
                "Other": "",
                "Diff": "",
                "t-stat": "",
                "p-value": "",
            }
        )
        rows.append(
            {
                "Statistic": "  Mean",
                "Special": _format_output(left.mean(), digits=digits),
                "Other": _format_output(right.mean(), digits=digits),
                "Diff": _with_star(left.mean() - right.mean(), p_value, digits=digits),
                "t-stat": _format_output(t_stat, digits=digits),
                "p-value": _format_output(p_value, digits=digits),
            }
        )
        rows.append(
            {
                "Statistic": "  Std. Dev.",
                "Special": _format_output(left.std(), digits=digits),
                "Other": _format_output(right.std(), digits=digits),
                "Diff": "",
                "t-stat": "",
                "p-value": "",
            }
        )
        for percentile in percentiles:
            label = _percentile_label(percentile)
            rows.append(
                {
                    "Statistic": f"  {label}",
                    "Special": _format_output(left.quantile(percentile), digits=digits),
                    "Other": _format_output(right.quantile(percentile), digits=digits),
                    "Diff": "",
                    "t-stat": "",
                    "p-value": "",
                }
            )
        if use_mannwhitney and len(left) and len(right):
            _, mw_p = stats.mannwhitneyu(left, right, alternative="two-sided")
            rows.append(
                {
                    "Statistic": "  Mann-Whitney p-value",
                    "Special": "",
                    "Other": "",
                    "Diff": _format_output(mw_p, digits=digits),
                    "t-stat": "",
                    "p-value": "",
                }
            )

    return pd.DataFrame(rows)


def build_event_study_frame(
    company_year_abc: pd.DataFrame,
    *,
    ucc_col: str = PATENT_UCC_COL,
    year_col: str = PATENT_YEAR_COL,
    outcome_col: str = "mean_quality",
    window: int = 5,
) -> pd.DataFrame:
    tmp = company_year_abc[company_year_abc["ever_special"] == 1].copy()
    tmp = tmp.dropna(subset=[year_col, "is_special_year", outcome_col])
    t0 = (
        tmp[tmp["is_special_year"] == 1]
        .groupby(ucc_col)[year_col]
        .min()
        .rename("t0")
    )
    tmp = tmp.join(t0, on=ucc_col)
    tmp = tmp.dropna(subset=["t0"])
    tmp["event_time"] = tmp[year_col] - tmp["t0"]
    tmp = tmp[(tmp["event_time"] >= -window) & (tmp["event_time"] <= window)].copy()
    return (
        tmp.groupby("event_time", sort=True)[outcome_col]
        .mean()
        .reset_index()
    )


def _percentile_label(percentile: float) -> str:
    if percentile == 0.50:
        return "Median"
    return f"P{int(round(percentile * 100))}"


def _significance_star(p_value: Any) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def _format_output(value: Any, *, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def _with_star(value: Any, p_value: Any, *, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}{_significance_star(p_value)}"
