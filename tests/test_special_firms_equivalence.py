from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from analysis.common.analysis import (
    PATENT_UCC_COL,
    PATENT_YEAR_COL,
    QUALITY_COL,
    INVALID_UCC_VALUES,
    build_company_special_panel_from_ucc_set,
    build_company_year_abc_panel,
    build_company_year_special_panel,
    normalize_string_series,
    to_numeric,
)


def _prepare_valid_ucc_patents_reference(
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


def _build_company_special_panel_reference(
    patent_df: pd.DataFrame,
    special_uccs: list[str],
    *,
    quality_threshold: float,
) -> pd.DataFrame:
    df = _prepare_valid_ucc_patents_reference(patent_df)
    special_ucc_set = {value for value in special_uccs if value and value not in INVALID_UCC_VALUES}
    company_agg = (
        df.groupby(PATENT_UCC_COL, dropna=False)
        .agg(
            total_patents=(QUALITY_COL, "size"),
            high_q_count=(QUALITY_COL, lambda series: int((to_numeric(series).fillna(-np.inf) >= quality_threshold).sum())),
            mean_quality=(QUALITY_COL, "mean"),
            max_quality=(QUALITY_COL, "max"),
        )
        .reset_index()
    )
    company_agg["log_total_patents"] = np.log1p(company_agg["total_patents"])
    company_agg["is_special"] = company_agg[PATENT_UCC_COL].isin(special_ucc_set).astype(int)
    return company_agg


def _build_company_year_special_panel_reference(
    p_dyn: pd.DataFrame,
    *,
    quality_threshold: float,
) -> pd.DataFrame:
    df = _prepare_valid_ucc_patents_reference(p_dyn)
    df["is_special_year"] = to_numeric(df["is_special_year"]).fillna(0).astype(int)
    agg = (
        df.groupby([PATENT_UCC_COL, PATENT_YEAR_COL], dropna=False)
        .agg(
            total_patents=(QUALITY_COL, "size"),
            high_q_count=(QUALITY_COL, lambda series: int((to_numeric(series).fillna(-np.inf) >= quality_threshold).sum())),
            mean_quality=(QUALITY_COL, "mean"),
            max_quality=(QUALITY_COL, "max"),
            is_special_year=("is_special_year", "max"),
        )
        .reset_index()
    )
    agg["log_total_patents"] = np.log1p(agg["total_patents"])
    return agg


def _build_company_year_abc_panel_reference(
    p_dyn: pd.DataFrame,
    *,
    quality_threshold: float,
) -> pd.DataFrame:
    df = _prepare_valid_ucc_patents_reference(p_dyn)
    df["is_special_year"] = to_numeric(df["is_special_year"]).fillna(0).astype("int8")
    ever_special = df.groupby(PATENT_UCC_COL, sort=False)["is_special_year"].max().rename("ever_special").astype("int8")
    df = df.join(ever_special, on=PATENT_UCC_COL)
    df["firm_group_3"] = np.select(
        [
            (df["ever_special"] == 1) & (df["is_special_year"] == 1),
            (df["ever_special"] == 1) & (df["is_special_year"] == 0),
            (df["ever_special"] == 0),
        ],
        ["A_treated_year", "B_same_firm_other_year", "C_never_treated"],
        default="C_never_treated",
    )
    df["_high_q"] = (df[QUALITY_COL] >= quality_threshold).astype("int8")
    agg = (
        df.groupby([PATENT_UCC_COL, PATENT_YEAR_COL], sort=False, observed=True)
        .agg(
            total_patents=(QUALITY_COL, "size"),
            high_q_count=("_high_q", "sum"),
            mean_quality=(QUALITY_COL, "mean"),
            max_quality=(QUALITY_COL, "max"),
            is_special_year=("is_special_year", "max"),
            ever_special=("ever_special", "max"),
            firm_group_3=("firm_group_3", "first"),
        )
        .reset_index()
    )
    agg["log_total_patents"] = np.log1p(agg["total_patents"])
    return agg


def _sort_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.sort_values(columns, kind="mergesort").reset_index(drop=True)


class SpecialFirmsEquivalenceTests(unittest.TestCase):
    def test_polars_aggregations_match_reference(self) -> None:
        patent_df = pd.DataFrame(
            [
                {PATENT_UCC_COL: " U1 ", PATENT_YEAR_COL: 2008, QUALITY_COL: 1.0, "is_special_year": 1},
                {PATENT_UCC_COL: "U1", PATENT_YEAR_COL: 2008, QUALITY_COL: 2.5, "is_special_year": 1},
                {PATENT_UCC_COL: "U1", PATENT_YEAR_COL: 2009, QUALITY_COL: 0.4, "is_special_year": 0},
                {PATENT_UCC_COL: "U2", PATENT_YEAR_COL: 2008, QUALITY_COL: 3.0, "is_special_year": 0},
                {PATENT_UCC_COL: "U2", PATENT_YEAR_COL: 2009, QUALITY_COL: 4.0, "is_special_year": 1},
                {PATENT_UCC_COL: "U3", PATENT_YEAR_COL: 2009, QUALITY_COL: 0.9, "is_special_year": 0},
                {PATENT_UCC_COL: "nan", PATENT_YEAR_COL: 2009, QUALITY_COL: 9.0, "is_special_year": 1},
                {PATENT_UCC_COL: "-", PATENT_YEAR_COL: 2009, QUALITY_COL: 9.0, "is_special_year": 1},
                {PATENT_UCC_COL: "U4", PATENT_YEAR_COL: None, QUALITY_COL: 2.0, "is_special_year": 0},
                {PATENT_UCC_COL: "U4", PATENT_YEAR_COL: 2010, QUALITY_COL: None, "is_special_year": 0},
                {PATENT_UCC_COL: "U5", PATENT_YEAR_COL: 2010, QUALITY_COL: 1.2, "is_special_year": None},
            ]
        )
        special_uccs = ["U1", "U9", ""]
        quality_threshold = 1.0

        expected_company = _sort_frame(
            _build_company_special_panel_reference(patent_df, special_uccs, quality_threshold=quality_threshold),
            [PATENT_UCC_COL],
        )
        actual_company = _sort_frame(
            build_company_special_panel_from_ucc_set(patent_df, special_uccs, quality_threshold=quality_threshold),
            [PATENT_UCC_COL],
        )
        assert_frame_equal(expected_company, actual_company, check_dtype=False, check_like=False)

        expected_company_year = _sort_frame(
            _build_company_year_special_panel_reference(patent_df, quality_threshold=quality_threshold),
            [PATENT_UCC_COL, PATENT_YEAR_COL],
        )
        actual_company_year = _sort_frame(
            build_company_year_special_panel(patent_df, quality_threshold=quality_threshold),
            [PATENT_UCC_COL, PATENT_YEAR_COL],
        )
        assert_frame_equal(expected_company_year, actual_company_year, check_dtype=False, check_like=False)

        expected_abc = _sort_frame(
            _build_company_year_abc_panel_reference(patent_df, quality_threshold=quality_threshold),
            [PATENT_UCC_COL, PATENT_YEAR_COL],
        )
        actual_abc = _sort_frame(
            build_company_year_abc_panel(patent_df, quality_threshold=quality_threshold),
            [PATENT_UCC_COL, PATENT_YEAR_COL],
        )
        assert_frame_equal(expected_abc, actual_abc, check_dtype=False, check_like=False)


if __name__ == "__main__":
    unittest.main()
