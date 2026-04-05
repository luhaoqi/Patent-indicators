from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.run_regressions import run_regressions


def _build_innovation_frame() -> pd.DataFrame:
    rows = []
    patent_counts = {
        "000001": [14, 13, 12, 13, 14],
        "000002": [12, 12, 11, 11, 10],
        "000003": [11, 10, 10, 8, 7],
        "000004": [10, 10, 9, 8, 6],
        "000005": [10, 9, 7, 5, 3],
    }
    interaction_noise = {
        "000001": [0.02, -0.03, 0.01, -0.02, 0.03],
        "000002": [-0.01, 0.02, -0.02, 0.03, -0.01],
        "000003": [0.03, 0.00, -0.01, 0.02, -0.02],
        "000004": [-0.02, 0.01, 0.03, -0.01, 0.00],
        "000005": [0.01, -0.02, 0.02, 0.00, -0.03],
    }
    years = list(range(2019, 2024))
    for firm_idx, (stkcd, counts) in enumerate(patent_counts.items(), start=1):
        for year_idx, (year, patent_count) in enumerate(zip(years, counts, strict=True)):
            noise = interaction_noise[stkcd][year_idx]
            mean_z = -0.3 + 0.12 * firm_idx + 0.04 * year_idx + noise
            highq_share = min(0.18 + 0.04 * firm_idx + 0.015 * year_idx + 0.5 * noise, 0.85)
            highq_count = max(1, int(round(patent_count * highq_share)))
            mean_raw_q_w = 1.2 + 0.25 * firm_idx + 0.08 * year_idx + 0.4 * noise
            rows.append(
                {
                    "Stkid": stkcd,
                    "ShortName": f"Firm{firm_idx}",
                    "year": year,
                    "PatentCount": patent_count,
                    "mean_z_q_ft": mean_z,
                    "highq_share_ft": highq_share,
                    "highq_count_ft": highq_count,
                    "log_highq_count_ft": float(np.log1p(highq_count)),
                    "mean_raw_q_w_ft": mean_raw_q_w,
                    "mean_raw_q_ft": mean_raw_q_w,
                    "log_patent_count_ft": float(np.log1p(patent_count)),
                    "Innovation_raw": mean_raw_q_w,
                    "Innovation_z": mean_z,
                }
            )
    return pd.DataFrame(rows)


def _build_financial_frame(innovation_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in innovation_df.to_dict(orient="records"):
        stkcd = record["Stkid"]
        year = int(record["year"])
        firm_idx = int(stkcd[-1])
        year_idx = year - 2019
        patent_count = float(record["PatentCount"])
        mean_z = float(record["mean_z_q_ft"])
        highq_share = float(record["highq_share_ft"])
        highq_count = float(record["highq_count_ft"])

        interaction = ((firm_idx * (year_idx + 2)) % 4 - 1.5) * 0.6
        asset = 140.0 + 18.0 * firm_idx + 7.0 * year_idx + interaction
        liability = 55.0 + 9.0 * firm_idx + 3.0 * year_idx + 0.4 * interaction
        sales = 320.0 + 28.0 * firm_idx + 20.0 * year_idx + 2.5 * patent_count + 1.8 * interaction
        profit = sales * (0.085 + 0.008 * mean_z + 0.015 * highq_share + 0.0008 * interaction)
        ebit = sales * (0.11 + 0.010 * mean_z + 0.012 * highq_share + 0.0010 * interaction)
        ebitda = sales * (0.14 + 0.012 * mean_z + 0.012 * highq_share + 0.0010 * interaction)
        roa = 0.030 + 0.005 * firm_idx + 0.003 * year_idx + 0.010 * mean_z + 0.008 * highq_share + 0.001 * interaction
        roe = 0.060 + 0.007 * firm_idx + 0.004 * year_idx + 0.015 * mean_z + 0.010 * highq_share + 0.001 * interaction

        rows.append(
            {
                "stkcd": stkcd,
                "Accper": f"{year}-12-31",
                "roa": roa,
                "roe": roe,
                "tq": 1.1 + 0.02 * firm_idx,
                "asset": asset,
                "liability": liability,
                "finlev": liability / asset,
                "gassets": 0.05 + 0.010 * year_idx + 0.002 * firm_idx + 0.0006 * interaction,
                "gfa": 0.03 + 0.008 * year_idx + 0.001 * firm_idx + 0.0005 * interaction,
                "ebit": ebit,
                "ebitda": ebitda,
                "profit": profit,
                "sales": sales,
                "soe": 1 if firm_idx % 2 == 0 else 0,
                "研发费用": 10.0 + 1.7 * highq_count + 0.8 * firm_idx + 0.5 * year_idx + 0.3 * interaction,
            }
        )
    return pd.DataFrame(rows)


class Stage2RegressionTests(unittest.TestCase):
    def test_run_regressions_outputs_new_summary_tables(self) -> None:
        try:
            import linearmodels  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("linearmodels 未安装")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "experiments"
            data_dir = output_root / "exp_reg" / "stage2" / "data"
            data_dir.mkdir(parents=True)

            innovation_df = _build_innovation_frame()
            financial_df = _build_financial_frame(innovation_df)

            innovation_path = data_dir / "firm_year_innovation.parquet"
            financial_path = root / "financial_annual_clean.parquet"
            innovation_df.to_parquet(innovation_path, index=False)
            financial_df.to_parquet(financial_path, index=False)

            summary = run_regressions(
                experiment_id="exp_reg",
                output_root=str(output_root),
                firm_year_innovation_path=innovation_path,
                financial_panel_path=financial_path,
                year_min=2019,
                year_max=2023,
                sample_thresholds=(10, 5, 1),
                winsor_lower=0.01,
                winsor_upper=0.99,
                rd_year_min=2019,
                rd_year_max=2023,
                future_horizons=(1, 2),
            )

            self.assertEqual(int(summary["schema_version"]), 4)
            self.assertTrue(Path(root / summary["regression_panel_path"]).exists())
            self.assertTrue(all((root / relative_path).exists() for relative_path in summary["sample_summary_outputs"]))

            regression_panel = pd.read_parquet(root / summary["regression_panel_path"])
            self.assertTrue(
                {
                    "profit_margin",
                    "rd_intensity_asset",
                    "sales_growth",
                    "roa_w",
                    "roa_w_lead1",
                    "log_sales_lead2",
                    "ln_asset_lead1",
                }.issubset(set(regression_panel.columns))
            )

            summary_csv = root / summary["table_outputs"][0]
            sample_csv = root / summary["sample_summary_outputs"][0]
            metadata_path = output_root / "exp_reg" / "stage2" / "metadata" / "run_regressions.json"

            summary_df = pd.read_csv(summary_csv)
            sample_df = pd.read_csv(sample_csv)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertTrue({"spec_id", "dep_var", "key_regressor", "sample_rule", "status"}.issubset(set(summary_df.columns)))
            self.assertIn("future_horizon", set(summary_df.columns))
            self.assertTrue({"threshold_rows", "final_reg_nobs", "dropped_by_controls_missing"}.issubset(set(sample_df.columns)))
            self.assertIn("PatentCount >= 10", set(sample_df["sample_rule"]))
            self.assertIn("PatentCount >= 5", set(sample_df["sample_rule"]))
            self.assertIn("PatentCount >= 1", set(sample_df["sample_rule"]))
            self.assertTrue((summary_df["status"] == "success").any())
            self.assertTrue(summary_df["spec_id"].astype("string").str.startswith("roa_mean_z_pc10").any())
            self.assertTrue((summary_df["spec_id"] == "logsales_mean_z_pc10_rdsame").any())
            self.assertTrue((summary_df["spec_id"] == "logsales_mean_z_pc5_rdsame").any())
            self.assertTrue((summary_df["spec_id"] == "logsales_mean_z_pc10_rdhorse").any())
            self.assertTrue((summary_df["spec_id"] == "logsales_mean_z_pc1_rdhorse").any())
            self.assertTrue((summary_df["spec_id"] == "logsales_rd_asset_pc10_rdonly").any())
            self.assertTrue((summary_df["spec_id"] == "logasset_rd_asset_pc1_rdonly").any())
            self.assertTrue((summary_df["spec_id"] == "logasset_mean_z_pc10_cnt1").any())
            self.assertTrue((summary_df["spec_id"] == "logasset_mean_z_pc10_rdsame").any())
            self.assertTrue((summary_df["spec_id"] == "logasset_mean_z_pc10_h1_cnt1").any())
            self.assertTrue((summary_df["spec_id"] == "roa_highq_share_pc5_h2_cnt1").any())
            logasset_row = summary_df.loc[summary_df["spec_id"] == "logasset_mean_z_pc10_cnt1"].iloc[0]
            self.assertNotIn("ln_asset +", str(logasset_row["formula"]))
            future_logasset_row = summary_df.loc[summary_df["spec_id"] == "logasset_mean_z_pc10_h1_cnt1"].iloc[0]
            self.assertIn(" + ln_asset + ", str(future_logasset_row["formula"]))
            self.assertTrue(str(logasset_row["output_txt"]).endswith("regressions/current/logasset_mean_z/reg_logasset_mean_z_pc10_cnt1.txt"))
            self.assertTrue(str(future_logasset_row["output_txt"]).endswith("regressions/future/logasset_mean_z/reg_logasset_mean_z_pc10_h1_cnt1.txt"))
            self.assertEqual(int(metadata["schema_version"]), 4)
            self.assertEqual(metadata["future_horizons"], [1, 2])
            self.assertIn("tbl_regression_sample_summary.csv", metadata["sample_summary_outputs"][0])


if __name__ == "__main__":
    unittest.main()
