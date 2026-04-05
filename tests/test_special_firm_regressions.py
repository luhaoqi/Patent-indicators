from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.analyze_special_firms import analyze_special_firms
from analysis.special_firm_regressions import prepare_special_regression_patent_frame


PUBLIC_YEAR_COL = "公开公告年份"


def _build_special_regression_test_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quality_map = {
        ("U1", 2008): [6.0, 5.0],
        ("U1", 2009): [8.0, 7.0],
        ("U1", 2010): [6.5, 6.0],
        ("U2", 2008): [4.0, 4.0],
        ("U2", 2009): [4.5, 4.0],
        ("U2", 2010): [8.0, 7.5],
        ("U3", 2008): [1.0, 1.0],
        ("U3", 2009): [1.2, 1.0],
        ("U3", 2010): [1.1, 1.0],
        ("U4", 2008): [2.0, 1.5],
        ("U4", 2009): [2.2, 2.0],
        ("U4", 2010): [2.1, 2.0],
    }

    counter = 1
    for (firm_id, year), quality_values in quality_map.items():
        for patent_idx, quality in enumerate(quality_values, start=1):
            applicant = f"企业{firm_id}"
            if firm_id == "U4" and year == 2010 and patent_idx == 1:
                applicant = "示范高校; 企业U4"
            rows.append(
                {
                    "申请号": f"P{counter:03d}",
                    "申请年份": year,
                    PUBLIC_YEAR_COL: year,
                    "统一社会信用代码": firm_id,
                    "Quality_q": quality,
                    "BS": 0.2,
                    "申请人": applicant,
                }
            )
            counter += 1
    return pd.DataFrame(rows)


class SpecialFirmRegressionTests(unittest.TestCase):
    def test_prepare_special_regression_patent_frame_builds_year_metrics(self) -> None:
        patent_df = pd.DataFrame(
            [
                {"申请号": "P2", "申请年份": 2020, "统一社会信用代码": "U1", "Quality_q": 10.0},
                {"申请号": "P1", "申请年份": 2020, "统一社会信用代码": "U1", "Quality_q": 10.0},
                {"申请号": "P3", "申请年份": 2020, "统一社会信用代码": "U2", "Quality_q": 5.0},
                {"申请号": "P4", "申请年份": 2021, "统一社会信用代码": "U1", "Quality_q": 7.0},
                {"申请号": "P5", "申请年份": 2021, "统一社会信用代码": "U2", "Quality_q": 7.0},
            ]
        )
        special_labels = pd.DataFrame(
            [
                {"统一社会信用代码": "U1", "申请年份": 2020, "is_special_year": 1},
            ]
        )

        frame = prepare_special_regression_patent_frame(
            patent_df,
            year_col="申请年份",
            special_uccs=["U1"],
            firm_year_special=special_labels,
            topk_share=0.5,
        )

        year_2020 = frame[frame["year"] == 2020].sort_values("rank_q_pft").reset_index(drop=True)
        self.assertEqual(year_2020["patent_id"].tolist(), ["P1", "P2", "P3"])
        self.assertEqual(int(year_2020.loc[0, "topk_q_pft"]), 1)
        self.assertEqual(int(year_2020.loc[1, "topk_q_pft"]), 0)
        self.assertEqual(int(year_2020.loc[0, "ever_special_f"]), 1)
        self.assertEqual(int(year_2020.loc[0, "special_year_ft"]), 1)
        self.assertAlmostEqual(float(year_2020.loc[2, "rank_q_pft"]), 1.0)

        year_2021 = frame[frame["year"] == 2021].reset_index(drop=True)
        self.assertTrue((year_2021["z_q_pft"] == 0.0).all())
        self.assertTrue((year_2021["special_year_ft"] == 0).all())

    def test_analyze_special_firms_outputs_regressions_for_both_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "experiments"
            experiment_id = "exp_special_reg"
            stage2_data_dir = output_root / experiment_id / "stage2_exact" / "data"
            stage2_data_dir.mkdir(parents=True)

            patent_df = _build_special_regression_test_frame()
            patent_panel_path = stage2_data_dir / "experiment_patent_panel.parquet"
            patent_df.to_parquet(patent_panel_path, index=False)

            special_labels = pd.DataFrame(
                [
                    {"统一社会信用代码": "U1", "申请年份": 2009, "is_special_year": 1},
                    {"统一社会信用代码": "U2", "申请年份": 2010, "is_special_year": 1},
                ]
            )
            special_labels_path = root / "firm_year_special_labels.parquet"
            special_labels.to_parquet(special_labels_path, index=False)

            special_ucc_path = root / "special_ucc_set.parquet"
            pd.DataFrame({"统一社会信用代码": ["U1", "U2"]}).to_parquet(special_ucc_path, index=False)

            terms_path = root / "filter_terms.txt"
            terms_path.write_text("高校\n", encoding="utf-8")

            summary = analyze_special_firms(
                experiment_id=experiment_id,
                output_root=str(output_root),
                experiment_patent_panel_path=patent_panel_path,
                firm_year_special_labels_path=special_labels_path,
                special_ucc_set_path=special_ucc_path,
                unit_filter_terms_path=terms_path,
                exclude_years=(),
                quality_min=0.0,
                bs_min=0.0,
                quality_threshold=1.0,
                regression_topk_share=0.5,
                policy_start_year=2008,
                event_window=2,
                exact_date=True,
            )

            baseline_root = output_root / experiment_id / "stage2_exact" / "tables" / "特殊企业对比" / "回归分析"
            filtered_root = output_root / experiment_id / "stage2_exact" / "tables" / "特殊企业_过滤部分单位" / "回归分析"

            self.assertTrue((baseline_root / "静态横截面" / "cluster_firm" / "reg_S2.csv").exists())
            self.assertTrue((baseline_root / "ABC分解" / "cluster_firm" / "reg_G3.csv").exists())
            self.assertTrue((baseline_root / "动态企业内" / "no_cluster" / "reg_D7.csv").exists())
            self.assertTrue((filtered_root / "静态横截面" / "cluster_firm" / "reg_S2.csv").exists())
            self.assertTrue((filtered_root / "动态企业内" / "cluster_firm" / "tbl_regression_summary.csv").exists())

            overall_summary = pd.read_csv(baseline_root / "tbl_regression_summary.csv")
            self.assertTrue({"S2", "G3", "D7"}.issubset(set(overall_summary["regression_id"])))
            self.assertTrue({"status", "warning_flag", "warning_message"}.issubset(set(overall_summary.columns)))
            self.assertTrue(overall_summary["status"].isin(["ok", "warning"]).any())

            g3_rows = pd.read_csv(baseline_root / "ABC分解" / "cluster_firm" / "reg_G3.csv")
            self.assertEqual(set(g3_rows["effect_label"]), {"A-B", "A-C", "B-C"})
            self.assertTrue({"warning_flag", "warning_message"}.issubset(set(g3_rows.columns)))

            filtered_variant = summary["variants"]["filtered_units"]
            self.assertIn("regression_summary", filtered_variant)
            self.assertGreater(filtered_variant["regression_summary"]["panel_rows"]["firm_year_dynamic_within"], 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
