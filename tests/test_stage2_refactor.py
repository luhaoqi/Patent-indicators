from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis.build_firm_year_innovation import build_firm_year_innovation
from analysis.build_main_enriched import build_experiment_patent_panel, build_patent_master
from analysis.build_ucc_panel import build_ucc_mapping
from analysis.shared_prep import build_financial_annual_panel, build_special_firm_labels, verify_shared_prep


class Stage2RefactorTests(unittest.TestCase):
    def test_patent_master_and_experiment_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            shared_root = root / "shared"
            output_root = root / "experiments"
            stage1_dir = output_root / "exp_a" / "stage1"
            stage1_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {"申请号": "A1", "申请年份": 2020, "统一社会信用代码": "U1", "被引证次数": 3, "专利类型": "发明申请"},
                    {"申请号": "A2", "申请年份": 2021, "统一社会信用代码": "U2", "被引证次数": 2, "专利类型": "发明申请"},
                ]
            ).to_csv(raw_dir / "part1.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"申请号": "A1", "申请年份": 2020, "统一社会信用代码": "U1", "被引证次数": 7, "专利类型": "发明授权"},
                ]
            ).to_csv(raw_dir / "part2.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"申请号": "A1", "BS": 0.3, "FS": 0.1, "Quality_q": 1.2},
                    {"申请号": "A2", "BS": 0.4, "FS": 0.2, "Quality_q": 2.5},
                ]
            ).to_csv(stage1_dir / "patent_quality_output.csv", index=False, encoding="utf-8-sig")

            patent_master = build_patent_master(
                raw_patent_dir=raw_dir,
                shared_root=str(shared_root),
                chunksize=1,
            )
            master_df = pd.read_parquet(patent_master["patent_master_path"])
            self.assertEqual(len(master_df), 2)
            self.assertEqual(int(master_df.loc[master_df["申请号"] == "A1", "被引证次数"].iloc[0]), 7)

            result = build_experiment_patent_panel(
                experiment_id="exp_a",
                stage1_output_path=stage1_dir / "patent_quality_output.csv",
                output_root=str(output_root),
                patent_master_path=patent_master["patent_master_path"],
            )
            panel_df = pd.read_parquet(result["experiment_patent_panel_path"])
            self.assertEqual(len(panel_df), 2)
            self.assertIn("申请年份", panel_df.columns)
            self.assertIn("统一社会信用代码", panel_df.columns)
            self.assertEqual(int(panel_df.loc[panel_df["申请号"] == "A1", "被引证次数"].iloc[0]), 7)

    def test_shared_prep_outputs_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shared_root = root / "shared"

            special_df = pd.DataFrame(
                [
                    {"统一社会信用代码": "U1", "年份": 2020, "科创企业称号总数": 1},
                    {"统一社会信用代码": "U2", "年份": 2021, "科创企业称号总数": 0},
                    {"统一社会信用代码": "U1", "年份": 2021, "科创企业称号总数": 1},
                ]
            )
            special_path = root / "special.dta"
            special_path.write_text("placeholder", encoding="utf-8")

            financial_df = pd.DataFrame(
                [
                    {"stkcd": "1", "Accper": "2020-12-31", "roa": 1.0, "roe": 2.0, "tq": 1.1, "asset": 10.0, "liability": 5.0, "finlev": 0.5, "gassets": 0.2, "soe": 1},
                    {"stkcd": "1", "Accper": "2020-06-30", "roa": 9.0, "roe": 9.0, "tq": 9.0, "asset": 99.0, "liability": 50.0, "finlev": 0.9, "gassets": 0.9, "soe": 0},
                    {"stkcd": "2", "Accper": "2021-12-31", "roa": 1.5, "roe": 2.5, "tq": 1.2, "asset": 20.0, "liability": 8.0, "finlev": 0.4, "gassets": 0.3, "soe": 0},
                ]
            )
            financial_path = root / "financial.dta"
            financial_df.to_stata(financial_path, write_index=False)

            parent_df = pd.DataFrame(
                [
                    {"stkid": "1", "shortname": "FirmA", "SocialCreditCode": "U1", "FirstYear": 2020, "LastYear": 2021},
                    {"stkid": "2", "shortname": "FirmB", "SocialCreditCode": "U2", "FirstYear": 2021, "LastYear": 2021},
                ]
            )
            parent_path = root / "parent.csv"
            parent_df.to_csv(parent_path, index=False, encoding="utf-8-sig")

            mapping_df = pd.DataFrame(
                [
                    {"企业名称": "子公司甲", "统一社会信用代码": "U1C"},
                    {"企业名称": "子公司乙", "统一社会信用代码": "U2C"},
                ]
            )
            mapping_path = root / "mapping.csv"
            mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")

            subjoint_df = pd.DataFrame(
                [
                    {"Symbol": "1", "EndDate": "2020-12-31", "RalatedParty": "子公司甲", "Relationship": "子公司"},
                    {"Symbol": "2", "EndDate": "2021-12-31", "RalatedParty": "子公司乙", "Relationship": "子公司"},
                ]
            )
            subjoint_path = root / "subjoint.csv"
            subjoint_df.to_csv(subjoint_path, index=False, encoding="utf-8-sig")

            with patch("analysis.shared_prep.pd.read_stata", return_value=special_df):
                special_outputs = build_special_firm_labels(
                    special_list_path=special_path,
                    shared_root=str(shared_root),
                )
            financial_outputs = build_financial_annual_panel(
                financial_data_path=financial_path,
                shared_root=str(shared_root),
                year_min=2020,
                year_max=2021,
            )
            ucc_outputs = build_ucc_mapping(
                parent_csv_path=parent_path,
                subsidiary_mapping_path=mapping_path,
                subjoint_csv_path=subjoint_path,
                shared_root=str(shared_root),
                chunksize=1,
            )

            special_labels_df = pd.read_parquet(special_outputs["firm_year_special_labels_path"])
            financial_panel_df = pd.read_parquet(financial_outputs["financial_annual_clean_path"])
            ucc_exploded_df = pd.read_parquet(ucc_outputs["ucc_exploded_path"])

            self.assertEqual(len(special_labels_df), 3)
            self.assertEqual(len(financial_panel_df), 2)
            self.assertTrue((financial_panel_df["Accper"].astype("string").str.contains("12-31")).all())
            self.assertIn("U1C", set(ucc_exploded_df["UCC"].tolist()))

            verify_summary = verify_shared_prep(shared_root=str(shared_root))
            self.assertTrue(verify_summary["checks"]["firm_year_special_labels"]["primary_key_unique"])
            self.assertTrue(verify_summary["checks"]["ucc_exploded"]["primary_key_unique"])
            self.assertTrue(verify_summary["checks"]["financial_annual_clean"]["primary_key_unique"])

    def test_build_firm_year_innovation_from_shared_ucc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shared_root = root / "shared"
            output_root = root / "experiments"
            stage1_dir = output_root / "exp_b" / "stage1"
            stage1_dir.mkdir(parents=True)

            patent_master_dir = shared_root / "patent_master"
            ucc_mapping_dir = shared_root / "ucc_mapping"
            patent_master_dir.mkdir(parents=True)
            ucc_mapping_dir.mkdir(parents=True)

            patent_master_df = pd.DataFrame(
                [
                    {"申请号": "P1", "申请年份": 2020, "统一社会信用代码": "U1"},
                    {"申请号": "P2", "申请年份": 2020, "统一社会信用代码": "U1"},
                    {"申请号": "P3", "申请年份": 2021, "统一社会信用代码": "U2"},
                ]
            )
            patent_master_path = patent_master_dir / "patent_master.parquet"
            patent_master_df.to_parquet(patent_master_path, index=False)

            pd.DataFrame(
                [
                    {"申请号": "P1", "BS": 0.2, "FS": 0.1, "Quality_q": 1.0},
                    {"申请号": "P2", "BS": 0.3, "FS": 0.2, "Quality_q": 3.0},
                    {"申请号": "P3", "BS": 0.4, "FS": 0.3, "Quality_q": 2.0},
                ]
            ).to_csv(stage1_dir / "patent_quality_output.csv", index=False, encoding="utf-8-sig")

            panel_result = build_experiment_patent_panel(
                experiment_id="exp_b",
                stage1_output_path=stage1_dir / "patent_quality_output.csv",
                output_root=str(output_root),
                patent_master_path=patent_master_path,
            )

            pd.DataFrame(
                [
                    {"Stkid": "000001", "ShortName": "FirmA", "year": 2020, "UCC": "U1"},
                    {"Stkid": "000002", "ShortName": "FirmB", "year": 2021, "UCC": "U2"},
                ]
            ).to_parquet(ucc_mapping_dir / "ucc_exploded.parquet", index=False)

            innovation_path = build_firm_year_innovation(
                experiment_id="exp_b",
                output_root=str(output_root),
                experiment_patent_panel_path=panel_result["experiment_patent_panel_path"],
                ucc_exploded_path=ucc_mapping_dir / "ucc_exploded.parquet",
                shared_root=str(shared_root),
                top_k=2,
                quality_cap=10.0,
            )
            innovation_df = pd.read_parquet(innovation_path)
            self.assertEqual(len(innovation_df), 2)
            self.assertAlmostEqual(
                float(innovation_df.loc[innovation_df["Stkid"] == "000001", "Innovation_raw"].iloc[0]),
                2.0,
            )


if __name__ == "__main__":
    unittest.main()
