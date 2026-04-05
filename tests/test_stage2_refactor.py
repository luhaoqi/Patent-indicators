from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis.build_firm_year_innovation import build_firm_year_innovation
from analysis.build_main_enriched import build_experiment_patent_panel, build_patent_master
from analysis.build_raw_patent_authorized_parts import build_raw_patent_authorized_parts
from analysis.build_ucc_panel import build_ucc_mapping
from analysis.export_top_patents_by_year import export_top_patents_by_year
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
            self.assertTrue(
                {
                    "mean_z_q_ft",
                    "highq_share_ft",
                    "highq_count_ft",
                    "log_highq_count_ft",
                    "mean_raw_q_w_ft",
                    "log_patent_count_ft",
                }.issubset(set(innovation_df.columns))
            )
            self.assertEqual(int(innovation_df.loc[innovation_df["Stkid"] == "000001", "PatentCount"].iloc[0]), 2)
            self.assertAlmostEqual(
                float(innovation_df.loc[innovation_df["Stkid"] == "000001", "highq_share_ft"].iloc[0]),
                0.5,
            )

    def test_export_top_patents_by_year_with_company_name_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "experiments"
            shared_root = root / "shared"
            stage2_data_dir = output_root / "exp_top" / "stage2" / "data"
            ucc_mapping_dir = shared_root / "ucc_mapping"
            raw_dir = root / "raw"
            stage2_data_dir.mkdir(parents=True)
            ucc_mapping_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)

            panel_path = stage2_data_dir / "experiment_patent_panel.parquet"
            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "申请年份": 2020,
                        "公开公告年份": 2020,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P1",
                        "申请人": "申请人P1",
                        "统一社会信用代码": "U1",
                        "BS": 0.1,
                        "FS": 0.9,
                        "Quality_q": 9.0,
                        "被引证次数": 3,
                    },
                    {
                        "申请号": "P1_LOW",
                        "申请年份": 2020,
                        "公开公告年份": 2020,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P1_LOW",
                        "申请人": "申请人P1_LOW",
                        "统一社会信用代码": "U1",
                        "BS": 0.1,
                        "FS": 0.2,
                        "Quality_q": 2.0,
                        "被引证次数": 1,
                    },
                    {
                        "申请号": "P2",
                        "申请年份": 2021,
                        "公开公告年份": 2020,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P2",
                        "申请人": "申请人P2",
                        "统一社会信用代码": "U2",
                        "BS": 0.1,
                        "FS": 0.8,
                        "Quality_q": 8.0,
                        "被引证次数": 4,
                    },
                    {
                        "申请号": "P3",
                        "申请年份": 2022,
                        "公开公告年份": 2022,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P3",
                        "申请人": "回退申请人",
                        "统一社会信用代码": "U3",
                        "BS": 0.1,
                        "FS": 0.7,
                        "Quality_q": 7.0,
                        "被引证次数": 5,
                    },
                ]
            ).to_parquet(panel_path, index=False)

            ucc_path = ucc_mapping_dir / "ucc_exploded.parquet"
            pd.DataFrame(
                [
                    {"Stkid": "000001", "ShortName": "上市公司甲", "year": 2020, "UCC": "U1"},
                    {"Stkid": "000002", "ShortName": "上市公司乙", "year": 2020, "UCC": "U2"},
                ]
            ).to_parquet(ucc_path, index=False)

            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "专利类型": "发明授权",
                        "专利名称": "原始标题P1",
                        "摘要文本": "原始摘要P1",
                        "申请日": "2020-01-01",
                        "公开公告日": "2020-06-01",
                        "授权公告日": "2020-07-01",
                        "申请人": "原始申请人P1",
                    },
                    {
                        "申请号": "P1_LOW",
                        "专利类型": "发明授权",
                        "专利名称": "原始标题P1_LOW",
                        "摘要文本": "原始摘要P1_LOW",
                        "申请日": "2020-01-02",
                        "公开公告日": "2020-06-02",
                        "授权公告日": "2020-07-02",
                        "申请人": "原始申请人P1_LOW",
                    },
                    {
                        "申请号": "P2",
                        "专利类型": "发明授权",
                        "专利名称": "原始标题P2",
                        "摘要文本": "原始摘要P2",
                        "申请日": "2021-03-01",
                        "公开公告日": "2020-08-01",
                        "授权公告日": "2020-09-01",
                        "申请人": "原始申请人P2",
                    },
                ]
            ).to_csv(raw_dir / "中国专利数据库2020年.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "申请号": "P3",
                        "专利类型": "发明授权",
                        "专利名称": "原始标题P3",
                        "摘要文本": "原始摘要P3",
                        "申请日": "2022-05-01",
                        "公开公告日": "2022-10-01",
                        "授权公告日": "2022-11-01",
                        "申请人": "原始申请人P3",
                    },
                ]
            ).to_csv(raw_dir / "中国专利数据库2022年.csv", index=False, encoding="utf-8-sig")

            build_raw_patent_authorized_parts(
                raw_patent_dir=raw_dir,
                shared_root=str(shared_root),
                chunksize=2,
                overwrite=True,
            )

            summary = export_top_patents_by_year(
                experiment_id="exp_top",
                output_root=str(output_root),
                experiment_patent_panel_path=panel_path,
                ucc_exploded_path=ucc_path,
                shared_root=str(shared_root),
                top_n=1,
            )

            self.assertEqual(len(summary["output_paths"]), 3)
            self.assertEqual(summary["company_name_source_counts"]["上市公司UCC映射(当年)"], 1)
            self.assertEqual(summary["company_name_source_counts"]["上市公司UCC映射(历史)"], 1)
            self.assertEqual(summary["company_name_source_counts"]["专利申请人回退"], 1)
            self.assertEqual(summary["raw_lookup_stats"]["matched"], 3)

            year_2020 = pd.read_csv(output_root / "exp_top" / "stage2" / "tables" / "top_patents_by_year" / "top_patents_year=2020_top1.csv")
            year_2021 = pd.read_csv(output_root / "exp_top" / "stage2" / "tables" / "top_patents_by_year" / "top_patents_year=2021_top1.csv")
            year_2022 = pd.read_csv(output_root / "exp_top" / "stage2" / "tables" / "top_patents_by_year" / "top_patents_year=2022_top1.csv")

            self.assertEqual(year_2020.loc[0, "申请号"], "P1")
            self.assertEqual(year_2020.loc[0, "专利名称"], "原始标题P1")
            self.assertEqual(year_2020.loc[0, "摘要文本"], "原始摘要P1")
            self.assertEqual(year_2020.loc[0, "公司名称"], "上市公司甲")
            self.assertEqual(year_2020.loc[0, "公司名称来源"], "上市公司UCC映射(当年)")
            self.assertEqual(str(year_2020.loc[0, "证券ID"]).zfill(6), "000001")

            self.assertEqual(year_2021.loc[0, "申请号"], "P2")
            self.assertEqual(year_2021.loc[0, "专利名称"], "原始标题P2")
            self.assertEqual(year_2021.loc[0, "公司名称"], "上市公司乙")
            self.assertEqual(year_2021.loc[0, "公司名称来源"], "上市公司UCC映射(历史)")

            self.assertEqual(year_2022.loc[0, "申请号"], "P3")
            self.assertEqual(year_2022.loc[0, "专利名称"], "原始标题P3")
            self.assertEqual(year_2022.loc[0, "公司名称"], "原始申请人P3")
            self.assertEqual(year_2022.loc[0, "公司名称来源"], "专利申请人回退")

    def test_export_top_patents_by_year_can_skip_heavy_lookups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "experiments"
            shared_root = root / "shared"
            stage2_data_dir = output_root / "exp_fast" / "stage2" / "data"
            stage2_data_dir.mkdir(parents=True)

            panel_path = stage2_data_dir / "experiment_patent_panel.parquet"
            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "申请年份": 2020,
                        "公开公告年份": 2020,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P1",
                        "申请人": "申请人P1",
                        "统一社会信用代码": "U1",
                        "BS": 0.1,
                        "FS": 0.9,
                        "Quality_q": 9.0,
                        "被引证次数": 3,
                    },
                    {
                        "申请号": "P2",
                        "申请年份": 2021,
                        "公开公告年份": 2021,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P2",
                        "申请人": "申请人P2",
                        "统一社会信用代码": "",
                        "BS": 0.1,
                        "FS": 0.8,
                        "Quality_q": 8.0,
                        "被引证次数": 2,
                    },
                ]
            ).to_parquet(panel_path, index=False)

            summary = export_top_patents_by_year(
                experiment_id="exp_fast",
                output_root=str(output_root),
                experiment_patent_panel_path=panel_path,
                shared_root=str(shared_root),
                top_n=1,
                skip_company_lookup=True,
                skip_raw_lookup=True,
            )

            self.assertIsNone(summary["ucc_mapping_path"])
            self.assertTrue(summary["company_lookup_stats"]["skipped"])
            self.assertTrue(summary["raw_lookup_stats"]["skipped"])

            year_2020 = pd.read_csv(output_root / "exp_fast" / "stage2" / "tables" / "top_patents_by_year" / "top_patents_year=2020_top1.csv")
            year_2021 = pd.read_csv(output_root / "exp_fast" / "stage2" / "tables" / "top_patents_by_year" / "top_patents_year=2021_top1.csv")

            self.assertEqual(year_2020.loc[0, "专利名称"], "面板标题P1")
            self.assertTrue(pd.isna(year_2020.loc[0, "摘要文本"]) or year_2020.loc[0, "摘要文本"] == "")
            self.assertEqual(year_2020.loc[0, "公司名称"], "申请人P1")
            self.assertEqual(year_2020.loc[0, "公司名称来源"], "专利申请人回退")

            self.assertEqual(year_2021.loc[0, "公司名称"], "申请人P2")
            self.assertEqual(year_2021.loc[0, "公司名称来源"], "专利申请人回退")

    def test_build_raw_patent_authorized_parts_and_export_from_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            shared_root = root / "shared"
            output_root = root / "experiments"
            stage2_data_dir = output_root / "exp_parquet" / "stage2" / "data"
            ucc_mapping_dir = shared_root / "ucc_mapping"
            raw_dir.mkdir()
            stage2_data_dir.mkdir(parents=True)
            ucc_mapping_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "专利类型": "发明授权",
                        "专利名称": "原始标题P1",
                        "摘要文本": "原始摘要P1",
                        "申请日": "2020-01-01",
                        "公开公告日": "2020-06-01",
                        "授权公告日": "2020-07-01",
                        "申请人": "原始申请人P1",
                    },
                    {
                        "申请号": "P1_APP",
                        "专利类型": "发明申请",
                        "专利名称": "非授权P1",
                        "摘要文本": "非授权摘要P1",
                        "申请日": "2020-01-02",
                        "公开公告日": "2020-06-02",
                        "授权公告日": "",
                        "申请人": "非授权申请人P1",
                    },
                ]
            ).to_csv(raw_dir / "中国专利数据库2020年.csv", index=False, encoding="utf-8-sig")

            result = build_raw_patent_authorized_parts(
                raw_patent_dir=raw_dir,
                shared_root=str(shared_root),
                chunksize=1,
                overwrite=True,
            )
            parquet_path = result["output_dir"] / "中国专利数据库2020年.parquet"
            parquet_df = pd.read_parquet(parquet_path)
            self.assertEqual(list(parquet_df["申请号"]), ["P1"])

            panel_path = stage2_data_dir / "experiment_patent_panel.parquet"
            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "申请年份": 2020,
                        "公开公告年份": 2020,
                        "专利类型": "发明授权",
                        "专利名称": "面板标题P1",
                        "申请人": "面板申请人P1",
                        "统一社会信用代码": "",
                        "BS": 0.2,
                        "FS": 0.8,
                        "Quality_q": 9.0,
                        "被引证次数": 2,
                    }
                ]
            ).to_parquet(panel_path, index=False)

            summary = export_top_patents_by_year(
                experiment_id="exp_parquet",
                output_root=str(output_root),
                experiment_patent_panel_path=panel_path,
                shared_root=str(shared_root),
                top_n=1,
                skip_company_lookup=True,
            )

            self.assertEqual(summary["raw_lookup_source"], "shared_authorized_parquet_parts")
            year_2020 = pd.read_csv(output_root / "exp_parquet" / "stage2" / "tables" / "top_patents_by_year" / "top_patents_year=2020_top1.csv")
            self.assertEqual(year_2020.loc[0, "专利名称"], "原始标题P1")
            self.assertEqual(year_2020.loc[0, "摘要文本"], "原始摘要P1")
            self.assertEqual(year_2020.loc[0, "公司名称"], "原始申请人P1")
            self.assertEqual(year_2020.loc[0, "公司名称来源"], "专利申请人回退")

    def test_build_raw_patent_authorized_parts_skips_bad_csv_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            shared_root = root / "shared"
            raw_dir.mkdir()

            raw_csv = raw_dir / "中国专利数据库2020年.csv"
            raw_csv.write_text(
                "\n".join(
                    [
                        "申请号,专利类型,专利名称,摘要文本,申请日,公开公告日,授权公告日,申请人",
                        "P1,发明授权,标题1,摘要1,2020-01-01,2020-06-01,2020-07-01,申请人1",
                        "BAD,发明授权,坏行,坏摘要,2020-01-02,2020-06-02,2020-07-02,申请人2,多出来的字段",
                        "P2,发明授权,标题2,摘要2,2020-01-03,2020-06-03,2020-07-03,申请人3",
                    ]
                ),
                encoding="utf-8-sig",
            )

            result = build_raw_patent_authorized_parts(
                raw_patent_dir=raw_dir,
                shared_root=str(shared_root),
                chunksize=2,
                overwrite=True,
            )
            parquet_path = result["output_dir"] / "中国专利数据库2020年.parquet"
            parquet_df = pd.read_parquet(parquet_path)

            self.assertEqual(parquet_df["申请号"].tolist(), ["P1", "P2"])

    def test_build_raw_patent_authorized_parts_heals_trailing_empty_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            shared_root = root / "shared"
            raw_dir.mkdir()

            raw_csv = raw_dir / "中国专利数据库2020年.csv"
            raw_csv.write_text(
                "\n".join(
                    [
                        "申请号,专利类型,专利名称,摘要文本,申请日,公开公告日,授权公告日,申请人",
                        "P1,发明授权,标题1,摘要1,2020-01-01,2020-06-01,2020-07-01,申请人1",
                        "P2,发明授权,标题2,摘要2,2020-01-02,2020-06-02,2020-07-02,申请人2,",
                        "P3,发明申请,标题3,摘要3,2020-01-03,2020-06-03,,申请人3,",
                    ]
                ),
                encoding="utf-8-sig",
            )

            result = build_raw_patent_authorized_parts(
                raw_patent_dir=raw_dir,
                shared_root=str(shared_root),
                chunksize=2,
                overwrite=True,
            )
            parquet_path = result["output_dir"] / "中国专利数据库2020年.parquet"
            parquet_df = pd.read_parquet(parquet_path)
            metadata = json.loads((result["output_dir"] / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(parquet_df["申请号"].tolist(), ["P1", "P2"])
            self.assertEqual(metadata["rows_healed_trailing_empty_field_total"], 2)
            self.assertEqual(metadata["rows_skipped_bad_width_total"], 0)
            self.assertEqual(metadata["parts"][0]["rows_scanned"], 3)


if __name__ == "__main__":
    unittest.main()
