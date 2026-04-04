from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from scipy import sparse

from analysis.build_firm_year_innovation import build_firm_year_innovation
from analysis.build_raw_patent_authorized_parts import build_raw_patent_authorized_parts
from analysis.export_top_patents_by_year import export_top_patents_by_year
from patent_quality.config import Config
from patent_quality.data_loader import iter_clean_docs
from patent_quality.similarity import compute_bs_fs


def _day_ord(text: str) -> int:
    dt = pd.Timestamp(text).date()
    return dt.toordinal() - date(1970, 1, 1).toordinal()


class ExactDateTests(unittest.TestCase):
    def test_build_raw_patent_authorized_parts_sorts_and_adds_date_ord(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            shared_root = root / "shared"
            raw_dir.mkdir()

            pd.DataFrame(
                [
                    {"申请号": "P2", "专利类型": "发明授权", "专利名称": "标题2", "摘要文本": "摘要2", "申请日": "2020-01-02", "公开公告日": "2020-06-02", "申请人": "申请人2"},
                    {"申请号": "P_BAD", "专利类型": "发明授权", "专利名称": "标题坏", "摘要文本": "摘要坏", "申请日": "2020-01-03", "公开公告日": "bad-date", "申请人": "申请人坏"},
                    {"申请号": "P1", "专利类型": "发明授权", "专利名称": "标题1", "摘要文本": "摘要1", "申请日": "2020-01-01", "公开公告日": "2020-06-01", "申请人": "申请人1"},
                ]
            ).to_csv(raw_dir / "中国专利数据库2020年.csv", index=False, encoding="utf-8-sig")

            result = build_raw_patent_authorized_parts(
                raw_patent_dir=raw_dir,
                shared_root=str(shared_root),
                chunksize=2,
                overwrite=True,
            )
            parquet_path = result["output_dir"] / "中国专利数据库2020年.parquet"
            parquet_df = pd.read_parquet(parquet_path)
            metadata = json.loads(result["metadata_path"].read_text(encoding="utf-8"))

            self.assertEqual(parquet_df["申请号"].tolist(), ["P1", "P2", "P_BAD"])
            self.assertIn("公开公告年份", parquet_df.columns)
            self.assertIn("公开公告日_ord", parquet_df.columns)
            self.assertEqual(parquet_df.loc[0, "公开公告年份"], "2020")
            self.assertEqual(int(parquet_df.loc[0, "公开公告日_ord"]), _day_ord("2020-06-01"))
            self.assertEqual(metadata["invalid_publish_date_rows_total"], 1)
            self.assertEqual(metadata["parts"][0]["sort_by"], ["公开公告日_ord", "申请号"])

    def test_exact_bsfs_same_day_and_boundary_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = Config(
                data_path=".",
                artifacts_dir=str(root),
                exact_date=True,
                window_size=1,
                similarity_threshold=0.0,
                use_vectors_filtered_for_bsfs=False,
                skip_if_exists=False,
            )
            cfg.ensure_dirs()

            matrix_2020 = sparse.csr_matrix([[1.0], [1.0], [1.0]], dtype="float32")
            matrix_2021 = sparse.csr_matrix([[1.0], [1.0]], dtype="float32")
            sparse.save_npz(root / "vectors" / "year=2020.npz", matrix_2020)
            sparse.save_npz(root / "vectors" / "year=2021.npz", matrix_2021)

            with (root / "index" / "year=2020.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "申请号", "公开公告年份", "公开公告日", "公开公告日_ord", "专利名称"])
                writer.writerow([0, "A", 2020, "2020-06-01", _day_ord("2020-06-01"), "A"])
                writer.writerow([1, "B", 2020, "2020-06-02", _day_ord("2020-06-02"), "B"])
                writer.writerow([2, "C", 2020, "2020-06-02", _day_ord("2020-06-02"), "C"])

            with (root / "index" / "year=2021.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "申请号", "公开公告年份", "公开公告日", "公开公告日_ord", "专利名称"])
                writer.writerow([0, "D", 2021, "2021-05-31", _day_ord("2021-05-31"), "D"])
                writer.writerow([1, "E", 2021, "2021-06-03", _day_ord("2021-06-03"), "E"])

            compute_bs_fs(cfg)

            year_2020 = pd.read_csv(root / "stats" / "bsfs_year=2020.csv")
            year_2021 = pd.read_csv(root / "stats" / "bsfs_year=2021.csv")

            self.assertEqual(year_2020["BS"].tolist(), [0.0, 1.0, 1.0])
            self.assertEqual(year_2020["FS"].tolist(), [3.0, 1.0, 1.0])
            self.assertEqual(year_2021["BS"].tolist(), [3.0, 1.0])
            self.assertEqual(year_2021["FS"].tolist(), [1.0, 0.0])

    def test_iter_clean_docs_backfills_missing_public_date_ord_from_publish_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shared_dir = root / "shared_parts"
            shared_dir.mkdir()

            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "专利类型": "发明授权",
                        "专利名称": "标题1",
                        "摘要文本": "摘要1",
                        "公开公告日": "2020-06-01",
                        "公开公告年份": "2020",
                    },
                    {
                        "申请号": "P2",
                        "专利类型": "发明授权",
                        "专利名称": "标题2",
                        "摘要文本": "摘要2",
                        "公开公告日": "bad-date",
                        "公开公告年份": "2020",
                    },
                ]
            ).to_parquet(shared_dir / "中国专利数据库2020年.parquet", index=False)

            cfg = Config(
                data_path=".",
                artifacts_dir=str(root / "artifacts"),
                exact_date=True,
                shared_authorized_parts_dir=str(shared_dir),
                col_text_parts=["专利名称", "摘要文本"],
                skip_if_exists=False,
            )

            docs = list(iter_clean_docs(cfg))

            self.assertEqual(docs, [("P1", 2020, "标题1 摘要1")])

    def test_stage2_exact_outputs_use_public_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "experiments"
            shared_root = root / "shared"
            stage2_exact_data_dir = output_root / "exp_exact" / "stage2_exact" / "data"
            stage2_exact_data_dir.mkdir(parents=True)

            panel_path = stage2_exact_data_dir / "experiment_patent_panel.parquet"
            pd.DataFrame(
                [
                    {
                        "申请号": "P1",
                        "申请年份": 2021,
                        "公开公告年份": 2020,
                        "专利名称": "标题1",
                        "申请人": "申请人1",
                        "统一社会信用代码": "U1",
                        "BS": 0.1,
                        "FS": 0.9,
                        "Quality_q": 9.0,
                        "被引证次数": 3,
                    }
                ]
            ).to_parquet(panel_path, index=False)

            summary = export_top_patents_by_year(
                experiment_id="exp_exact",
                output_root=str(output_root),
                experiment_patent_panel_path=panel_path,
                shared_root=str(shared_root),
                top_n=1,
                skip_company_lookup=True,
                skip_raw_lookup=True,
                exact_date=True,
            )
            top_2020 = pd.read_csv(output_root / "exp_exact" / "stage2_exact" / "tables" / "top_patents_by_year" / "top_patents_year=2020_top1.csv")
            self.assertEqual(summary["year_col"], "公开公告年份")
            self.assertEqual(top_2020.loc[0, "申请号"], "P1")

            (shared_root / "ucc_mapping").mkdir(parents=True)
            pd.DataFrame(
                [{"Stkid": "000001", "ShortName": "FirmA", "year": 2020, "UCC": "U1"}]
            ).to_parquet(shared_root / "ucc_mapping" / "ucc_exploded.parquet", index=False)
            innovation_path = build_firm_year_innovation(
                experiment_id="exp_exact",
                output_root=str(output_root),
                experiment_patent_panel_path=panel_path,
                ucc_exploded_path=shared_root / "ucc_mapping" / "ucc_exploded.parquet",
                shared_root=str(shared_root),
                top_k=1,
                exact_date=True,
            )
            innovation_df = pd.read_parquet(innovation_path)
            self.assertEqual(int(innovation_df.loc[0, "year"]), 2020)


if __name__ == "__main__":
    unittest.main()
