from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from patent_quality.case_analysis import analyze_patent_case, parse_config_literals


class PatentCaseAnalysisTests(unittest.TestCase):
    def test_parse_config_literals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            script_path = root / "config_script.py"
            script_path.write_text(
                "\n".join(
                    [
                        "from patent_quality.config import Config",
                        "cfg = Config(",
                        '    data_path="data/raw",',
                        '    stopword_paths=["stopword"],',
                        '    col_text_parts=["专利名称", "摘要文本"],',
                        "    min_term_count=5,",
                        ")",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_config_literals(script_path)
            self.assertEqual(parsed["data_path"], "data/raw")
            self.assertEqual(parsed["stopword_paths"], ["stopword"])
            self.assertEqual(parsed["col_text_parts"], ["专利名称", "摘要文本"])
            self.assertEqual(parsed["min_term_count"], 5)

    def test_analyze_patent_case_reports_pruning_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage1_dir = root / "stage1"
            raw_dir = root / "raw"
            for path in [
                stage1_dir / "df",
                stage1_dir / "vocab",
                stage1_dir / "tokens",
                stage1_dir / "index",
                stage1_dir / "stats",
                raw_dir,
            ]:
                path.mkdir(parents=True, exist_ok=True)

            config_script = root / "run_case.py"
            config_script.write_text(
                "\n".join(
                    [
                        "from patent_quality.config import Config",
                        "cfg = Config(",
                        '    data_path="data/raw",',
                        "    stopword_paths=[],",
                        "    user_dict_path=None,",
                        '    col_text_parts=["专利名称", "摘要文本"],',
                        "    min_term_count=2,",
                        "    max_doc_freq_ratio=0.5,",
                        '    manual_stopwords_path="manual_stopwords.txt",',
                        "    df_ratio_threshold=0.8,",
                        "    top_df_percent=0.25,",
                        "    topk_terms_per_doc=1,",
                        ")",
                    ]
                ),
                encoding="utf-8",
            )
            manual_stopwords = root / "manual_stopwords.txt"
            manual_stopwords.write_text("", encoding="utf-8")

            global_df = {
                "total_docs": 100,
                "df": {"电导": 10, "检测": 40, "池": 5, "进口": 5},
            }
            (stage1_dir / "df" / "global_df.json").write_text(
                json.dumps(global_df, ensure_ascii=False),
                encoding="utf-8",
            )
            (stage1_dir / "df" / "term_df_year=2019.json").write_text(
                json.dumps(
                    {
                        "year": 2019,
                        "docs": 20,
                        "df": {"电导": 2, "检测": 1, "池": 1, "进口": 5},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (stage1_dir / "df" / "term_df_year=2020.json").write_text(
                json.dumps(
                    {
                        "year": 2020,
                        "docs": 10,
                        "df": {"电导": 3, "检测": 9, "池": 2, "进口": 1},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (stage1_dir / "vocab" / "final_vocab.json").write_text(
                json.dumps(
                    {
                        "size": 4,
                        "vocab": {"电导": 0, "检测": 1, "池": 2, "进口": 3},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (stage1_dir / "tokens" / "year=2020.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "id": "P1",
                            "title": "电导检测池",
                            "tokens": ["电导", "检测", "池", "进口"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with (stage1_dir / "index" / "year=2020.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "申请号", "申请年份", "专利名称", "申请人"])
                writer.writerow([0, "P1", 2020, "电导检测池", "测试申请人"])
            with (stage1_dir / "stats" / "bsfs_year=2020.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "BS", "FS"])
                writer.writerow([0, 1.0, 3.0])
                writer.writerow([1, 1.0, 6.0])
                writer.writerow([2, 2.0, 2.0])
            with (raw_dir / "中国专利数据库2020年.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["申请号", "申请年份", "专利类型", "专利名称", "摘要文本", "公开（公告）日"])
                writer.writerow(["P1", 2020, "发明授权", "电导检测池", "电导 检测 池 进口", "2021-01-01"])

            result = analyze_patent_case(
                stage1_dir=stage1_dir,
                raw_data_path=raw_dir,
                application_no="P1",
                application_year=2020,
                title=None,
                title_contains=None,
                publication_date="2021",
                config_script=config_script,
                config_overrides={"manual_stopwords_path": str(manual_stopwords)},
                include_raw_cut=False,
            )

            detail_by_term = {
                item["term"]: item
                for item in result["term_analysis"]["term_details"]
            }
            self.assertEqual(result["patent"]["application_no"], "P1")
            self.assertAlmostEqual(result["patent"]["BS"], 1.0)
            self.assertAlmostEqual(result["patent"]["FS"], 3.0)
            self.assertEqual(result["year_quality_rank"]["rank_desc"], 2)
            self.assertEqual(result["year_quality_rank"]["total_patents_in_year"], 3)
            self.assertEqual(detail_by_term["检测"]["reason"], "year_df_ratio_pruning")
            self.assertEqual(detail_by_term["电导"]["reason"], "year_top_df_pruning")
            self.assertEqual(detail_by_term["进口"]["reason"], "document_topk_pruning")
            self.assertEqual(detail_by_term["池"]["reason"], "kept")
            self.assertIn("超过阈值", detail_by_term["检测"]["reason_detail"])
            self.assertIn("落入按年高频剪枝前", detail_by_term["电导"]["reason_detail"])
            self.assertIn("每篇文档只保留前 1 个词", detail_by_term["进口"]["reason_detail"])
            self.assertTrue(detail_by_term["池"]["participates_in_final_similarity"])


if __name__ == "__main__":
    unittest.main()
