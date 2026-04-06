from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from patent_quality.similarity_case_analysis import run_similarity_case_analysis


class SimilarityCaseAnalysisTests(unittest.TestCase):
    def test_run_similarity_case_analysis_outputs_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage1_dir = root / "experiment" / "stage1"
            output_dir = root / "outputs"
            for path in [
                stage1_dir / "index",
                stage1_dir / "tokens",
                stage1_dir / "vocab",
                stage1_dir / "vectors_filtered",
                stage1_dir / "pair_contrib",
            ]:
                path.mkdir(parents=True, exist_ok=True)

            (stage1_dir / "vocab" / "final_vocab.json").write_text(
                json.dumps(
                    {
                        "size": 3,
                        "vocab": {"alpha": 0, "beta": 1, "gamma": 2},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with (stage1_dir / "index" / "year=2019.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "申请号", "申请年份", "专利名称"])
                writer.writerow([0, "B1", 2019, "backward-1"])
                writer.writerow([1, "B2", 2019, "backward-2"])

            with (stage1_dir / "index" / "year=2020.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "申请号", "申请年份", "专利名称"])
                writer.writerow([0, "T1", 2020, "target"])

            with (stage1_dir / "index" / "year=2021.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row", "申请号", "申请年份", "专利名称"])
                writer.writerow([0, "F1", 2021, "forward-1"])
                writer.writerow([1, "F2", 2021, "forward-2"])

            with (stage1_dir / "tokens" / "year=2020.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "id": "T1",
                            "title": "target",
                            "tokens": ["alpha", "beta", "alpha", "delta"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            sparse.save_npz(
                stage1_dir / "vectors_filtered" / "year=2019.npz",
                sparse.csr_matrix(
                    np.array(
                        [
                            [0.5, 0.5, 0.0],
                            [0.1, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
                ),
            )
            sparse.save_npz(
                stage1_dir / "vectors_filtered" / "year=2020.npz",
                sparse.csr_matrix(
                    np.array(
                        [
                            [0.6, 0.8, 0.0],
                        ],
                        dtype=np.float32,
                    )
                ),
            )
            sparse.save_npz(
                stage1_dir / "vectors_filtered" / "year=2021.npz",
                sparse.csr_matrix(
                    np.array(
                        [
                            [0.0, 0.5, 0.0],
                            [-0.1, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
                ),
            )

            np.savez(
                stage1_dir / "pair_contrib" / "x=2019_y=2020.npz",
                contrib_x=np.zeros(2, dtype=np.float32),
                contrib_y=np.zeros(1, dtype=np.float32),
                meta_json=json.dumps({"thr": 0.05, "window_size": 1}, ensure_ascii=False),
            )

            summary = run_similarity_case_analysis(
                stage1_dir=stage1_dir,
                application_no="T1",
                year=2020,
                output_dir=output_dir,
            )

            self.assertEqual(summary["window_size"], 1)
            self.assertAlmostEqual(summary["similarity_threshold"], 0.05)
            self.assertEqual(summary["target_stage1_token_count"], 4)
            self.assertEqual(summary["target_final_vector_term_count"], 2)

            with (output_dir / "term_contribution.csv").open("r", encoding="utf-8") as fh:
                term_rows = {row["词汇"]: row for row in csv.DictReader(fh)}

            self.assertEqual(term_rows["alpha"]["是否参与最终计算"], "1")
            self.assertEqual(term_rows["alpha"]["stage1词频"], "2")
            self.assertAlmostEqual(float(term_rows["alpha"]["最终权重"]), 0.6, places=6)
            self.assertAlmostEqual(float(term_rows["alpha"]["向前原始贡献"]), 0.36, places=6)
            self.assertAlmostEqual(float(term_rows["alpha"]["向后原始贡献"]), -0.06, places=6)
            self.assertAlmostEqual(float(term_rows["alpha"]["总计入BSFS贡献"]), 0.36, places=6)

            self.assertEqual(term_rows["beta"]["是否参与最终计算"], "1")
            self.assertAlmostEqual(float(term_rows["beta"]["向前计入BS贡献"]), 0.4, places=6)
            self.assertAlmostEqual(float(term_rows["beta"]["向后计入FS贡献"]), 0.4, places=6)
            self.assertAlmostEqual(float(term_rows["beta"]["总计入BSFS贡献"]), 0.8, places=6)

            self.assertEqual(term_rows["delta"]["是否参与最终计算"], "0")
            self.assertAlmostEqual(float(term_rows["delta"]["总原始贡献"]), 0.0, places=6)

            with (output_dir / "backward_similarity.csv").open("r", encoding="utf-8") as fh:
                backward_rows = list(csv.DictReader(fh))
            self.assertEqual([row["申请号"] for row in backward_rows], ["B1", "B2"])
            self.assertAlmostEqual(float(backward_rows[0]["相似度"]), 0.7, places=6)
            self.assertAlmostEqual(float(backward_rows[1]["相似度"]), 0.06, places=6)
            self.assertEqual(backward_rows[0]["保存区段"], "all")

            with (output_dir / "forward_similarity.csv").open("r", encoding="utf-8") as fh:
                forward_rows = list(csv.DictReader(fh))
            self.assertEqual([row["申请号"] for row in forward_rows], ["F1", "F2"])
            self.assertAlmostEqual(float(forward_rows[0]["相似度"]), 0.4, places=6)
            self.assertAlmostEqual(float(forward_rows[1]["相似度"]), -0.06, places=6)
            self.assertEqual(forward_rows[1]["是否计入BSFS"], "0")

            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()

