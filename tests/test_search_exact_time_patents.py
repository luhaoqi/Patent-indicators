from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from search_exact_time_patents import parse_args, run


WINDOW_1 = "标题_摘要_ExactTime_window_1"
WINDOW_3 = "标题_摘要_ExactTime_window_3"


def _write_csv(path: Path, header: list[str], rows: list[list[object]], encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class SearchExactTimePatentsTests(unittest.TestCase):
    def test_batch_lookup_outputs_hits_and_missing_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "input.csv"
            output_csv = root / "output.csv"
            output_root = root / "experiments"
            shared_dir = root / "shared_authorized_parts"
            raw_dir = root / "raw"

            _write_csv(
                input_csv,
                ["申请号", "公开年份", "备注"],
                [
                    ["A", 2020, "found"],
                    ["B", 2020, "token_only"],
                    ["C", 2020, "not_authorized"],
                    ["D", 2020, "wrong_year"],
                ],
            )

            for experiment_id, stats_rows in (
                (
                    WINDOW_1,
                    [
                        [0, 1.0, 5.0],
                        [1, 1.0, 3.0],
                    ],
                ),
                (
                    WINDOW_3,
                    [
                        [0, 2.0, 2.0],
                        [1, 1.0, 3.0],
                    ],
                ),
            ):
                stage1_dir = output_root / experiment_id / "stage1_exact"
                _write_csv(
                    stage1_dir / "index" / "year=2020.csv",
                    ["row", "申请号", "公开公告年份", "公开公告日", "公开公告日_ord", "专利名称"],
                    [
                        [0, "A", 2020, "2020-01-01", 18262, "标题A"],
                        [1, "X", 2020, "2020-01-02", 18263, "标题X"],
                    ],
                )
                _write_csv(
                    stage1_dir / "stats" / "bsfs_year=2020.csv",
                    ["row", "BS", "FS"],
                    stats_rows,
                    encoding="utf-8",
                )
                _write_jsonl(
                    stage1_dir / "tokens" / "year=2020.jsonl",
                    [
                        {"id": "A", "title": "标题A", "tokens": ["词A"]},
                        {"id": "B", "title": "标题B", "tokens": ["词B"]},
                    ],
                )

            shared_dir.mkdir(parents=True, exist_ok=True)
            shared_table = pa.Table.from_pylist(
                [
                    {"申请号": "A", "公开公告年份": "2020", "公开公告日": "2020-01-01", "专利名称": "标题A", "专利类型": "发明授权"},
                    {"申请号": "B", "公开公告年份": "2020", "公开公告日": "2020-01-05", "专利名称": "标题B", "专利类型": "发明授权"},
                    {"申请号": "D", "公开公告年份": "2021", "公开公告日": "2021-03-01", "专利名称": "标题D", "专利类型": "发明授权"},
                ]
            )
            pq.write_table(shared_table, shared_dir / "part-000.parquet")

            _write_csv(
                raw_dir / "中国专利数据库2020年.csv",
                ["专利名称", "专利类型", "申请号", "公开公告日", "公开公告年份"],
                [
                    ["标题C", "实用新型", "C", "2020-02-01", "2020"],
                ],
            )

            args = parse_args(
                [
                    str(input_csv),
                    str(output_csv),
                    "--output-root",
                    str(output_root),
                    "--shared-authorized-parts-dir",
                    str(shared_dir),
                    "--raw-data-path",
                    str(raw_dir),
                    "--raw-lookup-mode",
                    "scan",
                ]
            )
            result_path = run(args)

            self.assertEqual(result_path, output_csv)
            with output_csv.open("r", encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))

            found_row = rows[0]
            self.assertEqual(found_row[f"{WINDOW_1}_状态"], "找到")
            self.assertEqual(found_row[f"{WINDOW_1}_排名"], "1")
            self.assertEqual(found_row[f"{WINDOW_1}_年内专利数"], "2")
            self.assertAlmostEqual(float(found_row[f"{WINDOW_1}_排名百分比"]), 50.0)
            self.assertAlmostEqual(float(found_row[f"{WINDOW_1}_quantity_q"]), 5.0, places=6)
            self.assertEqual(found_row[f"{WINDOW_3}_排名"], "2")
            self.assertAlmostEqual(float(found_row[f"{WINDOW_3}_排名百分比"]), 100.0)
            self.assertAlmostEqual(float(found_row[f"{WINDOW_3}_quantity_q"]), 1.0, places=6)

            token_only_row = rows[1]
            self.assertEqual(token_only_row[f"{WINDOW_1}_状态"], "未找到")
            self.assertIn("stage1 tokens", token_only_row[f"{WINDOW_1}_原因"])
            self.assertIn("未进入 stage1 index", token_only_row[f"{WINDOW_1}_原因"])

            non_auth_row = rows[2]
            self.assertIn("不是发明授权", non_auth_row[f"{WINDOW_1}_原因"])
            self.assertIn("实用新型", non_auth_row[f"{WINDOW_1}_原因"])

            wrong_year_row = rows[3]
            self.assertIn("公开年份为 2021", wrong_year_row[f"{WINDOW_1}_原因"])
            self.assertIn("不是输入的 2020", wrong_year_row[f"{WINDOW_1}_原因"])


if __name__ == "__main__":
    unittest.main()
