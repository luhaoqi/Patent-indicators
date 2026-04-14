from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


READ_ENCODINGS: Sequence[str] = ("utf-8-sig", "utf-8", "gb18030")
WINDOW_STAGE1_DIRS = {
    "window1": Path("outputs/experiments/标题_摘要_ExactTime_window_1/stage1_exact"),
    "window3": Path("outputs/experiments/标题_摘要_ExactTime_window_3/stage1_exact"),
}
WINDOW_SPECS = [
    {
        "window": "window1",
        "status_col": "标题_摘要_ExactTime_window_1_状态",
        "matched_year_col": "标题_摘要_ExactTime_window_1_命中公开年份",
        "rank_col": "标题_摘要_ExactTime_window_1_排名",
        "year_total_col": "标题_摘要_ExactTime_window_1_年内专利数",
        "rank_percent_col": "标题_摘要_ExactTime_window_1_排名百分比",
        "quantity_q_col": "标题_摘要_ExactTime_window_1_quantity_q",
    },
    {
        "window": "window3",
        "status_col": "标题_摘要_ExactTime_window_3_状态",
        "matched_year_col": "标题_摘要_ExactTime_window_3_命中公开年份",
        "rank_col": "标题_摘要_ExactTime_window_3_排名",
        "year_total_col": "标题_摘要_ExactTime_window_3_年内专利数",
        "rank_percent_col": "标题_摘要_ExactTime_window_3_排名百分比",
        "quantity_q_col": "标题_摘要_ExactTime_window_3_quantity_q",
    },
]
SUMMARY_SENTINEL = "__summary_min_rank_percent_top5__"


@dataclass(frozen=True)
class RankedEntry:
    source_lookup_csv: str
    source_award_csv: str
    application_no: str
    query_public_year: str
    matched_public_year: str
    window: str
    rank: str
    year_total: str
    rank_percent: float
    quantity_q: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总专利金奖 exact-time 查询结果中每个文件的最小排名百分比，并导出全局 TopN。")
    parser.add_argument("input_dir", help="包含 *_exact_time_lookup.csv 的目录")
    parser.add_argument("output_csv", nargs="?", help="输出 CSV 路径")
    parser.add_argument("--top-per-file", type=int, default=5, help="每个结果文件保留最小的前 N 个百分比，默认 5")
    parser.add_argument("--top-global", type=int, default=10, help="全局导出最小的前 N 个百分比，默认 10")
    return parser.parse_args(argv)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV 头为空: {path}")
                fieldnames = list(reader.fieldnames)
                if fieldnames:
                    fieldnames[0] = fieldnames[0].lstrip("\ufeff")
                    reader.fieldnames = fieldnames
                return [{key: value if value is not None else "" for key, value in row.items()} for row in reader]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: object) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_award_csv_name(lookup_name: str) -> str:
    suffix = "_exact_time_lookup.csv"
    if lookup_name.endswith(suffix):
        return lookup_name[: -len(suffix)] + ".csv"
    return lookup_name


def iter_ranked_entries(path: Path) -> Iterable[RankedEntry]:
    rows = read_csv_rows(path)
    for row in rows:
        application_no = normalize_text(row.get("申请号"))
        if not application_no or application_no == SUMMARY_SENTINEL:
            continue
        for spec in WINDOW_SPECS:
            if normalize_text(row.get(spec["status_col"])) != "找到":
                continue
            rank_percent = parse_float(row.get(spec["rank_percent_col"]))
            if rank_percent is None:
                continue
            yield RankedEntry(
                source_lookup_csv=path.name,
                source_award_csv=infer_award_csv_name(path.name),
                application_no=application_no,
                query_public_year=normalize_text(row.get("查询公开年份")),
                matched_public_year=normalize_text(row.get(spec["matched_year_col"])),
                window=spec["window"],
                rank=normalize_text(row.get(spec["rank_col"])),
                year_total=normalize_text(row.get(spec["year_total_col"])),
                rank_percent=rank_percent,
                quantity_q=normalize_text(row.get(spec["quantity_q_col"])),
            )


def lookup_bsfs_for_entries(entries: Sequence[RankedEntry]) -> Dict[tuple[str, str, str], tuple[str, str]]:
    grouped_apps: Dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        if not entry.matched_public_year:
            continue
        grouped_apps.setdefault((entry.window, entry.matched_public_year), set()).add(entry.application_no)

    result: Dict[tuple[str, str, str], tuple[str, str]] = {}
    for (window, public_year), application_nos in grouped_apps.items():
        stage1_dir = WINDOW_STAGE1_DIRS[window]
        index_path = stage1_dir / "index" / f"year={public_year}.csv"
        stats_path = stage1_dir / "stats" / f"bsfs_year={public_year}.csv"
        if not index_path.exists() or not stats_path.exists():
            continue

        target_rows: Dict[str, str] = {}
        with index_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                application_no = normalize_text(row.get("申请号"))
                row_idx = normalize_text(row.get("row"))
                if not application_no or not row_idx or application_no not in application_nos:
                    continue
                target_rows[application_no] = row_idx
                if len(target_rows) == len(application_nos):
                    break

        wanted_row_ids = set(target_rows.values())
        row_to_bsfs: Dict[str, tuple[str, str]] = {}
        with stats_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_idx = normalize_text(row.get("row"))
                if row_idx not in wanted_row_ids:
                    continue
                row_to_bsfs[row_idx] = (
                    normalize_text(row.get("BS")),
                    normalize_text(row.get("FS")),
                )
                if len(row_to_bsfs) == len(wanted_row_ids):
                    break

        for application_no, row_idx in target_rows.items():
            result[(window, public_year, application_no)] = row_to_bsfs.get(row_idx, ("", ""))
    return result


def default_output_path(input_dir: Path) -> Path:
    return input_dir / "专利金奖_exact_time_top10_rank_percent_summary.csv"


def run(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_csv) if args.output_csv else default_output_path(input_dir)

    lookup_paths = sorted(input_dir.glob("*_exact_time_lookup.csv"))
    per_file_top_entries: List[RankedEntry] = []
    for path in lookup_paths:
        entries = sorted(
            iter_ranked_entries(path),
            key=lambda item: (item.rank_percent, item.source_lookup_csv, item.application_no, item.window, item.query_public_year),
        )
        per_file_top_entries.extend(entries[: args.top_per_file])

    global_top_entries = sorted(
        per_file_top_entries,
        key=lambda item: (item.rank_percent, item.source_lookup_csv, item.application_no, item.window, item.query_public_year),
    )[: args.top_global]
    bsfs_by_entry = lookup_bsfs_for_entries(global_top_entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "全局排名",
                "排名百分比",
                "所属文件名",
                "原始名单文件名",
                "申请号",
                "window",
                "查询公开年份",
                "命中公开年份",
                "排名",
                "年内专利数",
                "BS",
                "FS",
                "quantity_q",
            ],
        )
        writer.writeheader()
        for idx, entry in enumerate(global_top_entries, start=1):
            bs, fs = bsfs_by_entry.get((entry.window, entry.matched_public_year, entry.application_no), ("", ""))
            writer.writerow(
                {
                    "全局排名": idx,
                    "排名百分比": f"{entry.rank_percent:.6f}",
                    "所属文件名": entry.source_lookup_csv,
                    "原始名单文件名": entry.source_award_csv,
                    "申请号": entry.application_no,
                    "window": entry.window,
                    "查询公开年份": entry.query_public_year,
                    "命中公开年份": entry.matched_public_year,
                    "排名": entry.rank,
                    "年内专利数": entry.year_total,
                    "BS": bs,
                    "FS": fs,
                    "quantity_q": entry.quantity_q,
                }
            )

    print(f"[done] output_csv={output_path}")
    for idx, entry in enumerate(global_top_entries, start=1):
        print(
            f"[top{idx}] pct={entry.rank_percent:.6f} file={entry.source_lookup_csv} "
            f"app={entry.application_no} window={entry.window}"
        )
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
