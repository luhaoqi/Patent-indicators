from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.dataset as ds

from patent_quality.project_paths import build_experiment_layout, resolve_project_path


READ_ENCODINGS: Sequence[str] = ("utf-8-sig", "utf-8", "gb18030")
DEFAULT_EXPERIMENT_IDS = [
    "标题_摘要_ExactTime_window_1",
    "标题_摘要_ExactTime_window_3",
]
DEFAULT_OUTPUT_ROOT = "outputs/experiments"
DEFAULT_SHARED_AUTHORIZED_PARTS_DIR = "outputs/shared/raw_patent_authorized_parts"
DEFAULT_RAW_DATA_PATH = "data/raw/中国专利分年份保存数据1985-2025"
AUTHORIZED_PATENT_TYPE = "发明授权"
EPSILON = 1e-8

APPLICATION_COL_CANDIDATES = (
    "申请号",
    "专利申请号",
    "application_no",
    "application number",
    "application_number",
)
PUBLIC_YEAR_COL_CANDIDATES = (
    "公开年份",
    "公开公告年份",
    "public_year",
    "public year",
    "year",
    "年份",
)


@dataclass(frozen=True)
class AuthorizedPatentRecord:
    application_no: str
    public_year: Optional[int]
    public_date: str
    title: str
    patent_type: str


@dataclass(frozen=True)
class RawPatentRecord:
    application_no: str
    patent_type: str
    public_year: Optional[int]
    public_date: str
    title: str
    source_file: str
    source_line: int


@dataclass(frozen=True)
class ExperimentHit:
    rank: int
    year_total: int
    rank_percent: float
    quantity_q: float


@dataclass
class YearLookup:
    metrics_by_app: Dict[str, List[ExperimentHit]]
    missing_parts: List[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量查询 exact-time 实验中给定专利在对应公开年份的排名、排名百分比和 quantity_q。"
    )
    parser.add_argument("input_csv", help="输入 CSV，至少包含申请号和公开年份两列")
    parser.add_argument("output_csv", nargs="?", help="输出 CSV；不传则默认写到输入文件同目录")
    parser.add_argument("--application-col", help="输入 CSV 中的申请号列名")
    parser.add_argument("--public-year-col", help="输入 CSV 中的公开年份列名")
    parser.add_argument(
        "--experiment-id",
        dest="experiment_ids",
        action="append",
        help="要查询的 experiment_id；可重复传，多次不传则默认查询 window_1 和 window_3",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="实验输出根目录，默认 outputs/experiments",
    )
    parser.add_argument(
        "--shared-authorized-parts-dir",
        default=DEFAULT_SHARED_AUTHORIZED_PARTS_DIR,
        help="共享发明授权 parquet 目录，默认 outputs/shared/raw_patent_authorized_parts",
    )
    parser.add_argument(
        "--raw-data-path",
        default=DEFAULT_RAW_DATA_PATH,
        help="原始专利 CSV 目录，仅用于补充缺失原因",
    )
    parser.add_argument(
        "--raw-lookup-mode",
        choices=("auto", "rg", "scan", "skip"),
        default="auto",
        help=(
            "缺失原因的原始数据回查方式：auto=优先用 rg，没有则跳过；"
            "rg=必须用 rg；scan=逐行扫描原始 CSV；skip=不回查原始 CSV。"
        ),
    )
    parser.add_argument(
        "--output-encoding",
        default="utf-8-sig",
        help="输出 CSV 编码，默认 utf-8-sig",
    )
    return parser.parse_args(argv)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_application_no(value: object) -> str:
    return normalize_text(value).upper()


def parse_public_year(value: object) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{4}\.0+", text):
        text = text.split(".", 1)[0]
    if re.fullmatch(r"\d{4}", text):
        return int(text)
    match = re.search(r"(\d{4})", text)
    if match:
        return int(match.group(1))
    return None


def _read_csv_dict_rows_with_fallback(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
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
                rows = []
                for row in reader:
                    rows.append({key: value if value is not None else "" for key, value in row.items()})
                return fieldnames, rows
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    return _read_csv_dict_rows_with_fallback(path)


def normalize_column_name(name: str) -> str:
    return re.sub(r"[\s_\-（）()]+", "", normalize_text(name)).lower()


def resolve_input_column(fieldnames: Sequence[str], explicit: Optional[str], candidates: Sequence[str], label: str) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"输入 CSV 中找不到列 {explicit!r}（用于 {label}）")
        return explicit

    field_map = {normalize_column_name(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in field_map:
            return field_map[key]
    raise ValueError(
        f"输入 CSV 中无法自动识别 {label} 列，请显式传 --{'application-col' if label == '申请号' else 'public-year-col'}。"
    )


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_exact_time_lookup.csv")


def format_years(years: Iterable[Optional[int]]) -> str:
    clean = sorted({year for year in years if year is not None})
    return ",".join(str(year) for year in clean)


class ExperimentLookup:
    def __init__(self, experiment_id: str, stage1_dir: Path, tracked_apps: Iterable[str], epsilon: float = EPSILON) -> None:
        self.experiment_id = experiment_id
        self.stage1_dir = stage1_dir
        self.tracked_apps = {app for app in tracked_apps if app}
        self.epsilon = float(epsilon)
        self._year_cache: Dict[int, YearLookup] = {}
        self._token_year_cache: Dict[int, set[str]] = {}

    def lookup(self, application_no: str, public_year: int) -> Tuple[Optional[ExperimentHit], str]:
        year_lookup = self._load_year(public_year)
        matches = year_lookup.metrics_by_app.get(application_no, [])
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return None, f"该实验在公开年份={public_year} 的 stage1 index 中命中了多条同申请号记录。"
        if year_lookup.missing_parts:
            missing = "、".join(year_lookup.missing_parts)
            return None, f"该实验缺少公开年份={public_year} 的产物文件：{missing}。"
        return None, ""

    def has_token_hit(self, application_no: str, public_year: int) -> bool:
        return application_no in self._load_token_year(public_year)

    def _load_year(self, public_year: int) -> YearLookup:
        cached = self._year_cache.get(public_year)
        if cached is not None:
            return cached

        index_path = self.stage1_dir / "index" / f"year={public_year}.csv"
        stats_path = self.stage1_dir / "stats" / f"bsfs_year={public_year}.csv"
        missing_parts = []
        if not index_path.exists():
            missing_parts.append(index_path.name)
        if not stats_path.exists():
            missing_parts.append(stats_path.name)
        if missing_parts:
            result = YearLookup(metrics_by_app={}, missing_parts=missing_parts)
            self._year_cache[public_year] = result
            return result

        stats_by_row: Dict[int, Tuple[float, float]] = {}
        _stats_fieldnames, stats_rows = _read_csv_dict_rows_with_fallback(stats_path)
        for row in stats_rows:
            try:
                row_idx = int(normalize_text(row.get("row")))
            except ValueError:
                continue
            try:
                bs = float(normalize_text(row.get("BS")) or 0.0)
            except ValueError:
                bs = 0.0
            try:
                fs = float(normalize_text(row.get("FS")) or 0.0)
            except ValueError:
                fs = 0.0
            stats_by_row[row_idx] = (bs, fs)

        staged_hits: List[Tuple[str, float]] = []
        _index_fieldnames, index_rows = _read_csv_dict_rows_with_fallback(index_path)
        for row in index_rows:
            try:
                row_idx = int(normalize_text(row.get("row")))
            except ValueError:
                continue
            bs, fs = stats_by_row.get(row_idx, (0.0, 0.0))
            quantity_q = fs / (bs + self.epsilon)
            app_no = normalize_application_no(row.get("申请号"))
            staged_hits.append((app_no, quantity_q))

        sorted_qs = sorted((quantity_q for _, quantity_q in staged_hits), reverse=True)
        rank_by_q: Dict[float, int] = {}
        for idx, quantity_q in enumerate(sorted_qs, start=1):
            rank_by_q.setdefault(quantity_q, idx)
        total = len(staged_hits)

        metrics_by_app: Dict[str, List[ExperimentHit]] = defaultdict(list)
        for app_no, quantity_q in staged_hits:
            rank = rank_by_q[quantity_q]
            rank_percent = (rank / total * 100.0) if total else 0.0
            metrics_by_app[app_no].append(
                ExperimentHit(
                    rank=rank,
                    year_total=total,
                    rank_percent=rank_percent,
                    quantity_q=quantity_q,
                )
            )

        result = YearLookup(metrics_by_app=dict(metrics_by_app), missing_parts=[])
        self._year_cache[public_year] = result
        return result

    def _load_token_year(self, public_year: int) -> set[str]:
        cached = self._token_year_cache.get(public_year)
        if cached is not None:
            return cached

        token_path = self.stage1_dir / "tokens" / f"year={public_year}.jsonl"
        matches: set[str] = set()
        if token_path.exists():
            with token_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    app_no = normalize_application_no(obj.get("id"))
                    if app_no and (not self.tracked_apps or app_no in self.tracked_apps):
                        matches.add(app_no)

        self._token_year_cache[public_year] = matches
        return matches


def load_authorized_records(shared_dir: Path, application_nos: Sequence[str]) -> Dict[str, List[AuthorizedPatentRecord]]:
    application_nos = sorted({app for app in application_nos if app})
    if not application_nos:
        return {}
    parquet_paths = sorted(shared_dir.glob("*.parquet"))
    if not parquet_paths:
        return {}

    dataset = ds.dataset([str(path) for path in parquet_paths], format="parquet")
    columns = ["申请号", "公开公告年份", "公开公告日", "专利名称", "专利类型"]
    records_by_app: Dict[str, List[AuthorizedPatentRecord]] = defaultdict(list)
    batch_size = 2000
    for start in range(0, len(application_nos), batch_size):
        batch = application_nos[start : start + batch_size]
        table = dataset.to_table(columns=columns, filter=ds.field("申请号").isin(batch))
        payload = table.to_pylist()
        for row in payload:
            app_no = normalize_application_no(row.get("申请号"))
            records_by_app[app_no].append(
                AuthorizedPatentRecord(
                    application_no=app_no,
                    public_year=parse_public_year(row.get("公开公告年份")),
                    public_date=normalize_text(row.get("公开公告日")),
                    title=normalize_text(row.get("专利名称")),
                    patent_type=normalize_text(row.get("专利类型")),
                )
            )
    return dict(records_by_app)


def list_csv_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".csv")
    return [path]


def load_csv_header(path: Path) -> Tuple[List[str], str]:
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                if header:
                    header[0] = header[0].lstrip("\ufeff")
                return header, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV 头部: {path}") from last_error


def parse_csv_row_from_line(content: str, expected_width: int) -> Optional[List[str]]:
    try:
        row = next(csv.reader([content]))
    except Exception:
        return None
    if len(row) == expected_width:
        return row
    if len(row) == expected_width + 1 and row[-1] == "":
        return row[:-1]
    return None


def raw_record_from_mapping(values: Dict[str, str], source_file: str, source_line: int) -> RawPatentRecord:
    public_date = normalize_text(values.get("公开公告日"))
    return RawPatentRecord(
        application_no=normalize_application_no(values.get("申请号")),
        patent_type=normalize_text(values.get("专利类型")),
        public_year=parse_public_year(values.get("公开公告年份")) or parse_public_year(public_date),
        public_date=public_date,
        title=normalize_text(values.get("专利名称")),
        source_file=source_file,
        source_line=source_line,
    )


def lookup_raw_records_with_rg(raw_path: Path, application_nos: Sequence[str]) -> Dict[str, List[RawPatentRecord]]:
    rg_path = shutil.which("rg")
    if not rg_path:
        raise RuntimeError("未找到 rg，可改用 --raw-lookup-mode scan 或 skip。")

    application_nos = sorted({app for app in application_nos if app})
    if not application_nos:
        return {}

    header_cache: Dict[Path, Tuple[List[str], Dict[str, int]]] = {}
    records_by_app: Dict[str, List[RawPatentRecord]] = defaultdict(list)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
        pattern_file = Path(fh.name)
        for app_no in application_nos:
            fh.write(app_no + "\n")

    try:
        cmd = [
            rg_path,
            "--vimgrep",
            "--no-messages",
            "--text",
            "--fixed-strings",
            "-i",
            "-f",
            str(pattern_file),
            str(raw_path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        for line in completed.stdout.splitlines():
            parts = line.split(":", 3)
            if len(parts) != 4:
                continue
            file_str, line_no_str, _column_str, content = parts
            file_path = Path(file_str)
            if file_path.suffix.lower() != ".csv":
                continue
            cached = header_cache.get(file_path)
            if cached is None:
                header, _encoding = load_csv_header(file_path)
                header_map = {name: idx for idx, name in enumerate(header)}
                header_cache[file_path] = (header, header_map)
                cached = header_cache[file_path]
            header, header_map = cached
            row = parse_csv_row_from_line(content, expected_width=len(header))
            if row is None:
                continue
            app_idx = header_map.get("申请号")
            if app_idx is None:
                continue
            app_no = normalize_application_no(row[app_idx])
            if app_no not in application_nos:
                continue
            values = {header[idx]: row[idx] for idx in range(len(header))}
            records_by_app[app_no].append(
                raw_record_from_mapping(
                    values,
                    source_file=str(file_path),
                    source_line=int(line_no_str),
                )
            )
        return dict(records_by_app)
    finally:
        try:
            pattern_file.unlink()
        except OSError:
            pass


def lookup_raw_records_by_scan(raw_path: Path, application_nos: Sequence[str]) -> Dict[str, List[RawPatentRecord]]:
    application_nos = {app for app in application_nos if app}
    if not application_nos:
        return {}

    records_by_app: Dict[str, List[RawPatentRecord]] = defaultdict(list)
    for csv_path in list_csv_files(raw_path):
        header, encoding = load_csv_header(csv_path)
        header_map = {name: idx for idx, name in enumerate(header)}
        app_idx = header_map.get("申请号")
        if app_idx is None:
            continue
        expected_width = len(header)
        with csv_path.open("r", encoding=encoding, newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for line_no, row in enumerate(reader, start=2):
                if len(row) == expected_width + 1 and row[-1] == "":
                    row = row[:-1]
                elif len(row) != expected_width:
                    continue
                app_no = normalize_application_no(row[app_idx])
                if app_no not in application_nos:
                    continue
                values = {header[idx]: row[idx] for idx in range(expected_width)}
                records_by_app[app_no].append(
                    raw_record_from_mapping(
                        values,
                        source_file=str(csv_path),
                        source_line=line_no,
                    )
                )
    return dict(records_by_app)


def load_raw_records(
    raw_path: Path,
    application_nos: Sequence[str],
    mode: str,
) -> Tuple[Dict[str, List[RawPatentRecord]], str]:
    application_nos = sorted({app for app in application_nos if app})
    if not application_nos or mode == "skip" or not raw_path.exists():
        return {}, "skipped"

    if mode == "auto":
        if shutil.which("rg"):
            return lookup_raw_records_with_rg(raw_path, application_nos), "rg"
        return {}, "skipped"
    if mode == "rg":
        return lookup_raw_records_with_rg(raw_path, application_nos), "rg"
    if mode == "scan":
        return lookup_raw_records_by_scan(raw_path, application_nos), "scan"
    raise ValueError(f"未知的 raw lookup mode: {mode}")


def build_missing_reason(
    *,
    base_reason: str,
    token_hit: bool,
    public_year: int,
    authorized_records: Sequence[AuthorizedPatentRecord],
    raw_records: Sequence[RawPatentRecord],
    raw_lookup_status: str,
) -> str:
    if base_reason:
        return base_reason

    if token_hit:
        return "该年在 stage1 tokens 中找到了该专利，但未进入 stage1 index；通常表示分词后没有保留词或未形成非空向量。"

    if authorized_records:
        same_year_records = [record for record in authorized_records if record.public_year == public_year]
        if same_year_records:
            return "该年在共享发明授权数据中存在该专利，但未进入 stage1 tokens/index；请检查阶段产物是否完整，或是否被同申请号去重。"
        years_text = format_years(record.public_year for record in authorized_records)
        if years_text:
            return f"在共享发明授权数据中找到了该专利，但公开年份为 {years_text}，不是输入的 {public_year}。"
        return "在共享发明授权数据中找到了该专利，但公开年份为空或无法解析。"

    if raw_records:
        same_year_records = [record for record in raw_records if record.public_year == public_year]
        same_year_types = sorted({record.patent_type for record in same_year_records if record.patent_type})
        if same_year_types and AUTHORIZED_PATENT_TYPE not in same_year_types:
            return f"原始数据中找到了该专利，但该年专利类型为 {','.join(same_year_types)}，不是发明授权，被过滤。"
        if any(
            record.patent_type == AUTHORIZED_PATENT_TYPE and record.public_year is None
            for record in raw_records
        ):
            return "原始数据中找到了该专利，且专利类型为发明授权，但公开公告日缺失或无效，构建 exact-time 数据时被过滤。"
        raw_years_text = format_years(record.public_year for record in raw_records)
        if raw_years_text:
            return f"原始数据中找到了该专利，但公开年份为 {raw_years_text}，不是输入的 {public_year}。"
        if any(record.patent_type and record.patent_type != AUTHORIZED_PATENT_TYPE for record in raw_records):
            patent_types = sorted({record.patent_type for record in raw_records if record.patent_type})
            return f"原始数据中找到了该专利，但专利类型为 {','.join(patent_types)}，不是发明授权，被过滤。"
        return "原始数据中找到了该专利，但没有命中输入的公开年份。"

    if raw_lookup_status == "skipped":
        return "未在共享发明授权数据中找到该专利；原始 CSV 回查未执行，常见原因包括：该年没有该专利、不是发明授权、或公开公告日无效。"
    return "在共享发明授权数据和原始数据中都未找到该专利。"


def write_output_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> Path:
    input_path = resolve_project_path(args.input_csv)
    output_path = resolve_project_path(args.output_csv) if args.output_csv else default_output_path(input_path)
    shared_dir = resolve_project_path(args.shared_authorized_parts_dir)
    raw_data_path = resolve_project_path(args.raw_data_path)
    experiment_ids = args.experiment_ids or list(DEFAULT_EXPERIMENT_IDS)

    input_fieldnames, input_rows = read_csv_rows(input_path)
    application_col = resolve_input_column(input_fieldnames, args.application_col, APPLICATION_COL_CANDIDATES, "申请号")
    public_year_col = resolve_input_column(input_fieldnames, args.public_year_col, PUBLIC_YEAR_COL_CANDIDATES, "公开年份")

    tracked_apps = {
        normalize_application_no(row.get(application_col))
        for row in input_rows
        if normalize_application_no(row.get(application_col))
    }
    experiment_lookups: List[ExperimentLookup] = []
    for experiment_id in experiment_ids:
        layout = build_experiment_layout(experiment_id, output_root=args.output_root)
        experiment_lookups.append(
            ExperimentLookup(
                experiment_id=experiment_id,
                stage1_dir=layout.stage1_exact_dir,
                tracked_apps=tracked_apps,
            )
        )

    authorized_records_by_app = load_authorized_records(shared_dir, sorted(tracked_apps)) if shared_dir.exists() else {}

    raw_lookup_candidates: set[str] = set()
    for row in input_rows:
        app_no = normalize_application_no(row.get(application_col))
        if not app_no:
            continue
        if authorized_records_by_app.get(app_no):
            continue
        raw_lookup_candidates.add(app_no)
    raw_records_by_app, raw_lookup_status = load_raw_records(raw_data_path, sorted(raw_lookup_candidates), args.raw_lookup_mode)

    output_rows: List[Dict[str, object]] = []
    output_fieldnames = list(input_fieldnames)
    for experiment_id in experiment_ids:
        output_fieldnames.extend(
            [
                f"{experiment_id}_状态",
                f"{experiment_id}_排名",
                f"{experiment_id}_年内专利数",
                f"{experiment_id}_排名百分比",
                f"{experiment_id}_quantity_q",
                f"{experiment_id}_原因",
            ]
        )

    for row in input_rows:
        output_row: Dict[str, object] = dict(row)
        app_no = normalize_application_no(row.get(application_col))
        public_year = parse_public_year(row.get(public_year_col))

        if not app_no:
            for experiment_id in experiment_ids:
                output_row[f"{experiment_id}_状态"] = "未找到"
                output_row[f"{experiment_id}_原因"] = "输入行缺少申请号。"
            output_rows.append(output_row)
            continue

        if public_year is None:
            for experiment_id in experiment_ids:
                output_row[f"{experiment_id}_状态"] = "未找到"
                output_row[f"{experiment_id}_原因"] = "输入行的公开年份为空或无法解析。"
            output_rows.append(output_row)
            continue

        authorized_records = authorized_records_by_app.get(app_no, [])
        raw_records = raw_records_by_app.get(app_no, [])

        for lookup in experiment_lookups:
            hit, base_reason = lookup.lookup(app_no, public_year)
            prefix = lookup.experiment_id
            if hit is not None:
                output_row[f"{prefix}_状态"] = "找到"
                output_row[f"{prefix}_排名"] = hit.rank
                output_row[f"{prefix}_年内专利数"] = hit.year_total
                output_row[f"{prefix}_排名百分比"] = round(hit.rank_percent, 6)
                output_row[f"{prefix}_quantity_q"] = round(hit.quantity_q, 10)
                output_row[f"{prefix}_原因"] = ""
                continue

            reason = build_missing_reason(
                base_reason=base_reason,
                token_hit=lookup.has_token_hit(app_no, public_year),
                public_year=public_year,
                authorized_records=authorized_records,
                raw_records=raw_records,
                raw_lookup_status=raw_lookup_status,
            )
            output_row[f"{prefix}_状态"] = "未找到"
            output_row[f"{prefix}_排名"] = ""
            output_row[f"{prefix}_年内专利数"] = ""
            output_row[f"{prefix}_排名百分比"] = ""
            output_row[f"{prefix}_quantity_q"] = ""
            output_row[f"{prefix}_原因"] = reason

        output_rows.append(output_row)

    write_output_csv(output_path, output_fieldnames, output_rows, args.output_encoding)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_path = run(args)
    print(f"[done] output_csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
