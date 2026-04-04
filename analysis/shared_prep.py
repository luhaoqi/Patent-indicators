from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd
import polars as pl

from common.analysis import (
    SPECIAL_UCC_COL,
    build_firm_year_special_panel,
    compute_special_ucc_set,
    load_special_panel,
)
from common.io import build_logger, write_json
from common.paths import build_shared_paths, repo_relative


REQUIRED_SHARED_FIELDS = {
    "patent_master": ["申请号"],
    "firm_year_special_labels": ["统一社会信用代码", "申请年份", "is_special_year"],
    "special_ucc_set": ["统一社会信用代码"],
    "ucc_exploded": ["Stkid", "ShortName", "year", "UCC"],
    "financial_annual_clean": ["stkcd", "year", "Accper"],
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_shared_metadata(
    metadata_path: Path,
    *,
    inputs: Dict[str, Optional[Path]],
    outputs: Dict[str, Path],
    row_counts: Dict[str, int],
    key_fields: Dict[str, list[str]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "generated_at": _timestamp(),
        "inputs": {key: repo_relative(value) if value is not None else None for key, value in inputs.items()},
        "outputs": {key: repo_relative(value) for key, value in outputs.items()},
        "row_counts": row_counts,
        "key_fields": key_fields,
    }
    if extra:
        payload.update(extra)
    write_json(metadata_path, payload)


def clean_financial_annual_panel(
    financial_df: pd.DataFrame,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> pd.DataFrame:
    need_cols = ["stkcd", "Accper", "roa", "roe", "tq", "asset", "liability", "finlev", "gassets", "soe"]
    missing = [column for column in need_cols if column not in financial_df.columns]
    if missing:
        raise KeyError(f"财务数据缺少列: {missing}")

    fin = financial_df.copy()
    fin["Accper"] = pd.to_datetime(fin["Accper"], errors="coerce")
    fin["year"] = fin["Accper"].dt.year
    fin["month"] = fin["Accper"].dt.month
    fin["day"] = fin["Accper"].dt.day
    fin = fin[(fin["month"] == 12) & (fin["day"] == 31)].copy()
    if year_min is not None:
        fin = fin[fin["year"] >= int(year_min)].copy()
    if year_max is not None:
        fin = fin[fin["year"] <= int(year_max)].copy()
    fin["stkcd"] = (
        pd.to_numeric(fin["stkcd"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(6)
    )
    fin = fin[fin["stkcd"].notna() & fin["year"].notna()].copy()
    fin = fin.sort_values(["stkcd", "Accper"]).drop_duplicates(["stkcd", "year"], keep="last")
    return fin.reset_index(drop=True)


def build_special_firm_labels(
    *,
    special_list_path: Path,
    shared_root: str = "outputs/shared",
) -> Dict[str, Path]:
    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()
    logger = build_logger(
        "build_special_firm_labels",
        shared_paths.logs_dir / "build_special_firm_labels.log",
    )

    logger.info("读取特殊企业名单: %s", repo_relative(special_list_path))
    special_df = load_special_panel(pd.read_stata(special_list_path))
    firm_year_special = build_firm_year_special_panel(special_df)
    special_uccs = sorted(compute_special_ucc_set(special_df))

    cleaned_path = shared_paths.special_firm_labels_dir / "special_panel_clean.parquet"
    firm_year_path = shared_paths.special_firm_labels_dir / "firm_year_special_labels.parquet"
    special_ucc_path = shared_paths.special_firm_labels_dir / "special_ucc_set.parquet"
    metadata_path = shared_paths.special_firm_labels_dir / "metadata.json"

    special_df.to_parquet(cleaned_path, index=False)
    firm_year_special.to_parquet(firm_year_path, index=False)
    pd.DataFrame({SPECIAL_UCC_COL: special_uccs}).to_parquet(special_ucc_path, index=False)
    logger.info("特殊企业共享产物已写出: %s", repo_relative(firm_year_path))

    _write_shared_metadata(
        metadata_path,
        inputs={"special_list_path": special_list_path},
        outputs={
            "special_panel_clean": cleaned_path,
            "firm_year_special_labels": firm_year_path,
            "special_ucc_set": special_ucc_path,
        },
        row_counts={
            "special_panel_clean": int(len(special_df)),
            "firm_year_special_labels": int(len(firm_year_special)),
            "special_ucc_set": int(len(special_uccs)),
        },
        key_fields={
            "special_panel_clean": list(special_df.columns),
            "firm_year_special_labels": REQUIRED_SHARED_FIELDS["firm_year_special_labels"],
            "special_ucc_set": REQUIRED_SHARED_FIELDS["special_ucc_set"],
        },
    )
    return {
        "special_panel_clean_path": cleaned_path,
        "firm_year_special_labels_path": firm_year_path,
        "special_ucc_set_path": special_ucc_path,
        "metadata_path": metadata_path,
    }


def build_financial_annual_panel(
    *,
    financial_data_path: Path,
    shared_root: str = "outputs/shared",
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Path]:
    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()
    logger = build_logger(
        "build_financial_annual_panel",
        shared_paths.logs_dir / "build_financial_annual_panel.log",
    )

    logger.info("读取原始财务数据: %s", repo_relative(financial_data_path))
    financial_df = pd.read_stata(financial_data_path)
    clean_df = clean_financial_annual_panel(financial_df, year_min=year_min, year_max=year_max)

    output_path = shared_paths.financial_panel_dir / "financial_annual_clean.parquet"
    metadata_path = shared_paths.financial_panel_dir / "metadata.json"
    clean_df.to_parquet(output_path, index=False)
    logger.info("财务年报面板已写出: %s", repo_relative(output_path))

    _write_shared_metadata(
        metadata_path,
        inputs={"financial_data_path": financial_data_path},
        outputs={"financial_annual_clean": output_path},
        row_counts={"financial_annual_clean": int(len(clean_df))},
        key_fields={"financial_annual_clean": REQUIRED_SHARED_FIELDS["financial_annual_clean"]},
        extra={"year_min": year_min, "year_max": year_max},
    )
    return {
        "financial_annual_clean_path": output_path,
        "metadata_path": metadata_path,
    }


def verify_shared_prep(
    *,
    shared_root: str = "outputs/shared",
) -> Dict[str, Any]:
    shared_paths = build_shared_paths(shared_root)
    shared_paths.ensure_dirs()

    checks: Dict[str, Dict[str, Any]] = {}

    datasets = {
        "patent_master": shared_paths.patent_master_dir / "patent_master.parquet",
        "firm_year_special_labels": shared_paths.special_firm_labels_dir / "firm_year_special_labels.parquet",
        "special_ucc_set": shared_paths.special_firm_labels_dir / "special_ucc_set.parquet",
        "ucc_exploded": shared_paths.ucc_mapping_dir / "ucc_exploded.parquet",
        "financial_annual_clean": shared_paths.financial_panel_dir / "financial_annual_clean.parquet",
    }

    for name, path in datasets.items():
        exists = path.exists()
        record: Dict[str, Any] = {"path": repo_relative(path), "exists": exists}
        if exists:
            lf = pl.scan_parquet(str(path))
            columns = lf.collect_schema().names()
            required = REQUIRED_SHARED_FIELDS[name]
            missing = [column for column in required if column not in columns]
            record.update(
                {
                    "rows": int(lf.select(pl.len()).collect().item()),
                    "columns": columns,
                    "missing_required_fields": missing,
                }
            )
            if not missing:
                if name == "patent_master":
                    record["primary_key_unique"] = bool(
                        lf.select((pl.col("申请号").n_unique() == pl.len()).alias("is_unique")).collect().item()
                    )
                elif name == "firm_year_special_labels":
                    record["primary_key_unique"] = bool(
                        lf.select(
                            (pl.struct(["统一社会信用代码", "申请年份"]).n_unique() == pl.len()).alias("is_unique")
                        ).collect().item()
                    )
                elif name == "ucc_exploded":
                    record["primary_key_unique"] = bool(
                        lf.select((pl.struct(["Stkid", "year", "UCC"]).n_unique() == pl.len()).alias("is_unique")).collect().item()
                    )
                elif name == "financial_annual_clean":
                    record["primary_key_unique"] = bool(
                        lf.select((pl.struct(["stkcd", "year"]).n_unique() == pl.len()).alias("is_unique")).collect().item()
                    )
        checks[name] = record

    raw_authorized_dir = shared_paths.raw_patent_authorized_parts_dir
    raw_authorized_parts = []
    if raw_authorized_dir.exists():
        raw_authorized_parts = sorted(
            path for path in raw_authorized_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".parquet"
        )
    checks["raw_patent_authorized_parts"] = {
        "path": repo_relative(raw_authorized_dir),
        "exists": raw_authorized_dir.exists(),
        "metadata_exists": (raw_authorized_dir / "metadata.json").exists(),
        "parquet_parts": len(raw_authorized_parts),
    }

    summary = {
        "shared_root": repo_relative(shared_paths.root),
        "checks": checks,
        "generated_at": _timestamp(),
    }
    write_json(shared_paths.metadata_dir / "verify_shared_prep.json", summary)
    return summary
