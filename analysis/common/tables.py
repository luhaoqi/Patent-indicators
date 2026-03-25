from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd


def format_scalar(value: Any, *, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def format_dataframe(df: pd.DataFrame, *, digits: int = 3) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda value: format_scalar(value, digits=digits))
    if formatted.index.name is not None:
        formatted.index = formatted.index.map(str)
    return formatted


def export_table(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    tex_path: Optional[Path] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    digits: int = 3,
    index: bool = True,
    escape: bool = False,
    column_format: Optional[str] = None,
) -> Optional[str]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index, encoding="utf-8-sig")

    if tex_path is None:
        return None

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = format_dataframe(df, digits=digits)
    latex = formatted.to_latex(
        index=index,
        escape=escape,
        caption=caption,
        label=label,
        column_format=column_format,
    )
    tex_path.write_text(latex, encoding="utf-8")
    return latex
