from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_CJK_FONTS: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "STHeiti",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
)


def set_chinese_font(
    *,
    candidates: Optional[Iterable[str]] = None,
    logger: Any = None,
) -> Optional[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates or DEFAULT_CJK_FONTS:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name]
            mpl.rcParams["axes.unicode_minus"] = False
            if logger is not None:
                logger.info("使用中文字体: %s", name)
            return name
    if logger is not None:
        logger.warning("未找到可用中文字体，图表可能出现乱码")
    return None


def save_figure(path: Path, *, dpi: int = 300, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()
