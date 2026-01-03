import os
import json
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from scipy import sparse
from .config import Config
from .log import get_logger
from .postings import build_postings_for_year
from .pair_compute import compute_pair_contrib
import time


def _resolve_vectors_base(cfg: Config) -> str:
    if getattr(cfg, "use_vectors_filtered_for_bsfs", False):
        return os.path.join(cfg.artifacts_dir, getattr(cfg, "vectors_filtered_dir", "vectors_filtered"))
    return os.path.join(cfg.artifacts_dir, "vectors")


def _years_with_vectors(cfg: Config) -> List[int]:
    base = _resolve_vectors_base(cfg)
    ys = []
    for name in os.listdir(base):
        if name.startswith("year=") and name.endswith(".npz"):
            ys.append(int(name[len("year=") : -len(".npz")]))
    ys.sort()
    return ys


def _pair_list(cfg: Config, years: List[int]) -> List[Tuple[int, int]]:
    win = int(getattr(cfg, "window_size", 0))
    pairs = set()
    for t in years:
        for d in range(1, win + 1):
            x = t
            y = t + d
            if y in years:
                a, b = (x, y) if x < y else (y, x)
                pairs.add((a, b))
        for d in range(1, win + 1):
            y = t - d
            x = t
            if y in years:
                a, b = (x, y) if x < y else (y, x)
                pairs.add((a, b))
    lst = sorted(list(pairs))
    p_json = os.path.join(cfg.artifacts_dir, "pair_list.json")
    try:
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump({"pairs": lst}, f, ensure_ascii=False)
    except Exception:
        pass
    return lst


def compute_bs_fs(cfg: Config) -> None:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level)
    years = _years_with_vectors(cfg)
    if not years:
        logger.warning("未检测到任何向量文件，无法计算BS/FS")
        return
    logger.info(f"检测到向量年份范围: {min(years)} ~ {max(years)} (共{len(years)}个年份)")
    base = _resolve_vectors_base(cfg)
    # Ensure postings exist for all years used in pairs
    logger.info("阶段5: 生成与缓存 pair_list")
    pairs = _pair_list(cfg, years)
    # Ensure postings for all years that appear as target
    target_years = sorted({b for (a, b) in pairs})
    for y in target_years:
        build_postings_for_year(cfg, y, base)
    # Generate missing pair_contrib
    logger.info(f"阶段5: 计算 pair_contrib 对数={len(pairs)}")
    for (x, y) in pairs:
        compute_pair_contrib(cfg, x, y)
    # Summarize per year
    logger.info("阶段5: 汇总年份级 BS/FS")
    for t in years:
        t0 = time.perf_counter()
        M_T = sparse.load_npz(os.path.join(base, f"year={t}.npz"))
        bs = np.zeros(M_T.shape[0], dtype="float64")
        fs = np.zeros(M_T.shape[0], dtype="float64")
        out = os.path.join(cfg.artifacts_dir, "stats", f"bsfs_year={t}.csv")
        if cfg.skip_if_exists and os.path.exists(out):
            logger.info(f"年份={t} BS/FS结果已存在，跳过。")
            continue
        back_years = [y for y in years if t - cfg.window_size <= y <= t - 1]
        forward_years = [y for y in years if t + 1 <= y <= t + cfg.window_size]
        # accumulate BS
        for y in back_years:
            x, y2 = (t, y) if t < y else (y, t)
            pair_p = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir, f"x={x}_y={y2}.npz")
            if not os.path.exists(pair_p):
                logger.warning(f"缺失pair文件: {pair_p}")
                continue
            obj = np.load(pair_p, allow_pickle=True)
            contrib = obj["contrib_x"] if x == t else obj["contrib_y"]
            bs += contrib.astype("float64", copy=False)
        # accumulate FS
        for y in forward_years:
            x, y2 = (t, y) if t < y else (y, t)
            pair_p = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir, f"x={x}_y={y2}.npz")
            if not os.path.exists(pair_p):
                logger.warning(f"缺失pair文件: {pair_p}")
                continue
            obj = np.load(pair_p, allow_pickle=True)
            contrib = obj["contrib_x"] if x == t else obj["contrib_y"]
            fs += contrib.astype("float64", copy=False)
        with open(out, "w", encoding="utf-8") as f:
            f.write("row,BS,FS\n")
            for i in range(M_T.shape[0]):
                f.write(f"{i},{bs[i]},{fs[i]}\n")
        logger.info(f"年份={t} BS/FS输出: {out} 总耗时={(time.perf_counter()-t0):.2f}s")
