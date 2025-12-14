import os
import numpy as np
from typing import List
from tqdm import tqdm
from scipy import sparse
from .config import Config
from .log import get_logger
import time


def _years_with_vectors(cfg: Config) -> List[int]:
    base = os.path.join(cfg.artifacts_dir, "vectors")
    ys = []
    for name in os.listdir(base):
        if name.startswith("year=") and name.endswith(".npz"):
            ys.append(int(name[len("year=") : -len(".npz")]))
    ys.sort()
    return ys


def _sum_rows_after_threshold(m: sparse.csr_matrix, thr: float) -> np.ndarray:
    m = m.tocsr()
    if m.nnz:
        data = m.data
        data[data < thr] = 0.0
        m.eliminate_zeros()
    return np.array(m.sum(axis=1)).ravel()


def compute_bs_fs(cfg: Config) -> None:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level)
    years = _years_with_vectors(cfg)
    if not years:
        logger.warning("未检测到任何向量文件，无法计算BS/FS")
        return
    logger.info(f"检测到向量年份范围: {min(years)} ~ {max(years)} (共{len(years)}个年份)")
    
    for t in years:
        t0 = time.perf_counter()
        M_T = sparse.load_npz(os.path.join(cfg.artifacts_dir, "vectors", f"year={t}.npz"))
        bs = np.zeros(M_T.shape[0], dtype="float64")
        fs = np.zeros(M_T.shape[0], dtype="float64")
        back_years = [y for y in years if t - cfg.window_size <= y <= t - 1]
        forward_years = [y for y in years if t + 1 <= y <= t + cfg.window_size]
        logger.info(f"开始计算BS/FS 年份={t} 回看窗口={back_years} 前看窗口={forward_years}")
        
        out = os.path.join(cfg.artifacts_dir, "stats", f"bsfs_year={t}.csv")
        if cfg.skip_if_exists and os.path.exists(out):
            logger.info(f"年份={t} BS/FS结果已存在，跳过。")
            continue

        if not back_years:
            logger.warning(f"年份={t} 无回看年份 (BS=0)")
        for y in back_years:
            t1 = time.perf_counter()
            M_Y = sparse.load_npz(os.path.join(cfg.artifacts_dir, "vectors", f"year={y}.npz"))
            logger.info(f"BS计算: 加载年份={y} 向量 维度={M_Y.shape}")
            S = M_T.dot(M_Y.T)
            bs += _sum_rows_after_threshold(S, cfg.similarity_threshold)
            logger.info(f"BS乘法 T={t}×Y={y} 结果非零={S.nnz} 耗时={(time.perf_counter()-t1):.2f}s")
        
        if not forward_years:
            logger.warning(f"年份={t} 无前看年份 (FS=0)")
        for y in forward_years:
            t1 = time.perf_counter()
            M_Y = sparse.load_npz(os.path.join(cfg.artifacts_dir, "vectors", f"year={y}.npz"))
            logger.info(f"FS计算: 加载年份={y} 向量 维度={M_Y.shape}")
            S = M_T.dot(M_Y.T)
            fs += _sum_rows_after_threshold(S, cfg.similarity_threshold)
            logger.info(f"FS乘法 T={t}×Y={y} 结果非零={S.nnz} 耗时={(time.perf_counter()-t1):.2f}s")
        out = os.path.join(cfg.artifacts_dir, "stats", f"bsfs_year={t}.csv")
        with open(out, "w", encoding="utf-8") as f:
            f.write("row,BS,FS\n")
            for i in range(M_T.shape[0]):
                f.write(f"{i},{bs[i]},{fs[i]}\n")
        logger.info(f"年份={t} BS/FS输出: {out} 总耗时={(time.perf_counter()-t0):.2f}s")
