import os
import numpy as np
from scipy import sparse
from .config import Config
from .log import get_logger
from typing import Tuple


def _postings_paths(cfg: Config, year: int) -> Tuple[str, str, str, str]:
    base = os.path.join(cfg.artifacts_dir, cfg.postings_dir)
    os.makedirs(base, exist_ok=True)
    p_ptr = os.path.join(base, f"year={year}_ptr.npy")
    p_docs = os.path.join(base, f"year={year}_docs.npy")
    p_vals = os.path.join(base, f"year={year}_vals.npy")
    p_max = os.path.join(base, f"year={year}_max.npy")
    return p_ptr, p_docs, p_vals, p_max


def build_postings_for_year(cfg: Config, year: int, vectors_base: str) -> None:
    logger = get_logger(level=cfg.log_level)
    p_ptr, p_docs, p_vals, p_max = _postings_paths(cfg, year)
    if cfg.skip_if_exists and os.path.exists(p_ptr) and os.path.exists(p_docs) and os.path.exists(p_vals) and os.path.exists(p_max):
        logger.info(f"postings: 年份={year} 已存在，跳过构建")
        return
    m_path = os.path.join(vectors_base, f"year={year}.npz")
    M = sparse.load_npz(m_path).tocsc()
    indptr = M.indptr.astype(np.int64, copy=False)
    indices = M.indices.astype(np.int32, copy=False)
    data = M.data.astype(np.float32, copy=False)
    V = M.shape[1]
    maxy = np.zeros(V, dtype=np.float32)
    for c in range(V):
        start = indptr[c]
        end = indptr[c + 1]
        if start < end:
            maxy[c] = np.max(data[start:end]).astype(np.float32)
        else:
            maxy[c] = np.float32(0.0)
    np.save(p_ptr, indptr)
    np.save(p_docs, indices)
    np.save(p_vals, data)
    np.save(p_max, maxy)
    logger.info(f"postings: 年份={year} 构建完成 ptr={indptr.size} docs={indices.size} vals={data.size} V={V}")


def load_postings_for_year(cfg: Config, year: int, mmap: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_ptr, p_docs, p_vals, p_max = _postings_paths(cfg, year)
    if mmap:
        ptr = np.load(p_ptr, mmap_mode="r")
        docs = np.load(p_docs, mmap_mode="r")
        vals = np.load(p_vals, mmap_mode="r")
        maxy = np.load(p_max, mmap_mode="r")
    else:
        ptr = np.load(p_ptr)
        docs = np.load(p_docs)
        vals = np.load(p_vals)
        maxy = np.load(p_max)
    return ptr, docs, vals, maxy

