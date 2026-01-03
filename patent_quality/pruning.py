import os
import json
from typing import Dict, List, Set, Tuple
import numpy as np
from scipy import sparse
import time
from .config import Config
from .log import get_logger
from .nlp import load_stopwords
from .io_utils import load_checkpoint, save_checkpoint


def _load_vocab(cfg: Config) -> Dict[str, int]:
    with open(f"{cfg.artifacts_dir}/vocab/final_vocab.json", "r", encoding="utf-8") as f:
        return json.load(f)["vocab"]


def _load_term_df_year(cfg: Config, year: int) -> Tuple[Dict[str, int], int]:
    with open(f"{cfg.artifacts_dir}/df/term_df_year={year}.json", "r", encoding="utf-8") as f:
        obj = json.load(f)
        return obj["df"], obj["docs"]


def _years_with_vectors(cfg: Config) -> List[int]:
    base = os.path.join(cfg.artifacts_dir, "vectors")
    ys: List[int] = []
    if not os.path.exists(base):
        return ys
    for name in os.listdir(base):
        if name.startswith("year=") and name.endswith(".npz"):
            ys.append(int(name[len("year=") : -len(".npz")]))
    ys.sort()
    return ys


def _zero_columns_in_csr(m: sparse.csr_matrix, col_set: Set[int]) -> None:
    if not col_set or m.nnz == 0:
        return
    indptr = m.indptr
    indices = m.indices
    data = m.data
    for r in range(m.shape[0]):
        start = indptr[r]
        end = indptr[r + 1]
        if start == end:
            continue
        cols = indices[start:end]
        if not cols.size:
            continue
        mask = np.isin(cols, list(col_set))
        if np.any(mask):
            data[start:end][mask] = 0.0
    m.eliminate_zeros()


def _topk_per_row(m: sparse.csr_matrix, k: int) -> None:
    if k <= 0 or m.nnz == 0:
        m.data[:] = 0.0
        m.eliminate_zeros()
        return
    indptr = m.indptr
    indices = m.indices
    data = m.data
    for r in range(m.shape[0]):
        start = indptr[r]
        end = indptr[r + 1]
        nnz = end - start
        if nnz <= k:
            continue
        vals = data[start:end]
        if nnz > 0:
            keep_idx = np.argpartition(vals, nnz - k)[nnz - k :]
            mask = np.ones(nnz, dtype=bool)
            mask[keep_idx] = False
            vals[mask] = 0.0
    m.eliminate_zeros()


def prune_vectors_by_year(cfg: Config) -> None:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level)
    t_stage = time.perf_counter()
    vocab = _load_vocab(cfg)
    inv_vocab: Dict[str, int] = vocab
    years = _years_with_vectors(cfg)
    if not years:
        logger.warning("未检测到任何向量文件，无法执行向量剪枝阶段")
        return
    stop_paths: List[str] = []
    if getattr(cfg, "manual_stopwords_path", None):
        stop_paths = [cfg.manual_stopwords_path]
    stopwords: Set[str] = load_stopwords(stop_paths) if stop_paths else set()
    stop_cols: Set[int] = set()
    for w in stopwords:
        idx = inv_vocab.get(w)
        if idx is not None:
            stop_cols.add(idx)
    logger.info(f"阶段4: 向量剪枝 初始化 停用词列数={len(stop_cols)}")
    ckpt = load_checkpoint(cfg)
    pruned_years: List[int] = ckpt.get("vectors_pruned_years", [])
    out_base = os.path.join(cfg.artifacts_dir, getattr(cfg, "vectors_filtered_dir", "vectors_filtered"))
    os.makedirs(out_base, exist_ok=True)
    for y in years:
        t0 = time.perf_counter()
        out_p = os.path.join(out_base, f"year={y}.npz")
        if cfg.skip_if_exists and os.path.exists(out_p):
            if y not in pruned_years:
                pruned_years.append(y)
                ckpt["vectors_pruned_years"] = pruned_years
                save_checkpoint(cfg, ckpt)
            logger.info(f"年份={y} 剪枝结果已存在，跳过")
            continue
        M = sparse.load_npz(os.path.join(cfg.artifacts_dir, "vectors", f"year={y}.npz"))
        nnz_before_all = M.nnz
        logger.info(f"剪枝前统计 年份={y} 形状={M.shape} 非零={nnz_before_all}")
        if stop_cols:
            nnz_b = M.nnz
            _zero_columns_in_csr(M, stop_cols)
            logger.info(f"Step4.1 年份={y} 删除词数={len(stop_cols)} 非零变更 {nnz_b}->{M.nnz}")
        df_y, N_y = _load_term_df_year(cfg, y)
        removed_ratio_cols: Set[int] = set()
        thr = getattr(cfg, "df_ratio_threshold", 0.20)
        for term, dfv in df_y.items():
            if N_y > 0 and (dfv / N_y) >= thr:
                idx = inv_vocab.get(term)
                if idx is not None:
                    removed_ratio_cols.add(idx)
        if removed_ratio_cols:
            nnz_b = M.nnz
            _zero_columns_in_csr(M, removed_ratio_cols)
            logger.info(f"Step4.2-ratio 年份={y} N_y={N_y} V_y={len([t for t,c in df_y.items() if c>0])} 删除列数={len(removed_ratio_cols)} 非零变更 {nnz_b}->{M.nnz}")
        V_y = len([t for t, c in df_y.items() if c > 0])
        top_pct = getattr(cfg, "top_df_percent", 0.002)
        top_k = int(max(0, np.floor(V_y * top_pct)))
        if top_k > 0:
            pairs: List[Tuple[int, int]] = []
            for term, dfv in df_y.items():
                if dfv <= 0:
                    continue
                idx = inv_vocab.get(term)
                if idx is None or idx in removed_ratio_cols:
                    continue
                pairs.append((idx, dfv))
            if pairs:
                pairs.sort(key=lambda x: x[1], reverse=True)
                to_remove = {idx for idx, _ in pairs[:top_k]}
                nnz_b = M.nnz
                _zero_columns_in_csr(M, to_remove)
                logger.info(f"Step4.2-top 年份={y} 删除列数={len(to_remove)} 非零变更 {nnz_b}->{M.nnz}")
        k_terms = int(getattr(cfg, "topk_terms_per_doc", 30))
        nnz_b = M.nnz
        _topk_per_row(M, k_terms)
        logger.info(f"Step4.3 年份={y} K={k_terms} 行数={M.shape[0]} 非零变更 {nnz_b}->{M.nnz}")
        sparse.save_npz(out_p, M)
        dt = time.perf_counter() - t0
        logger.info(f"年份={y} 剪枝输出={out_p} 耗时={dt:.2f}s")
        if y not in pruned_years:
            pruned_years.append(y)
            ckpt["vectors_pruned_years"] = pruned_years
            save_checkpoint(cfg, ckpt)
    logger.info(f"阶段4完成 总耗时={(time.perf_counter()-t_stage):.2f}s")

