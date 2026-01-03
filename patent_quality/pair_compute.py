import os
import json
import numpy as np
from scipy import sparse
from typing import Tuple
from .config import Config
from .log import get_logger
from .postings import load_postings_for_year, build_postings_for_year
import time
from numba import njit


@njit
def _compute_block_numba(
    ptr_y, docs_y, vals_y, max_y,
    Mx_indptr, Mx_indices, Mx_data,
    i0, i1,
    acc, mark, touched,
    thr, use_maxscore,
    contrib_x, contrib_y
):
    """
    Numba accelerated kernel for processing a block of rows.
    Returns (total_touched, max_touched, total_hits)
    """
    total_touched = 0
    max_touched = 0
    total_hits = 0
    
    # We need to manually handle 'touched' array as a stack
    # touched array should be large enough (Ny), passed from outside
    
    for i in range(i0, i1):
        start = Mx_indptr[i]
        end = Mx_indptr[i + 1]
        if start == end:
            continue
            
        # Get row data
        cols = Mx_indices[start:end]
        vals = Mx_data[start:end]
        
        # MaxScore sorting (if enabled)
        # Note: Sorting inside loop can be expensive. 
        # For simplicity in Numba, we might skip sorting or implement a simple one if needed.
        # But even without sorting, basic pruning works. 
        # For strict MaxScore, we need to sort terms by upper bound.
        # Let's do a simple argsort if use_maxscore is True.
        
        remain = 0.0
        if use_maxscore:
            # Calculate upper bounds
            n_terms = len(cols)
            ub = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                ub[k] = vals[k] * max_y[cols[k]]
            
            # Sort indices by ub descending
            # Numba doesn't support np.argsort with 'kind' or complex types well sometimes,
            # but basic argsort on 1D array works.
            # We want descending, so we sort -ub
            order = np.argsort(-ub)
            
            # Reorder cols/vals
            # We need to create new arrays or overwrite
            new_cols = np.empty(n_terms, dtype=np.int32)
            new_vals = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                idx = order[k]
                new_cols[k] = cols[idx]
                new_vals[k] = vals[idx]
            cols = new_cols
            vals = new_vals
            
            # Calculate total remain
            for k in range(n_terms):
                remain += ub[k]

        touched_len = 0
        
        for k in range(len(cols)):
            w = cols[k]
            xw = vals[k]
            
            uw = 0.0
            if use_maxscore:
                uw = xw * max_y[w]
            
            ps = ptr_y[w]
            pe = ptr_y[w + 1]
            
            if ps == pe:
                if use_maxscore:
                    remain -= uw
                continue
                
            for p in range(ps, pe):
                j = docs_y[p]
                
                if mark[j] == 0:
                    mark[j] = 1
                    touched[touched_len] = j
                    touched_len += 1
                elif mark[j] == 2:
                    # dead candidate
                    continue
                
                acc[j] += xw * vals_y[p]
            
            if use_maxscore:
                remain -= uw
                # Pruning pass
                # To avoid too frequent scanning, maybe do it every X terms?
                # But here we do it every term as per standard WAND/MaxScore
                if remain > 0.0: # If remain is 0, we can't prune more based on future
                    # We need to iterate over current touched to prune
                    # This inner loop over touched can be costly if touched is large.
                    # Standard MaxScore maintains a 'pivot' or manages lists differently.
                    # For simple 'term-at-a-time' with pruning:
                    for t_idx in range(touched_len):
                        tj = touched[t_idx]
                        if mark[tj] != 2:
                             if acc[tj] + remain < thr:
                                 mark[tj] = 2 # Mark as dead
        
        # Finalize row
        total_touched += touched_len
        if touched_len > max_touched:
            max_touched = touched_len
            
        hits = 0
        for t_idx in range(touched_len):
            j = touched[t_idx]
            # If mark is 2 (dead), we still need to clear it and acc, but don't count as hit
            # Actually if it's dead, acc[j] < thr is guaranteed (or it wouldn't be dead)
            # But we must check threshold for non-dead ones.
            
            if mark[j] != 0: # 1 or 2
                if mark[j] == 1 and acc[j] >= thr:
                    contrib_x[i] += acc[j]
                    contrib_y[j] += acc[j]
                    hits += 1
                
                # Reset
                acc[j] = 0.0
                mark[j] = 0
        
        total_hits += hits

    return total_touched, max_touched, total_hits


def _resolve_vectors_base(cfg: Config) -> str:
    if getattr(cfg, "use_vectors_filtered_for_bsfs", False):
        return os.path.join(cfg.artifacts_dir, getattr(cfg, "vectors_filtered_dir", "vectors_filtered"))
    return os.path.join(cfg.artifacts_dir, "vectors")


def _pair_path(cfg: Config, x: int, y: int) -> str:
    base = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir)
    os.makedirs(base, exist_ok=True)
    # unify x < y
    a, b = (x, y) if x < y else (y, x)
    return os.path.join(base, f"x={a}_y={b}.npz")


def _pair_tmp_path(cfg: Config, x: int, y: int) -> str:
    base = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir)
    a, b = (x, y) if x < y else (y, x)
    return os.path.join(base, f"x={a}_y={b}.tmp.npz")


def compute_pair_contrib(cfg: Config, x: int, y: int) -> str:
    if x > y:
        panic(f"x={x} > y={y}, which is not allowed")
    logger = get_logger(level=cfg.log_level)
    out_p = _pair_path(cfg, x, y)
    if cfg.skip_if_exists and os.path.exists(out_p):
        return out_p
    t0 = time.perf_counter()
    vectors_base = _resolve_vectors_base(cfg)
    # Ensure postings for y
    p_ptr = os.path.join(cfg.artifacts_dir, cfg.postings_dir, f"year={y}_ptr.npy")
    need_postings = not (os.path.exists(p_ptr))
    if need_postings:
        build_postings_for_year(cfg, y, vectors_base)
    ptr_y, docs_y, vals_y, max_y = load_postings_for_year(cfg, y, mmap=getattr(cfg, "postings_mmap", True))
    Mx = sparse.load_npz(os.path.join(vectors_base, f"year={x}.npz")).tocsr()
    My = sparse.load_npz(os.path.join(vectors_base, f"year={y}.npz")).tocsr()
    Nx = Mx.shape[0]
    Ny = My.shape[0]
    thr = float(getattr(cfg, "similarity_threshold", 0.0))
    block = int(getattr(cfg, "block_size_docs", 10000))
    use_maxscore = bool(getattr(cfg, "enable_maxscore", False))
    contrib_x = np.zeros(Nx, dtype=np.float32)
    contrib_y = np.zeros(Ny, dtype=np.float32)
    acc = np.zeros(Ny, dtype=np.float32)
    mark = np.zeros(Ny, dtype=np.uint8)
    touched = np.empty(Ny, dtype=np.int32)
    total_touched = 0
    max_touched = 0
    total_hits = 0
    # per-block logging
    for i0 in range(0, Nx, block):
        i1 = min(Nx, i0 + block)
        
        # Use Numba kernel
        # We need to pass numpy arrays, not sparse matrix objects directly, 
        # so we pass indptr, indices, data
        
        t_blk, m_blk, h_blk = _compute_block_numba(
            ptr_y, docs_y, vals_y, max_y,
            Mx.indptr, Mx.indices, Mx.data,
            i0, i1,
            acc, mark, touched,
            np.float32(thr), use_maxscore,
            contrib_x, contrib_y
        )
        
        total_touched += t_blk
        if m_blk > max_touched:
            max_touched = m_blk
        total_hits += h_blk
        
        logger.info(f"pair: ({x},{y}) 处理文档 {i0}-{i1} / {Nx}")
    meta = {
        "thr": thr,
        "window_size": int(getattr(cfg, "window_size", 0)),
        "method_version": getattr(cfg, "method_version", "ir_v1"),
        "vectors_dir": vectors_base,
        "Nx": int(Nx),
        "Ny": int(Ny),
        "avg_touched": float(total_touched / max(1, Nx)),
        "max_touched": int(max_touched),
        "hits": int(total_hits),
        "x": int(x),
        "y": int(y),
    }
    tmp_p = _pair_tmp_path(cfg, x, y)
    np.savez(tmp_p, contrib_x=contrib_x, contrib_y=contrib_y, meta_json=json.dumps(meta, ensure_ascii=False))
    os.replace(tmp_p, out_p)
    dt = time.perf_counter() - t0
    logger.info(f"pair: ({x},{y}) 完成 Nx={Nx} Ny={Ny} avg_touched={meta['avg_touched']:.2f} max_touched={max_touched} 命中={total_hits} 耗时={dt:.2f}s 输出={out_p}")
    return out_p

