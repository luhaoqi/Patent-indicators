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
    contrib_x, contrib_y,
    stats
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
        row_idx = i - i0
        start = Mx_indptr[i]
        end = Mx_indptr[i + 1]
        if start == end:
            continue
            
        # Get row data
        cols = Mx_indices[start:end]
        vals = Mx_data[start:end]
        
        n_terms = len(cols)
        remain = 0.0
        
        # Optimize: If threshold is 0, MaxScore pruning is impossible (cannot prune positive scores < 0).
        # Force disable to avoid sorting/scanning overhead.
        do_maxscore = use_maxscore and (thr > 1e-9)

        if do_maxscore:
            # Calculate upper bounds
            ub = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                ub[k] = vals[k] * max_y[cols[k]]
            
            # Sort indices by ub descending
            # We want descending, so we sort -ub
            order = np.argsort(-ub)
            
            # Reorder cols/vals
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
            
            stats[row_idx, 2] = remain

        touched_len = 0
        
        for k in range(len(cols)):
            w = cols[k]
            xw = vals[k]
            
            uw = 0.0
            if do_maxscore:
                uw = xw * max_y[w]
            
            ps = ptr_y[w]
            pe = ptr_y[w + 1]
            
            if ps == pe:
                if do_maxscore:
                    remain -= uw
                    if k == 0: stats[row_idx, 3] = remain
                    if k == 2: stats[row_idx, 4] = remain
                    if k == 4: stats[row_idx, 5] = remain
                    if k == 9: stats[row_idx, 6] = remain
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
                stats[row_idx, 7] += 1
            
            if do_maxscore:
                remain -= uw
                if k == 0: stats[row_idx, 3] = remain
                if k == 2: stats[row_idx, 4] = remain
                if k == 4: stats[row_idx, 5] = remain
                if k == 9: stats[row_idx, 6] = remain
                
                # Pruning pass
                if remain > 0.0:
                    for t_idx in range(touched_len):
                        tj = touched[t_idx]
                        if mark[tj] != 2:
                             if acc[tj] + remain < thr:
                                 mark[tj] = 2 # Mark as dead
        
        # Finalize row
        total_touched += touched_len
        if touched_len > max_touched:
            max_touched = touched_len
            
        stats[row_idx, 0] = touched_len
        
        pruned_cnt = 0
        hits = 0
        for t_idx in range(touched_len):
            j = touched[t_idx]
            if mark[j] != 0: # 1 or 2
                if mark[j] == 2:
                    pruned_cnt += 1
                elif mark[j] == 1 and acc[j] >= thr:
                    contrib_x[i] += acc[j]
                    contrib_y[j] += acc[j]
                    hits += 1
                
                # Reset
                acc[j] = 0.0
                mark[j] = 0
        
        stats[row_idx, 1] = pruned_cnt
        
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
        x, y = y, x
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
    
    # --- Profiling: Analyze Input Sparsity ---
    row_lens = np.diff(Mx.indptr)
    avg_len = float(np.mean(row_lens)) if Nx > 0 else 0.0
    p50_len = float(np.percentile(row_lens, 50)) if Nx > 0 else 0.0
    p90_len = float(np.percentile(row_lens, 90)) if Nx > 0 else 0.0
    p99_len = float(np.percentile(row_lens, 99)) if Nx > 0 else 0.0
    max_len = int(np.max(row_lens)) if Nx > 0 else 0
    
    logger.info(f"Query Matrix ({x}) Stats: Rows={Nx}, AvgTerms={avg_len:.2f}, P50={p50_len}, P90={p90_len}, P99={p99_len}, Max={max_len}")
    # -----------------------------------------

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
    
    kernel_time_total = 0.0
    all_stats_list = []
    
    # per-block logging
    for i0 in range(0, Nx, block):
        i1 = min(Nx, i0 + block)
        
        # Stats buffer for this block
        block_stats = np.zeros((i1 - i0, 8), dtype=np.float32)

        t_k_start = time.perf_counter()
        # Use Numba kernel
        t_blk, m_blk, h_blk = _compute_block_numba(
            ptr_y, docs_y, vals_y, max_y,
            Mx.indptr, Mx.indices, Mx.data,
            i0, i1,
            acc, mark, touched,
            np.float32(thr), use_maxscore,
            contrib_x, contrib_y,
            block_stats
        )
        kernel_time_total += (time.perf_counter() - t_k_start)
        
        all_stats_list.append(block_stats)

        total_touched += t_blk
        if m_blk > max_touched:
            max_touched = m_blk
        total_hits += h_blk
        
        if (i0 // block) % 5 == 0: # reduce log noise
             logger.info(f"pair: ({x},{y}) 处理文档 {i0}-{i1} / {Nx}")
    
    if len(all_stats_list) > 0:
        all_stats = np.vstack(all_stats_list)
        # 0: touched_len, 1: pruned_cnt, 2: remain_init, 3: k1, 4: k3, 5: k5, 6: k10, 7: ops
        
        avg_touched = np.mean(all_stats[:, 0])
        p90_touched = np.percentile(all_stats[:, 0], 90)
        max_touched_stat = np.max(all_stats[:, 0])
        
        touched_lens = all_stats[:, 0]
        pruned_cnts = all_stats[:, 1]
        ratios = np.divide(pruned_cnts, touched_lens, out=np.zeros_like(pruned_cnts), where=touched_lens!=0)
        avg_pruned_ratio = np.mean(ratios)
        p90_pruned_ratio = np.percentile(ratios, 90)
        
        avg_remain_init = np.mean(all_stats[:, 2])
        avg_remain_k1 = np.mean(all_stats[:, 3])
        avg_remain_k3 = np.mean(all_stats[:, 4])
        avg_remain_k5 = np.mean(all_stats[:, 5])
        avg_remain_k10 = np.mean(all_stats[:, 6])
        
        total_ops = np.sum(all_stats[:, 7])
        avg_ops = np.mean(all_stats[:, 7])
        
        logger.info(f"Diag ({x},{y}): Touched Avg={avg_touched:.1f} P90={p90_touched:.1f} Max={max_touched_stat:.0f}")
        logger.info(f"Diag ({x},{y}): PrunedRatio Avg={avg_pruned_ratio:.4f} P90={p90_pruned_ratio:.4f}")
        logger.info(f"Diag ({x},{y}): Remain Init={avg_remain_init:.4f} K1={avg_remain_k1:.4f} K3={avg_remain_k3:.4f} K5={avg_remain_k5:.4f} K10={avg_remain_k10:.4f}")
        logger.info(f"Diag ({x},{y}): PostingOps Total={total_ops:.0f} Avg={avg_ops:.1f}")
             
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
    logger.info(f"pair: ({x},{y}) 完成 Nx={Nx} Ny={Ny} AvgLen={avg_len:.2f} KernelTime={kernel_time_total:.2f}s TotalTime={dt:.2f}s MaxScore={use_maxscore}")
    return out_p
