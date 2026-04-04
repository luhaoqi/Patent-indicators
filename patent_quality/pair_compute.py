import json
import os
import time
from datetime import date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse

from .config import Config
from .log import get_logger
from .postings import build_postings_for_year, load_postings_for_year


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
    total_touched = 0
    max_touched = 0
    total_hits = 0
    do_maxscore = use_maxscore and (thr > 1e-9)

    for i in range(i0, i1):
        row_idx = i - i0
        start = Mx_indptr[i]
        end = Mx_indptr[i + 1]
        if start == end:
            continue

        cols = Mx_indices[start:end]
        vals = Mx_data[start:end]
        n_terms = len(cols)
        remain = 0.0

        if do_maxscore:
            ub = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                ub[k] = vals[k] * max_y[cols[k]]
            order = np.argsort(-ub)
            new_cols = np.empty(n_terms, dtype=np.int32)
            new_vals = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                idx = order[k]
                new_cols[k] = cols[idx]
                new_vals[k] = vals[idx]
            cols = new_cols
            vals = new_vals
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
                    if k == 0:
                        stats[row_idx, 3] = remain
                    if k == 2:
                        stats[row_idx, 4] = remain
                    if k == 4:
                        stats[row_idx, 5] = remain
                    if k == 9:
                        stats[row_idx, 6] = remain
                continue

            for p in range(ps, pe):
                j = docs_y[p]
                if mark[j] == 0:
                    mark[j] = 1
                    touched[touched_len] = j
                    touched_len += 1
                elif mark[j] == 2:
                    continue
                acc[j] += xw * vals_y[p]
                stats[row_idx, 7] += 1

            if do_maxscore:
                remain -= uw
                if k == 0:
                    stats[row_idx, 3] = remain
                if k == 2:
                    stats[row_idx, 4] = remain
                if k == 4:
                    stats[row_idx, 5] = remain
                if k == 9:
                    stats[row_idx, 6] = remain
                if remain > 0.0:
                    for t_idx in range(touched_len):
                        tj = touched[t_idx]
                        if mark[tj] != 2 and acc[tj] + remain < thr:
                            mark[tj] = 2

        total_touched += touched_len
        if touched_len > max_touched:
            max_touched = touched_len
        stats[row_idx, 0] = touched_len

        hits = 0
        if do_maxscore:
            pruned_cnt = 0
            for t_idx in range(touched_len):
                j = touched[t_idx]
                if mark[j] != 0:
                    if mark[j] == 2:
                        pruned_cnt += 1
                    elif mark[j] == 1 and acc[j] >= thr:
                        contrib_x[i] += acc[j]
                        contrib_y[j] += acc[j]
                        hits += 1
                    acc[j] = 0.0
                    mark[j] = 0
            stats[row_idx, 1] = pruned_cnt
        else:
            for t_idx in range(touched_len):
                j = touched[t_idx]
                if acc[j] >= thr:
                    contrib_x[i] += acc[j]
                    contrib_y[j] += acc[j]
                    hits += 1
                acc[j] = 0.0
                mark[j] = 0

        total_hits += hits

    return total_touched, max_touched, total_hits


@njit
def _lower_bound_segment(arr, left, right, target):
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


@njit
def _compute_block_numba_ranged(
    ptr_y, docs_y, vals_y, max_y,
    Mx_indptr, Mx_indices, Mx_data,
    lo_arr, hi_arr,
    i0, i1,
    acc, mark, touched,
    thr, use_maxscore,
    contrib_x, contrib_y,
    stats
):
    total_touched = 0
    max_touched = 0
    total_hits = 0
    do_maxscore = use_maxscore and (thr > 1e-9)

    for i in range(i0, i1):
        row_idx = i - i0
        lo = lo_arr[i]
        hi = hi_arr[i]
        if hi <= lo:
            continue

        start = Mx_indptr[i]
        end = Mx_indptr[i + 1]
        if start == end:
            continue

        cols = Mx_indices[start:end]
        vals = Mx_data[start:end]
        n_terms = len(cols)
        remain = 0.0

        if do_maxscore:
            ub = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                ub[k] = vals[k] * max_y[cols[k]]
            order = np.argsort(-ub)
            new_cols = np.empty(n_terms, dtype=np.int32)
            new_vals = np.empty(n_terms, dtype=np.float32)
            for k in range(n_terms):
                idx = order[k]
                new_cols[k] = cols[idx]
                new_vals[k] = vals[idx]
            cols = new_cols
            vals = new_vals
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
                    if k == 0:
                        stats[row_idx, 3] = remain
                    if k == 2:
                        stats[row_idx, 4] = remain
                    if k == 4:
                        stats[row_idx, 5] = remain
                    if k == 9:
                        stats[row_idx, 6] = remain
                continue

            start_p = _lower_bound_segment(docs_y, ps, pe, lo)
            end_p = _lower_bound_segment(docs_y, start_p, pe, hi)
            for p in range(start_p, end_p):
                j = docs_y[p]
                if mark[j] == 0:
                    mark[j] = 1
                    touched[touched_len] = j
                    touched_len += 1
                elif mark[j] == 2:
                    continue
                acc[j] += xw * vals_y[p]
                stats[row_idx, 7] += 1

            if do_maxscore:
                remain -= uw
                if k == 0:
                    stats[row_idx, 3] = remain
                if k == 2:
                    stats[row_idx, 4] = remain
                if k == 4:
                    stats[row_idx, 5] = remain
                if k == 9:
                    stats[row_idx, 6] = remain
                if remain > 0.0:
                    for t_idx in range(touched_len):
                        tj = touched[t_idx]
                        if mark[tj] != 2 and acc[tj] + remain < thr:
                            mark[tj] = 2

        total_touched += touched_len
        if touched_len > max_touched:
            max_touched = touched_len
        stats[row_idx, 0] = touched_len

        hits = 0
        if do_maxscore:
            pruned_cnt = 0
            for t_idx in range(touched_len):
                j = touched[t_idx]
                if mark[j] != 0:
                    if mark[j] == 2:
                        pruned_cnt += 1
                    elif mark[j] == 1 and acc[j] >= thr:
                        contrib_x[i] += acc[j]
                        contrib_y[j] += acc[j]
                        hits += 1
                    acc[j] = 0.0
                    mark[j] = 0
            stats[row_idx, 1] = pruned_cnt
        else:
            for t_idx in range(touched_len):
                j = touched[t_idx]
                if acc[j] >= thr:
                    contrib_x[i] += acc[j]
                    contrib_y[j] += acc[j]
                    hits += 1
                acc[j] = 0.0
                mark[j] = 0

        total_hits += hits

    return total_touched, max_touched, total_hits


_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_DATE_ORD_CACHE: Dict[Tuple[str, int, str], np.ndarray] = {}


def _resolve_vectors_base(cfg: Config) -> str:
    if getattr(cfg, "use_vectors_filtered_for_bsfs", False):
        return os.path.join(cfg.artifacts_dir, getattr(cfg, "vectors_filtered_dir", "vectors_filtered"))
    return os.path.join(cfg.artifacts_dir, "vectors")


def _pair_path(cfg: Config, x: int, y: int) -> str:
    base = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir)
    os.makedirs(base, exist_ok=True)
    a, b = (x, y) if x < y else (y, x)
    return os.path.join(base, f"x={a}_y={b}.npz")


def _pair_tmp_path(cfg: Config, x: int, y: int) -> str:
    base = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir)
    a, b = (x, y) if x < y else (y, x)
    return os.path.join(base, f"x={a}_y={b}.tmp.npz")


def same_year_path(cfg: Config, year: int) -> str:
    base = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"same_year={year}.npz")


def _same_year_tmp_path(cfg: Config, year: int) -> str:
    base = os.path.join(cfg.artifacts_dir, cfg.pair_contrib_dir)
    return os.path.join(base, f"same_year={year}.tmp.npz")


def _date_from_ord(day_ord: int) -> date:
    return date.fromordinal(_EPOCH_ORDINAL + int(day_ord))


def _day_ord_from_date(value: date) -> int:
    return value.toordinal() - _EPOCH_ORDINAL


def _add_years(base: date, years: int) -> date:
    try:
        return base.replace(year=base.year + years)
    except ValueError:
        return base.replace(month=2, day=28, year=base.year + years)


def _load_date_ords(cfg: Config, year: int) -> np.ndarray:
    cache_key = (os.path.abspath(cfg.artifacts_dir), year, cfg.public_date_ord_col)
    cached = _DATE_ORD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index_path = os.path.join(cfg.artifacts_dir, "index", f"year={year}.csv")
    ords = pd.read_csv(index_path, usecols=[cfg.public_date_ord_col])[cfg.public_date_ord_col].to_numpy(dtype=np.int32, copy=True)
    _DATE_ORD_CACHE[cache_key] = ords
    return ords


def _build_same_year_hi(date_ords: np.ndarray) -> np.ndarray:
    hi = np.empty(len(date_ords), dtype=np.int32)
    i = 0
    while i < len(date_ords):
        j = i + 1
        while j < len(date_ords) and date_ords[j] == date_ords[i]:
            j += 1
        hi[i:j] = i
        i = j
    return hi


def _build_boundary_hi(source_ords: np.ndarray, target_ords: np.ndarray, years: int) -> np.ndarray:
    hi = np.empty(len(source_ords), dtype=np.int32)
    for i, value in enumerate(source_ords):
        end_ord = _day_ord_from_date(_add_years(_date_from_ord(int(value)), years))
        hi[i] = int(np.searchsorted(target_ords, end_ord, side="right"))
    return hi


def _log_input_stats(logger, x: int, Nx: int, row_lens: np.ndarray) -> None:
    avg_len = float(np.mean(row_lens)) if Nx > 0 else 0.0
    p50_len = float(np.percentile(row_lens, 50)) if Nx > 0 else 0.0
    p90_len = float(np.percentile(row_lens, 90)) if Nx > 0 else 0.0
    p99_len = float(np.percentile(row_lens, 99)) if Nx > 0 else 0.0
    max_len = int(np.max(row_lens)) if Nx > 0 else 0
    logger.info(
        "Query Matrix (%s) Stats: Rows=%s, AvgTerms=%.2f, P50=%.0f, P90=%.0f, P99=%.0f, Max=%s",
        x,
        Nx,
        avg_len,
        p50_len,
        p90_len,
        p99_len,
        max_len,
    )


def _log_diag_stats(logger, x: int, y: int, all_stats: np.ndarray, do_maxscore: bool) -> None:
    avg_touched = np.mean(all_stats[:, 0])
    p90_touched = np.percentile(all_stats[:, 0], 90)
    max_touched_stat = np.max(all_stats[:, 0])
    total_ops = np.sum(all_stats[:, 7])
    avg_ops = np.mean(all_stats[:, 7])
    logger.info("Diag (%s,%s): Touched Avg=%.1f P90=%.1f Max=%.0f", x, y, avg_touched, p90_touched, max_touched_stat)
    if do_maxscore:
        touched_lens = all_stats[:, 0]
        pruned_cnts = all_stats[:, 1]
        ratios = np.divide(pruned_cnts, touched_lens, out=np.zeros_like(pruned_cnts), where=touched_lens != 0)
        avg_pruned_ratio = np.mean(ratios)
        p90_pruned_ratio = np.percentile(ratios, 90)
        avg_remain_init = np.mean(all_stats[:, 2])
        avg_remain_k1 = np.mean(all_stats[:, 3])
        avg_remain_k3 = np.mean(all_stats[:, 4])
        avg_remain_k5 = np.mean(all_stats[:, 5])
        avg_remain_k10 = np.mean(all_stats[:, 6])
        logger.info("Diag (%s,%s): PrunedRatio Avg=%.4f P90=%.4f", x, y, avg_pruned_ratio, p90_pruned_ratio)
        logger.info(
            "Diag (%s,%s): Remain Init=%.4f K1=%.4f K3=%.4f K5=%.4f K10=%.4f",
            x,
            y,
            avg_remain_init,
            avg_remain_k1,
            avg_remain_k3,
            avg_remain_k5,
            avg_remain_k10,
        )
    logger.info("Diag (%s,%s): PostingOps Total=%.0f Avg=%.1f", x, y, total_ops, avg_ops)


def _compute_contrib_arrays(
    cfg: Config,
    x: int,
    y: int,
    *,
    lo_arr: Optional[np.ndarray] = None,
    hi_arr: Optional[np.ndarray] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    logger = get_logger(level=cfg.log_level)
    t0 = time.perf_counter()
    vectors_base = _resolve_vectors_base(cfg)
    p_ptr = os.path.join(cfg.artifacts_dir, cfg.postings_dir, f"year={y}_ptr.npy")
    if not os.path.exists(p_ptr):
        build_postings_for_year(cfg, y, vectors_base)

    ptr_y, docs_y, vals_y, max_y = load_postings_for_year(cfg, y, mmap=getattr(cfg, "postings_mmap", True))
    Mx = sparse.load_npz(os.path.join(vectors_base, f"year={x}.npz")).tocsr()
    My = sparse.load_npz(os.path.join(vectors_base, f"year={y}.npz")).tocsr()
    Nx = Mx.shape[0]
    Ny = My.shape[0]
    row_lens = np.diff(Mx.indptr)
    _log_input_stats(logger, x, Nx, row_lens)

    if lo_arr is None:
        lo_arr = np.zeros(Nx, dtype=np.int32)
    else:
        lo_arr = np.asarray(lo_arr, dtype=np.int32)
    if hi_arr is None:
        hi_arr = np.full(Nx, Ny, dtype=np.int32)
    else:
        hi_arr = np.asarray(hi_arr, dtype=np.int32)
    if len(lo_arr) != Nx or len(hi_arr) != Nx:
        raise ValueError(f"非法范围长度: Nx={Nx}, len(lo)={len(lo_arr)}, len(hi)={len(hi_arr)}")

    thr = float(getattr(cfg, "similarity_threshold", 0.0))
    block = int(getattr(cfg, "block_size_docs", 10000))
    use_maxscore = bool(getattr(cfg, "enable_maxscore", False))
    do_maxscore = use_maxscore and (thr > 1e-9)
    use_ranges = bool(np.any(lo_arr != 0) or np.any(hi_arr != Ny))

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

    for i0 in range(0, Nx, block):
        i1 = min(Nx, i0 + block)
        block_stats = np.zeros((i1 - i0, 8), dtype=np.float32)
        t_k_start = time.perf_counter()
        if use_ranges:
            t_blk, m_blk, h_blk = _compute_block_numba_ranged(
                ptr_y,
                docs_y,
                vals_y,
                max_y,
                Mx.indptr,
                Mx.indices,
                Mx.data,
                lo_arr,
                hi_arr,
                i0,
                i1,
                acc,
                mark,
                touched,
                np.float32(thr),
                use_maxscore,
                contrib_x,
                contrib_y,
                block_stats,
            )
        else:
            t_blk, m_blk, h_blk = _compute_block_numba(
                ptr_y,
                docs_y,
                vals_y,
                max_y,
                Mx.indptr,
                Mx.indices,
                Mx.data,
                i0,
                i1,
                acc,
                mark,
                touched,
                np.float32(thr),
                use_maxscore,
                contrib_x,
                contrib_y,
                block_stats,
            )
        kernel_time_total += time.perf_counter() - t_k_start
        all_stats_list.append(block_stats)
        total_touched += t_blk
        if m_blk > max_touched:
            max_touched = m_blk
        total_hits += h_blk
        if (i0 // block) % 5 == 0:
            logger.info("pair: (%s,%s) 处理文档 %s-%s / %s", x, y, i0, i1, Nx)

    if all_stats_list:
        _log_diag_stats(logger, x, y, np.vstack(all_stats_list), do_maxscore)

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
        "kernel_time_sec": float(kernel_time_total),
        "total_time_sec": float(time.perf_counter() - t0),
        "range_mode": "ranged" if use_ranges else "full",
    }
    if meta_extra:
        meta.update(meta_extra)
    logger.info(
        "pair: (%s,%s) 完成 Nx=%s Ny=%s KernelTime=%.2fs TotalTime=%.2fs MaxScore=%s RangeMode=%s",
        x,
        y,
        Nx,
        Ny,
        kernel_time_total,
        time.perf_counter() - t0,
        use_maxscore,
        meta["range_mode"],
    )
    return contrib_x, contrib_y, meta


def compute_pair_contrib(cfg: Config, x: int, y: int) -> str:
    if x > y:
        x, y = y, x
    if cfg.exact_date and x == y:
        raise ValueError("exact 模式的同年贡献请调用 compute_same_year_contrib")

    out_p = _pair_path(cfg, x, y)
    if cfg.skip_if_exists and os.path.exists(out_p):
        return out_p

    diff = y - x
    lo_arr: Optional[np.ndarray] = None
    hi_arr: Optional[np.ndarray] = None
    meta_extra: Dict[str, Any] = {"exact_date": bool(cfg.exact_date), "year_diff": int(diff)}
    if cfg.exact_date and diff == int(cfg.window_size):
        source_ords = _load_date_ords(cfg, x)
        target_ords = _load_date_ords(cfg, y)
        hi_arr = _build_boundary_hi(source_ords, target_ords, int(cfg.window_size))
        meta_extra["range_mode"] = "forward_boundary"

    contrib_x, contrib_y, meta = _compute_contrib_arrays(
        cfg,
        x,
        y,
        lo_arr=lo_arr,
        hi_arr=hi_arr,
        meta_extra=meta_extra,
    )
    tmp_p = _pair_tmp_path(cfg, x, y)
    np.savez(tmp_p, contrib_x=contrib_x, contrib_y=contrib_y, meta_json=json.dumps(meta, ensure_ascii=False))
    os.replace(tmp_p, out_p)
    return out_p


def compute_same_year_contrib(cfg: Config, year: int) -> str:
    out_p = same_year_path(cfg, year)
    if cfg.skip_if_exists and os.path.exists(out_p):
        return out_p

    date_ords = _load_date_ords(cfg, year)
    hi_arr = _build_same_year_hi(date_ords)
    bs_same, fs_same, meta = _compute_contrib_arrays(
        cfg,
        year,
        year,
        hi_arr=hi_arr,
        meta_extra={"exact_date": True, "same_year": True, "year_diff": 0, "range_mode": "same_year_backward"},
    )
    tmp_p = _same_year_tmp_path(cfg, year)
    np.savez(tmp_p, bs_same=bs_same, fs_same=fs_same, meta_json=json.dumps(meta, ensure_ascii=False))
    os.replace(tmp_p, out_p)
    return out_p
