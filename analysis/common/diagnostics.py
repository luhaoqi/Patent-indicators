from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse


YEAR_PATTERN = re.compile(r"year=(\d+)")


def _log_info(logger: Optional[logging.Logger], message: str, *args: object) -> None:
    if logger is not None:
        logger.info(message, *args)


def _extract_year(path: Path) -> int:
    match = YEAR_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"无法从文件名提取年份: {path.name}")
    return int(match.group(1))


def _load_vocab(vocab_path: Path) -> Dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["vocab"] if "vocab" in data else data


def compute_avg_vocab_usage(stage1_dir: Path, *, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    vectors_dir = stage1_dir / "vectors"
    rows: List[Dict[str, float]] = []
    vector_paths = sorted(vectors_dir.glob("year=*.npz"), key=_extract_year)
    _log_info(logger, "开始计算 avg_vocab_usage，共 %s 个年份矩阵", len(vector_paths))
    for index, path in enumerate(vector_paths, start=1):
        year = _extract_year(path)
        _log_info(logger, "avg_vocab_usage [%s/%s] 读取年份 %s: %s", index, len(vector_paths), year, path.name)
        matrix = sparse.load_npz(path)
        n_docs = int(matrix.shape[0])
        avg_nonzero_terms = float(np.mean(matrix.getnnz(axis=1))) if n_docs else 0.0
        rows.append(
            {
                "year": year,
                "n_docs": n_docs,
                "avg_nonzero_terms": avg_nonzero_terms,
            }
        )
        _log_info(logger, "avg_vocab_usage [%s/%s] 年份 %s 完成: n_docs=%s avg_nonzero_terms=%.2f", index, len(vector_paths), year, n_docs, avg_nonzero_terms)
    return pd.DataFrame(rows)


def compute_df_pair_sum(stage1_dir: Path, max_year_gap: int = 5, *, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    vocab = _load_vocab(stage1_dir / "vocab" / "final_vocab.json")
    vocab_size = len(vocab)
    vectors = []
    years = []
    df_paths = sorted((stage1_dir / "df").glob("term_df_year=*.json"), key=_extract_year)
    _log_info(logger, "开始计算 df_pair_sum，共 %s 个年份 DF 文件，vocab_size=%s", len(df_paths), vocab_size)
    for index, path in enumerate(df_paths, start=1):
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        term_df = payload.get("df", {})
        indices = []
        values = []
        for term, count in term_df.items():
            if term in vocab:
                indices.append(vocab[term])
                values.append(count)
        vec = sparse.csr_matrix(
            (values, ([0] * len(indices), indices)),
            shape=(1, vocab_size),
            dtype=np.float64,
        )
        vectors.append(vec)
        year = _extract_year(path)
        years.append(year)
        _log_info(logger, "df_pair_sum [%s/%s] 年份 %s 完成: 有效词数=%s", index, len(df_paths), year, len(indices))
    if not vectors:
        return pd.DataFrame(columns=["year_x", "year_y", "sum_df_product"])
    _log_info(logger, "df_pair_sum 开始计算年份两两乘积矩阵: years=%s", len(years))
    matrix = sparse.vstack(vectors)
    dense = np.asarray(cast(Any, matrix.dot(matrix.T)).toarray())
    rows = []
    for i, year_x in enumerate(years):
        for j, year_y in enumerate(years):
            if abs(year_x - year_y) <= max_year_gap:
                rows.append(
                    {
                        "year_x": year_x,
                        "year_y": year_y,
                        "sum_df_product": float(dense[i, j]),
                    }
                )
    _log_info(logger, "df_pair_sum 完成: 输出 %s 行，max_year_gap=%s", len(rows), max_year_gap)
    return pd.DataFrame(rows)


def _keep_topk_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    return _keep_multi_topk_per_row(matrix, [k])[k]


def _keep_multi_topk_per_row(matrix: sparse.csr_matrix, topk_values: Sequence[int]) -> Dict[int, sparse.csr_matrix]:
    ks = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    matrix = matrix.tocsr()
    if not ks:
        return {0: matrix}

    matrix_shape = cast(tuple[int, int], matrix.shape)
    data_array = cast(np.ndarray, matrix.data)
    indices_array = cast(np.ndarray, matrix.indices)
    indptr_array = cast(np.ndarray, matrix.indptr)
    row_nnz = np.diff(indptr_array)

    out_indptr: Dict[int, np.ndarray] = {}
    out_data: Dict[int, np.ndarray] = {}
    out_indices: Dict[int, np.ndarray] = {}
    offsets: Dict[int, int] = {}

    for k in ks:
        keep_counts = np.minimum(row_nnz, k).astype(indptr_array.dtype, copy=False)
        indptr = np.empty(matrix_shape[0] + 1, dtype=indptr_array.dtype)
        indptr[0] = 0
        np.cumsum(keep_counts, out=indptr[1:])
        total_keep = int(indptr[-1])
        out_indptr[k] = indptr
        out_data[k] = np.empty(total_keep, dtype=data_array.dtype)
        out_indices[k] = np.empty(total_keep, dtype=indices_array.dtype)
        offsets[k] = 0

    for row_idx in range(matrix_shape[0]):
        start = indptr_array[row_idx]
        end = indptr_array[row_idx + 1]
        row_data = data_array[start:end]
        row_indices = indices_array[start:end]
        row_len = len(row_data)
        if row_len == 0:
            continue

        for k in ks:
            if row_len <= k:
                chosen = np.arange(row_len)
            else:
                chosen = np.argpartition(row_data, -k)[-k:]
            write_start = offsets[k]
            write_end = write_start + len(chosen)
            out_data[k][write_start:write_end] = row_data[chosen]
            out_indices[k][write_start:write_end] = row_indices[chosen]
            offsets[k] = write_end

    return {
        k: sparse.csr_matrix((out_data[k], out_indices[k], out_indptr[k]), shape=matrix_shape)
        for k in ks
    }


def compute_topk_pair_sum(stage1_dir: Path, topk: int, max_year_gap: int = 5, *, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    return compute_multi_topk_pair_sum(
        stage1_dir,
        topk_values=[topk],
        max_year_gap=max_year_gap,
        logger=logger,
    )[topk]


def compute_multi_topk_pair_sum(
    stage1_dir: Path,
    topk_values: Sequence[int],
    max_year_gap: int = 5,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    vectors_dir = stage1_dir / "vectors"
    requested_topks = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    if not requested_topks:
        return {}
    years = []
    df_vectors_by_k: Dict[int, list[sparse.csr_matrix]] = {topk: [] for topk in requested_topks}
    yearly_rows_by_k: Dict[int, list[Dict[str, float]]] = {topk: [] for topk in requested_topks}
    vector_paths = sorted(vectors_dir.glob("year=*.npz"), key=_extract_year)
    _log_info(logger, "开始计算多组 topk_pair_sum: topk_values=%s，共 %s 个年份矩阵", list(requested_topks), len(vector_paths))
    for index, path in enumerate(vector_paths, start=1):
        year = _extract_year(path)
        _log_info(logger, "multi_topk_pair_sum [%s/%s] 读取年份 %s: %s", index, len(vector_paths), year, path.name)
        matrix = sparse.load_npz(path)
        matrices_topk = _keep_multi_topk_per_row(matrix, requested_topks)
        years.append(year)
        for topk, matrix_topk in matrices_topk.items():
            matrix_topk_shape = cast(tuple[int, int], matrix_topk.shape)
            doc_sums = np.asarray(matrix_topk.sum(axis=1)).reshape(-1)
            squared = matrix_topk.copy()
            squared.data = cast(np.ndarray, squared.data) ** 2
            sq_sums = np.asarray(squared.sum(axis=1)).reshape(-1)
            yearly_rows_by_k[topk].append(
                {
                    "year": year,
                    "n_docs": int(matrix_topk_shape[0]),
                    "avg_weight_sum": float(doc_sums.mean()) if len(doc_sums) else 0.0,
                    "avg_squared_weight_sum": float(sq_sums.mean()) if len(sq_sums) else 0.0,
                }
            )
            df_vec = matrix_topk.astype(bool).astype(np.float64).sum(axis=0)
            df_vectors_by_k[topk].append(sparse.csr_matrix(df_vec))
            _log_info(
                logger,
                "multi_topk_pair_sum(k=%s) [%s/%s] 年份 %s 完成: n_docs=%s avg_weight_sum=%.4f",
                topk,
                index,
                len(vector_paths),
                year,
                int(matrix_topk_shape[0]),
                float(doc_sums.mean()) if len(doc_sums) else 0.0,
            )

    outputs: Dict[int, Dict[str, pd.DataFrame]] = {}
    for topk in requested_topks:
        df_vectors = df_vectors_by_k[topk]
        yearly_rows = yearly_rows_by_k[topk]
        if not df_vectors:
            empty = pd.DataFrame(columns=["year_x", "year_y", "sum_df_product"])
            outputs[topk] = {
                "yearly": pd.DataFrame(columns=["year", "n_docs", "avg_weight_sum", "avg_squared_weight_sum"]),
                "pairwise": empty,
            }
            continue
        _log_info(logger, "multi_topk_pair_sum(k=%s) 开始计算年份两两乘积矩阵: years=%s", topk, len(years))
        stacked = sparse.vstack(df_vectors)
        dense = np.asarray(cast(Any, stacked.dot(stacked.T)).toarray())
        pair_rows = []
        for i, year_x in enumerate(years):
            for j, year_y in enumerate(years):
                if abs(year_x - year_y) <= max_year_gap:
                    pair_rows.append(
                        {
                            "year_x": year_x,
                            "year_y": year_y,
                            "sum_df_product": float(dense[i, j]),
                        }
                    )
        _log_info(logger, "multi_topk_pair_sum(k=%s) 完成: yearly_rows=%s pair_rows=%s", topk, len(yearly_rows), len(pair_rows))
        outputs[topk] = {
            "yearly": pd.DataFrame(yearly_rows),
            "pairwise": pd.DataFrame(pair_rows),
        }
    return outputs


def compute_yearly_top_vocab(stage1_dir: Path, topk: int, *, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    rows = []
    df_paths = sorted((stage1_dir / "df").glob("term_df_year=*.json"), key=_extract_year)
    _log_info(logger, "开始计算 yearly_top_vocab，共 %s 个年份 DF 文件，topk=%s", len(df_paths), topk)
    for index, path in enumerate(df_paths, start=1):
        year = _extract_year(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        total_docs = payload.get("docs", 0)
        for rank, (word, count) in enumerate(
            sorted(payload.get("df", {}).items(), key=lambda item: item[1], reverse=True)[:topk],
            start=1,
        ):
            rows.append(
                {
                    "year": year,
                    "rank": rank,
                    "word": word,
                    "doc_count": int(count),
                    "doc_ratio": (float(count) / total_docs) if total_docs else 0.0,
                }
            )
        _log_info(logger, "yearly_top_vocab [%s/%s] 年份 %s 完成: total_docs=%s", index, len(df_paths), year, total_docs)
    return pd.DataFrame(rows)


def compute_yearly_vocab_size(stage1_dir: Path, *, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    vocab_path = stage1_dir / "vocab" / "final_vocab.json"
    valid_vocab = set(_load_vocab(vocab_path).keys()) if vocab_path.exists() else set()
    rows = []
    union_vocab = set()
    df_paths = sorted((stage1_dir / "df").glob("term_df_year=*.json"), key=_extract_year)
    _log_info(logger, "开始计算 yearly_vocab_size，共 %s 个年份 DF 文件", len(df_paths))
    for index, path in enumerate(df_paths, start=1):
        year = _extract_year(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        df_dict = payload.get("df", {})
        current_vocab = [term for term in df_dict if not valid_vocab or term in valid_vocab]
        union_vocab.update(current_vocab)
        rows.append(
            {
                "year": year,
                "unique_vocab_size": len(current_vocab),
                "total_docs": int(payload.get("docs", 0)),
            }
        )
        _log_info(logger, "yearly_vocab_size [%s/%s] 年份 %s 完成: unique_vocab_size=%s", index, len(df_paths), year, len(current_vocab))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["vocab_union_size"] = len(union_vocab)
    _log_info(logger, "yearly_vocab_size 完成: union_vocab_size=%s", len(union_vocab))
    return frame


def run_diagnostics(
    stage1_dir: Path,
    diagnostics_dir: Path,
    *,
    topk_values: Sequence[int] = (10, 30, 50),
    yearly_top_vocab_k: int = 50,
    max_year_gap: int = 5,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []

    avg_path = diagnostics_dir / "avg_vocab_usage.csv"
    _log_info(logger, "diagnostics 任务 1/4: avg_vocab_usage")
    compute_avg_vocab_usage(stage1_dir, logger=logger).to_csv(avg_path, index=False)
    written_paths.append(avg_path)
    _log_info(logger, "diagnostics 已写出: %s", avg_path.name)

    pair_path = diagnostics_dir / "df_pair_sum.csv"
    _log_info(logger, "diagnostics 任务 2/4: df_pair_sum")
    compute_df_pair_sum(stage1_dir, max_year_gap=max_year_gap, logger=logger).to_csv(pair_path, index=False)
    written_paths.append(pair_path)
    _log_info(logger, "diagnostics 已写出: %s", pair_path.name)

    vocab_path = diagnostics_dir / f"yearly_top_vocab_top{yearly_top_vocab_k}.csv"
    _log_info(logger, "diagnostics 任务 3/4: yearly_top_vocab topk=%s", yearly_top_vocab_k)
    compute_yearly_top_vocab(stage1_dir, topk=yearly_top_vocab_k, logger=logger).to_csv(vocab_path, index=False)
    written_paths.append(vocab_path)
    _log_info(logger, "diagnostics 已写出: %s", vocab_path.name)

    vocab_size_path = diagnostics_dir / "yearly_vocab_size.csv"
    _log_info(logger, "diagnostics 任务 4/4: yearly_vocab_size")
    compute_yearly_vocab_size(stage1_dir, logger=logger).to_csv(vocab_size_path, index=False)
    written_paths.append(vocab_size_path)
    _log_info(logger, "diagnostics 已写出: %s", vocab_size_path.name)

    topk_outputs = compute_multi_topk_pair_sum(
        stage1_dir,
        topk_values=topk_values,
        max_year_gap=max_year_gap,
        logger=logger,
    )
    for topk in topk_values:
        _log_info(logger, "diagnostics TopK 权重统计: topk=%s", topk)
        outputs = topk_outputs[int(topk)]
        pairwise_path = diagnostics_dir / f"topk_df_pair_sum_k{topk}.csv"
        outputs["pairwise"].to_csv(pairwise_path, index=False)
        written_paths.append(pairwise_path)
        _log_info(logger, "diagnostics 已写出: %s", pairwise_path.name)

        yearly_path = diagnostics_dir / f"topk_weight_stats_k{topk}.csv"
        outputs["yearly"].to_csv(yearly_path, index=False)
        written_paths.append(yearly_path)
        _log_info(logger, "diagnostics 已写出: %s", yearly_path.name)

    return written_paths
