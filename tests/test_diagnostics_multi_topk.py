from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from analysis.common.diagnostics import compute_multi_topk_pair_sum


def _keep_topk_per_row_reference(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    if k <= 0:
        return matrix
    matrix = matrix.tocsr()
    data_array = matrix.data
    indices_array = matrix.indices
    indptr_array = matrix.indptr
    new_data = []
    new_indices = []
    new_indptr = [0]
    for row_idx in range(matrix.shape[0]):
        start = indptr_array[row_idx]
        end = indptr_array[row_idx + 1]
        row_data = data_array[start:end]
        row_indices = indices_array[start:end]
        if len(row_data) <= k:
            chosen = np.arange(len(row_data))
        else:
            chosen = np.argpartition(row_data, -k)[-k:]
        new_data.append(row_data[chosen])
        new_indices.append(row_indices[chosen])
        new_indptr.append(new_indptr[-1] + len(chosen))
    if new_data:
        data = np.concatenate(new_data)
        indices = np.concatenate(new_indices)
    else:
        data = np.array([], dtype=data_array.dtype)
        indices = np.array([], dtype=indices_array.dtype)
    return sparse.csr_matrix((data, indices, new_indptr), shape=matrix.shape)


def _compute_topk_pair_sum_reference(stage1_dir: Path, topk: int, max_year_gap: int = 5) -> dict[str, pd.DataFrame]:
    vectors_dir = stage1_dir / "vectors"
    years = []
    df_vectors = []
    yearly_rows = []
    vector_paths = sorted(vectors_dir.glob("year=*.npz"), key=lambda path: int(path.stem.split("=")[1]))
    for path in vector_paths:
        year = int(path.stem.split("=")[1])
        matrix = sparse.load_npz(path)
        matrix_topk = _keep_topk_per_row_reference(matrix, topk)
        doc_sums = np.asarray(matrix_topk.sum(axis=1)).reshape(-1)
        squared = matrix_topk.copy()
        squared.data = squared.data ** 2
        sq_sums = np.asarray(squared.sum(axis=1)).reshape(-1)
        yearly_rows.append(
            {
                "year": year,
                "n_docs": int(matrix_topk.shape[0]),
                "avg_weight_sum": float(doc_sums.mean()) if len(doc_sums) else 0.0,
                "avg_squared_weight_sum": float(sq_sums.mean()) if len(sq_sums) else 0.0,
            }
        )
        df_vec = matrix_topk.astype(bool).astype(np.float64).sum(axis=0)
        df_vectors.append(sparse.csr_matrix(df_vec))
        years.append(year)
    dense = np.asarray(sparse.vstack(df_vectors).dot(sparse.vstack(df_vectors).T).toarray())
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
    return {
        "yearly": pd.DataFrame(yearly_rows),
        "pairwise": pd.DataFrame(pair_rows),
    }


class DiagnosticsMultiTopkTests(unittest.TestCase):
    def test_multi_topk_matches_single_topk_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vectors_dir = root / "vectors"
            vectors_dir.mkdir(parents=True)

            matrix_2000 = sparse.csr_matrix(
                np.array(
                    [
                        [0.0, 0.9, 0.1, 0.7, 0.2],
                        [0.4, 0.3, 0.8, 0.1, 0.6],
                        [0.5, 0.2, 0.0, 0.4, 0.3],
                    ],
                    dtype=np.float64,
                )
            )
            matrix_2001 = sparse.csr_matrix(
                np.array(
                    [
                        [0.6, 0.1, 0.2, 0.5, 0.4],
                        [0.0, 0.7, 0.9, 0.3, 0.2],
                        [0.8, 0.4, 0.1, 0.2, 0.5],
                    ],
                    dtype=np.float64,
                )
            )
            sparse.save_npz(vectors_dir / "year=2000.npz", matrix_2000)
            sparse.save_npz(vectors_dir / "year=2001.npz", matrix_2001)

            expected_k1 = _compute_topk_pair_sum_reference(root, topk=1, max_year_gap=5)
            expected_k3 = _compute_topk_pair_sum_reference(root, topk=3, max_year_gap=5)
            actual = compute_multi_topk_pair_sum(root, topk_values=[1, 3], max_year_gap=5)

            pd.testing.assert_frame_equal(expected_k1["yearly"], actual[1]["yearly"], check_dtype=False)
            pd.testing.assert_frame_equal(expected_k1["pairwise"], actual[1]["pairwise"], check_dtype=False)
            pd.testing.assert_frame_equal(expected_k3["yearly"], actual[3]["yearly"], check_dtype=False)
            pd.testing.assert_frame_equal(expected_k3["pairwise"], actual[3]["pairwise"], check_dtype=False)


if __name__ == "__main__":
    unittest.main()
