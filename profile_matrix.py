import os
import numpy as np
from scipy import sparse

def profile_year(year):
    path = f"D:\\BaiduNetdiskDownload\\工具\\专利指标\\artifacts_full\\vectors_filtered\\year={year}.npz"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    Mx = sparse.load_npz(path).tocsr()
    Nx = Mx.shape[0]
    row_lens = np.diff(Mx.indptr)
    
    if Nx == 0:
        print("Empty matrix")
        return

    avg_len = np.mean(row_lens)
    p50 = np.percentile(row_lens, 50)
    p90 = np.percentile(row_lens, 90)
    p99 = np.percentile(row_lens, 99)
    max_len = np.max(row_lens)
    
    print(f"Year {year} Stats:")
    print(f"  Rows: {Nx}")
    print(f"  Avg Terms: {avg_len:.2f}")
    print(f"  Median (P50): {p50}")
    print(f"  P90: {p90}")
    print(f"  P99: {p99}")
    print(f"  Max: {max_len}")
    
    # Histogram
    counts, bins = np.histogram(row_lens, bins=[0, 1, 5, 10, 20, 50, 100, 1000])
    print("  Distribution:")
    for i in range(len(counts)):
        print(f"    {bins[i]}-{bins[i+1]}: {counts[i]} ({counts[i]/Nx*100:.1f}%)")

if __name__ == "__main__":
    profile_year(2011)
