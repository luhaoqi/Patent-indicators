import os
import glob
import numpy as np
from scipy import sparse
import argparse
import re

def calculate_avg_nonzero(vectors_dir):
    print(f"正在处理目录: {vectors_dir}")
    
    if not os.path.exists(vectors_dir):
        print(f"错误: 目录 {vectors_dir} 不存在")
        return

    # 获取所有 year=*.npz 文件
    pattern = os.path.join(vectors_dir, "year=*.npz")
    files = glob.glob(pattern)
    
    if not files:
        print(f"在 {vectors_dir} 中未找到 npz 文件")
        return

    # 按年份排序
    def get_year(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r"year=(\d+)", basename)
        return int(match.group(1)) if match else 0

    files.sort(key=get_year)

    print(f"{'年份':<10} | {'文档总数':<10} | {'平均非零词汇数':<15}")
    print("-" * 45)

    results = []

    for filepath in files:
        year = get_year(filepath)
        try:
            # 加载 npz 文件
            # scipy.sparse.load_npz 返回的是 CSR 矩阵
            matrix = sparse.load_npz(filepath)
            
            # 获取文档数 (行数)
            n_docs = matrix.shape[0]
            
            if n_docs == 0:
                print(f"{year:<10} | {0:<10} | {0:<15.2f}")
                continue

            # 计算每一行的非零元素个数
            # getnnz(axis=1) 返回每一行的非零元素个数
            nnz_per_doc = matrix.getnnz(axis=1)
            
            # 计算平均值
            avg_nnz = np.mean(nnz_per_doc)
            
            print(f"{year:<10} | {n_docs:<10} | {avg_nnz:<15.2f}")
            results.append((year, n_docs, avg_nnz))
            
        except Exception as e:
            print(f"处理年份 {year} 时出错: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算每一年份文档的平均非零词汇数")
    parser.add_argument("dir", nargs="?", default="artifacts_full/vectors", help="包含 year=*.npz 文件的 vectors 目录路径")
    
    args = parser.parse_args()
    
    calculate_avg_nonzero(args.dir)
