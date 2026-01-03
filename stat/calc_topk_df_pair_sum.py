import os
import glob
import re
import argparse
import numpy as np
from scipy import sparse
from tqdm import tqdm

def keep_topk_per_row(matrix, k):
    """
    对 CSR 矩阵的每一行，只保留数值最大的 Top K 个元素，其余置零（即从稀疏表示中移除）。
    返回一个新的 CSR 矩阵。
    """
    if k <= 0:
        return matrix
        
    # 确保是 CSR 格式
    matrix = matrix.tocsr()
    n_rows = matrix.shape[0]
    
    new_data = []
    new_indices = []
    new_indptr = [0]
    
    # 遍历每一行
    for i in range(n_rows):
        start = matrix.indptr[i]
        end = matrix.indptr[i+1]
        
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        
        if len(row_data) <= k:
            # 如果非零元素个数 <= k，全部保留
            new_data.append(row_data)
            new_indices.append(row_indices)
            new_indptr.append(new_indptr[-1] + len(row_data))
        else:
            # 找到 Top K 的索引
            # argpartition 将第 K 大的元素放在位置 -k，且右边的都比它大（或相等）
            # 我们需要后 K 个
            topk_idx = np.argpartition(row_data, -k)[-k:]
            
            new_data.append(row_data[topk_idx])
            new_indices.append(row_indices[topk_idx])
            new_indptr.append(new_indptr[-1] + k)
            
    # 合并数据
    if new_data:
        new_data = np.concatenate(new_data)
        new_indices = np.concatenate(new_indices)
    else:
        new_data = np.array([])
        new_indices = np.array([])
        
    return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape)

def get_year_from_filename(filename):
    match = re.search(r"year=(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def main():
    parser = argparse.ArgumentParser(description="统计 TopK 过滤后的词汇 DF Pairwise Sum")
    parser.add_argument("--dir", default="artifacts_full", help="artifacts 根目录")
    parser.add_argument("--k", type=int, default=10, help="每个文档保留的 TopK 词汇数")
    parser.add_argument("--output", default="topk_stats.txt", help="结果输出文件路径")
    
    args = parser.parse_args()
    
    vectors_dir = os.path.join(args.dir, "vectors")
    if not os.path.exists(vectors_dir):
        print(f"错误: 向量目录不存在 {vectors_dir}")
        return

    # 找到所有向量文件
    pattern = os.path.join(vectors_dir, "year=*.npz")
    files = glob.glob(pattern)
    files.sort(key=get_year_from_filename)
    
    if not files:
        print("未找到向量文件")
        return

    print(f"开始处理，TopK={args.k}，文件数={len(files)}")
    
    years = []
    df_vectors = []
    
    # 存储每年的平均权重和
    year_stats = []
    
    for filepath in tqdm(files, desc="Processing Years"):
        year = get_year_from_filename(filepath)
        
        # 1. 加载向量 (Docs x Vocab)
        try:
            mat = sparse.load_npz(filepath)
        except Exception as e:
            print(f"无法加载 {filepath}: {e}")
            continue
            
        # 2. TopK 过滤
        mat_topk = keep_topk_per_row(mat, args.k)
        
        # --- 统计 TopK 后的权重和 ---
        # mat_topk 是 CSR 矩阵，sum(axis=1) 计算每一行的和
        # 结果是 (n_docs, 1) 的矩阵
        doc_sums = mat_topk.sum(axis=1)
        
        # 计算平均值
        avg_weight_sum = np.mean(doc_sums)
        
        # 也可以计算平方和的平均值 (L2 Energy)
        # mat_topk.data 是所有非零元素的值
        # 为了按行计算平方和，我们可以先平方 data，再构造矩阵，再求和
        mat_sq = mat_topk.copy()
        mat_sq.data = mat_sq.data ** 2
        doc_sq_sums = mat_sq.sum(axis=1)
        avg_sq_sum = np.mean(doc_sq_sums)
        
        year_stats.append({
            "year": year,
            "avg_sum": avg_weight_sum,
            "avg_sq_sum": avg_sq_sum
        })
        # ---------------------------
        
        # 3. 计算 DF (该年每个词出现在多少个文档中)
        # 将非零位置设为 1，然后按列求和
        # mat_topk > 0 返回 bool 矩阵
        # astype(int) 转换为 0/1
        # sum(axis=0) 对列求和，得到 (1, Vocab) 的矩阵
        # 注意: sum 返回的是 np.matrix (dense)
        
        # 为了节省内存，我们手动操作:
        # 只需要统计每一列的非零个数
        # CSR 格式求列非零计数稍微麻烦，转 CSC 更快？
        # 或者直接利用 (mat_topk > 0).sum(axis=0)
        # 考虑到 mat_topk 已经减少了数据量，直接操作应该可以
        
        # 转换为 bool 矩阵 (结构同 mat_topk)
        mat_bool = mat_topk.astype(bool).astype(np.float64) # 使用 float 方便后续点积
        
        # 计算 DF 向量: 每一列的和
        # result shape: (1, n_vocab)
        df_vec = mat_bool.sum(axis=0)
        
        # 转换为 CSR 存储 (1, n_vocab)
        df_vec_sparse = sparse.csr_matrix(df_vec)
        
        years.append(year)
        df_vectors.append(df_vec_sparse)
        
    if not df_vectors:
        print("没有有效数据")
        return
        
    # 4. 堆叠所有年份的 DF 向量 (n_years, n_vocab)
    print("堆叠 DF 向量...")
    df_matrix = sparse.vstack(df_vectors)
    
    # 5. 计算 Pairwise Sum: matrix * matrix.T
    # 结果 (i, j) = sum(df_vec_i * df_vec_j)
    print("计算 Pairwise Sum...")
    result_matrix = df_matrix.dot(df_matrix.T).toarray()
    
    # 6. 输出结果
    print(f"正在写入结果到 {args.output} ...")
    total_sum = 0.0
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"TopK={args.k}\n\n")
        
        # 输出每年的权重统计
        f.write("=== Yearly TopK Weight Stats ===\n")
        f.write(f"{'Year':<10} | {'Avg Sum(Weights)':<20} | {'Avg Sum(Squared)':<20}\n")
        f.write("-" * 55 + "\n")
        for stat in year_stats:
            f.write(f"{stat['year']:<10} | {stat['avg_sum']:<20.4f} | {stat['avg_sq_sum']:<20.4f}\n")
        f.write("\n")
        
        f.write("=== Pairwise DF Sum ===\n")
        f.write(f"{'Year1':<10} | {'Year2':<10} | {'Sum(DF*DF)':<20}\n")
        f.write("-" * 45 + "\n")
        
        n_years = len(years)
        for i in range(n_years):
            for j in range(n_years):
                # 过滤条件: abs(y1 - y2) <= 5
                if abs(years[i] - years[j]) <= 5:
                    val = result_matrix[i, j]
                    total_sum += val
                    f.write(f"{years[i]:<10} | {years[j]:<10} | {val:<20.0f}\n")
        
        f.write("-" * 45 + "\n")
        f.write(f"Total Sum (filtered): {total_sum:.0f}\n")
        
    print(f"完成。总和: {total_sum:.0f}")

if __name__ == "__main__":
    main()
