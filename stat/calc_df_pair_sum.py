import os
import glob
import json
import argparse
import re
import numpy as np
from scipy import sparse
from tqdm import tqdm

def load_vocab(vocab_path):
    print(f"正在加载词表: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 兼容两种格式: {"vocab": {...}} 或 直接 {...}
        if "vocab" in data:
            return data["vocab"]
        return data

def load_df_vector(df_path, vocab, year):
    with open(df_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        term_df = data.get("df", {})
    
    # 构建稀疏向量
    # 只需要非零的那些词
    indices = []
    values = []
    
    for term, count in term_df.items():
        if term in vocab:
            idx = vocab[term]
            indices.append(idx)
            values.append(count)
            
    # 创建 csr_matrix 向量 (1, vocab_size)
    vec = sparse.csr_matrix((values, (np.zeros(len(indices)), indices)), 
                           shape=(1, len(vocab)), 
                           dtype=np.float64) # 使用 float64 防止溢出
    return vec

def main(artifacts_dir):
    vocab_path = os.path.join(artifacts_dir, "vocab", "final_vocab.json")
    df_dir = os.path.join(artifacts_dir, "df")
    
    if not os.path.exists(vocab_path):
        print(f"错误: 词表文件不存在 {vocab_path}")
        return
        
    if not os.path.exists(df_dir):
        print(f"错误: DF目录不存在 {df_dir}")
        return

    # 1. 加载词表
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    # 2. 找到所有年份文件
    pattern = os.path.join(df_dir, "term_df_year=*.json")
    files = glob.glob(pattern)
    
    def get_year(filepath):
        match = re.search(r"year=(\d+)", os.path.basename(filepath))
        return int(match.group(1)) if match else 0
        
    files.sort(key=get_year)
    
    years = []
    vectors = []
    
    print("正在加载年份DF数据...")
    for filepath in tqdm(files):
        year = get_year(filepath)
        vec = load_df_vector(filepath, vocab, year)
        years.append(year)
        vectors.append(vec)
        
    if not vectors:
        print("未找到任何DF文件")
        return

    # 3. 堆叠成矩阵 (Years x Vocab)
    print("正在构建矩阵...")
    matrix = sparse.vstack(vectors)
    
    # 4. 计算 pairwise dot product (Years x Years)
    # result[i, j] = vec[i] dot vec[j]
    print("正在计算 pairwise sum...")
    # matrix 是 csr_matrix, dot 是矩阵乘法
    # matrix * matrix.T
    pairwise_matrix = matrix.dot(matrix.T)
    
    # 转换为 dense 方便访问
    pairwise_dense = pairwise_matrix.toarray()
    
    print("\n年份对统计结果 (Sum(df_i * df_j)):")
    print("-" * 60)
    print(f"{'年份1':<10} | {'年份2':<10} | {'Sum(DF*DF)':<20}")
    print("-" * 60)
    
    total_sum = 0.0
    
    n_years = len(years)
    for i in range(n_years):
        for j in range(n_years):
            # 只需要枚举往前往后各5年的年份对
            if abs(years[i] - years[j]) <= 5:
                val = pairwise_dense[i, j]
                total_sum += val
                print(f"{years[i]:<10} | {years[j]:<10} | {val:<20.0f}")

    print("-" * 60)
    print(f"满足条件的年份对总和: {total_sum:.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计任意两年词汇DF乘积之和")
    parser.add_argument("dir", nargs="?", default="artifacts_full", help="artifacts 目录路径")
    
    args = parser.parse_args()
    main(args.dir)
