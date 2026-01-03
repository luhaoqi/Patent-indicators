import os
import glob
import json
import re
import argparse
from tqdm import tqdm

def load_vocab(vocab_path):
    print(f"正在加载词表: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        vocab_map = data["vocab"] if "vocab" in data else data
        
    # 反转词表: ID -> Word
    id2word = {v: k for k, v in vocab_map.items()}
    return id2word

def get_year_from_filename(filename):
    match = re.search(r"year=(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def process_year(df_file, id2word, top_k):
    with open(df_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df_dict = data.get("df", {})
    total_docs = data.get("docs", 0)
    
    # 转换为列表并排序: (count, word)
    # df_dict 是 {word: count}
    items = list(df_dict.items())
    
    # 按 count 降序排序
    items.sort(key=lambda x: x[1], reverse=True)
    
    # 取 Top K
    top_items = items[:top_k]
    
    return total_docs, top_items

def main():
    parser = argparse.ArgumentParser(description="统计每年出现频率最高的 TopK 词汇")
    parser.add_argument("--dir", default="artifacts_full", help="artifacts 根目录")
    parser.add_argument("--k", type=int, default=50, help="每年显示的 TopK 词汇数")
    parser.add_argument("--output", default="stat/yearly_top_vocab.txt", help="输出文件路径")
    
    args = parser.parse_args()
    
    df_dir = os.path.join(args.dir, "df")
    vocab_path = os.path.join(args.dir, "vocab", "final_vocab.json")
    
    if not os.path.exists(df_dir):
        print(f"错误: DF 目录不存在 {df_dir}")
        return
        
    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1. 找到所有 DF 文件
    # 我们直接使用 json 文件，因为这里不需要 "每个文档只取 TopK" 的逻辑
    # 用户的需求是 "统计包含所有文档"，即统计词汇出现的文档次数
    # 这正是 term_df_year=*.json 存储的内容
    pattern = os.path.join(df_dir, "term_df_year=*.json")
    files = glob.glob(pattern)
    files.sort(key=get_year_from_filename)
    
    if not files:
        print("未找到 DF 文件")
        return

    print(f"开始处理，文件数={len(files)}")
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Top {args.k} Frequent Words Per Year\n")
        f.write("=" * 60 + "\n\n")
        
        for filepath in tqdm(files, desc="Processing"):
            year = get_year_from_filename(filepath)
            
            # 由于 json 文件里直接存储了 word -> count，我们甚至不需要加载 vocab 文件
            # 除非 json 里存的是 id（但根据之前的观察，存的是 term 字符串）
            # 让我们直接读取看看
            
            total_docs, top_items = process_year(filepath, None, args.k)
            
            f.write(f"Year: {year} (Total Docs: {total_docs})\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Rank':<5} | {'Word':<30} | {'Doc Count':<10} | {'Frequency (%)':<15}\n")
            f.write("-" * 60 + "\n")
            
            for i, (word, count) in enumerate(top_items):
                freq = (count / total_docs * 100) if total_docs > 0 else 0
                f.write(f"{i+1:<5} | {word:<30} | {count:<10} | {freq:<15.2f}\n")
            
            f.write("\n")
            
    print(f"完成。结果已保存至 {args.output}")

if __name__ == "__main__":
    main()
