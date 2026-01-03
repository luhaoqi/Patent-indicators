import os
import glob
import json
import re
import argparse
from tqdm import tqdm

def get_year_from_filename(filename):
    match = re.search(r"year=(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def main():
    parser = argparse.ArgumentParser(description="统计每一年实际使用的词汇量（词表大小）")
    parser.add_argument("--dir", default="artifacts_full", help="artifacts 根目录")
    
    args = parser.parse_args()
    
    df_dir = os.path.join(args.dir, "df")
    if not os.path.exists(df_dir):
        print(f"错误: DF 目录不存在 {df_dir}")
        return

    # 找到所有 DF 文件
    pattern = os.path.join(df_dir, "term_df_year=*.json")
    files = glob.glob(pattern)
    files.sort(key=get_year_from_filename)
    
    if not files:
        print("未找到 DF 文件")
        return

    # 加载 final_vocab.json
    vocab_path = os.path.join(args.dir, "vocab", "final_vocab.json")
    valid_vocab = set()
    if os.path.exists(vocab_path):
        print(f"正在加载词表: {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            # final_vocab.json 结构通常是 {"size": N, "vocab": {"word": index, ...}}
            # 我们只需要 keys
            if "vocab" in vocab_data:
                valid_vocab = set(vocab_data["vocab"].keys())
            else:
                # 兼容可能的直接 list 格式
                valid_vocab = set(vocab_data)
        print(f"有效词表大小: {len(valid_vocab)}")
    else:
        print(f"警告: 未找到词表文件 {vocab_path}，将统计所有原始词汇！")

    results = []
    total_vocab_union = set()

    for filepath in tqdm(files, desc="Processing"):
        year = get_year_from_filename(filepath)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # data["df"] 是一个字典: {word: count}
            # 只要 word 在这里出现，说明这一年至少有一个文档用到了它
            df_dict = data.get("df", {})
            total_docs = data.get("docs", 0)
            
            # 过滤: 只保留在 valid_vocab 中的词
            if valid_vocab:
                current_year_vocab = [w for w in df_dict.keys() if w in valid_vocab]
                vocab_size = len(current_year_vocab)
                total_vocab_union.update(current_year_vocab)
            else:
                # 如果没有 vocab 文件，回退到统计所有词
                vocab_size = len(df_dict)
                total_vocab_union.update(df_dict.keys())
            
            results.append((year, vocab_size, total_docs))
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

    print(f"\n{'Year':<10} | {'Unique Vocab Size':<20} | {'Total Docs':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r[0]:<10} | {r[1]:<20} | {r[2]:<15}")
            
    print("-" * 50)
    print(f"所有年份去重后的总词汇量 (Union): {len(total_vocab_union)}")

if __name__ == "__main__":
    main()
