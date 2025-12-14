import json
import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.preprocessing import normalize
from .config import Config
from .data_loader import iter_docs_with_title
from .nlp import init_jieba, load_stopwords, tokenize
from .log import get_logger
import time
from multiprocessing import Pool, cpu_count, current_process


def _init_worker(user_dict_path: str):
    """Worker process initialization"""
    # Each worker needs to initialize jieba independently
    # Using a global flag or check inside nlp.py would be safer, 
    # but calling init_jieba is idempotent-ish (loads dict).
    # Ideally we only load once per process.
    try:
        init_jieba(user_dict_path)
    except Exception:
        pass

def _worker_tokenize(args):
    """
    Worker function to tokenize a batch of documents.
    args: (docs_list, stopword_paths)
    docs_list: list of (pid, year, title, text, extra)
    """
    docs, stopword_paths = args
    # Load stopwords in worker if needed, or pass set. 
    # Passing large set via pickle is slow. Better load in init or lazy load.
    # For simplicity, we load stopwords here (cached) or passed.
    # Actually, let's load stopwords in _init_worker if possible, or just load here.
    stop = load_stopwords(stopword_paths)
    
    results = []
    for pid, year, title, text, extra in docs:
        toks = tokenize(text, stop)
        results.append((pid, year, title, toks, extra))
    return results

def prepare_tokens(cfg: Config) -> Dict[int, int]:
    cfg.ensure_dirs()
    # 清理旧的token文件，防止append模式导致重复
    import glob
    token_dir = os.path.join(cfg.artifacts_dir, "tokens")
    for f in glob.glob(os.path.join(token_dir, "year=*.jsonl")):
        try:
            os.remove(f)
        except OSError:
            pass

    logger = get_logger(level=cfg.log_level)
    
    # We do NOT init jieba here in main process if we use spawn (Windows),
    # but it's fine to init for main process logging etc.
    # Workers need their own init.
    
    stop_paths = cfg.stopword_paths
    docs_per_year: Dict[int, int] = defaultdict(int)
    files: Dict[int, str] = {}
    
    logger.info(f"开始分词并按年落盘tokens (并行模式 processes={min(8, cpu_count())})")
    
    # Prepare batch processing
    batch_size = 10000 
    batch_buffer = []
    
    # Function to flush results to disk
    def flush_results(results):
        for pid, year, title, toks, extra in results:
            p = os.path.join(cfg.artifacts_dir, "tokens", f"year={year}.jsonl")
            if year not in files:
                files[year] = p
            
            obj = {"id": pid, "title": title, "tokens": toks}
            obj.update(extra)
            
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            docs_per_year[year] += 1

    import sys
    sys.stdout.flush()
    
    # Use multiprocessing Pool
    # We use a modest number of processes to avoid excessive overhead
    n_jobs = min(8, cpu_count())
    
    with Pool(processes=n_jobs, initializer=_init_worker, initargs=(cfg.user_dict_path,)) as pool:
        # We will use imap_unordered for better responsiveness, 
        # but we need to feed it an iterator of batches.
        
        def batch_generator():
            batch = []
            for item in iter_docs_with_title(cfg):
                batch.append(item)
                if len(batch) >= batch_size:
                    yield (batch, stop_paths)
                    batch = []
            if batch:
                yield (batch, stop_paths)

        # Process chunks
        # Note: iter_docs_with_title yields (pid, year, title, text, extra)
        # We wrap it in tqdm to show progress of *batches* or items?
        # Since we don't know total count easily without scanning, we just use tqdm on the generator result.
        
        for batch_results in tqdm(pool.imap(_worker_tokenize, batch_generator()), desc="tokens(batch)"):
            flush_results(batch_results)

    with open(os.path.join(cfg.artifacts_dir, "stats", "docs_per_year.json"), "w", encoding="utf-8") as f:
        json.dump(docs_per_year, f, ensure_ascii=False)
    for y, c in sorted(docs_per_year.items()):
        logger.info(f"完成年份={y} 文档数={c}")
    return docs_per_year


def _load_vocab(cfg: Config) -> Dict[str, int]:
    with open(f"{cfg.artifacts_dir}/vocab/final_vocab.json", "r", encoding="utf-8") as f:
        return json.load(f)["vocab"]


def _load_term_df_year(cfg: Config, year: int) -> Tuple[Dict[str, int], int]:
    with open(f"{cfg.artifacts_dir}/df/term_df_year={year}.json", "r", encoding="utf-8") as f:
        obj = json.load(f)
        return obj["df"], obj["docs"]


def vectorize_by_year(cfg: Config) -> None:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level)
    vocab = _load_vocab(cfg)
    years = []
    for name in os.listdir(os.path.join(cfg.artifacts_dir, "tokens")):
        if name.startswith("year=") and name.endswith(".jsonl"):
            y = int(name[len("year=") : -len(".jsonl")])
            years.append(y)
    years.sort()
    cumulative_df: Dict[str, int] = defaultdict(int)
    total_docs_so_far = 0
    skip_if_exists = cfg.skip_if_exists
    for y in years:
        t0 = time.perf_counter()
        # 检查是否已完成，实现年级断点续跑
        target_vec = os.path.join(cfg.artifacts_dir, "vectors", f"year={y}.npz")
        target_idx = os.path.join(cfg.artifacts_dir, "index", f"year={y}.csv")
        
        # 只有在配置允许跳过，且文件存在时，才跳过
        if skip_if_exists and os.path.exists(target_vec) and os.path.exists(target_idx):
            logger.info(f"年份={y} 向量已存在，跳过计算（但会加载统计信息以维护回顾性IDF状态）")
            # 必须加载这一年的统计信息，因为后续年份的IDF计算依赖 total_docs_so_far
            df_y, docs_y = _load_term_df_year(cfg, y)
            cumulative_df_y = df_y
            for term, c in cumulative_df_y.items():
                cumulative_df[term] = cumulative_df.get(term, 0) + c
            total_docs_so_far += docs_y
            continue

        df_y, docs_y = _load_term_df_year(cfg, y)
        idf_by_index: Dict[int, float] = {}
        for term, idx in vocab.items():
            c = cumulative_df.get(term, 0)
            idf_by_index[idx] = 0.0 if total_docs_so_far == 0 else float(np.log(total_docs_so_far / (1.0 + c)))
            
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        ids: List[str] = []
        titles: List[str] = []
        extra_data_list: List[Dict[str, str]] = []
        
        import sys
        sys.stdout.flush()
        with open(os.path.join(cfg.artifacts_dir, "tokens", f"year={y}.jsonl"), "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"vec:{y}"):
                obj = json.loads(line)
                pid = obj["id"]
                title = obj.get("title", "")
                toks: List[str] = obj["tokens"]
                
                # Extract extra cols
                extra_vals = {}
                for c in cfg.extra_cols:
                    extra_vals[c] = obj.get(c, "")
                    
                if not toks:
                    continue
                tf: Dict[int, int] = defaultdict(int)
                for w in toks:
                    if w in vocab:
                        tf[vocab[w]] += 1
                if not tf:
                    continue
                row_idx = len(ids)
                ids.append(pid)
                titles.append(title)
                extra_data_list.append(extra_vals)
                
                for j, cnt in tf.items():
                    rows.append(row_idx)
                    cols.append(j)
                    data.append(cnt * idf_by_index.get(j, 0.0))
        if ids:
            m = sparse.csr_matrix((np.array(data, dtype=cfg.dtype), (np.array(rows), np.array(cols))), shape=(len(ids), len(vocab)))
            normalize(m, norm="l2", copy=False)
            sparse.save_npz(os.path.join(cfg.artifacts_dir, "vectors", f"year={y}.npz"), m)
            
            # Save index with extra cols using csv writer
            with open(os.path.join(cfg.artifacts_dir, "index", f"year={y}.csv"), "w", encoding="utf-8", newline='') as fidx:
                writer = csv.writer(fidx)
                header = ["row", "申请号", "申请年份", "专利名称"] + cfg.extra_cols
                writer.writerow(header)
                for i, pid in enumerate(ids):
                    row = [i, pid, y, titles[i]]
                    for c in cfg.extra_cols:
                        row.append(extra_data_list[i].get(c, ""))
                    writer.writerow(row)
                    
            nnz = m.nnz
            dt = time.perf_counter() - t0
            logger.info(f"向量化完成 年份={y} 文档={len(ids)} 维度={len(vocab)} 非零={nnz} 耗时={(time.perf_counter()-t0):.2f}s 基于历史文档={total_docs_so_far}")
        cumulative_df_y = df_y
        for term, c in cumulative_df_y.items():
            cumulative_df[term] = cumulative_df.get(term, 0) + c
        total_docs_so_far += docs_y
