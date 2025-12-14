import json
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
from .config import Config
from .data_loader import iter_clean_docs
from .nlp import init_jieba, load_stopwords, tokenize
from .log import get_logger
import time
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

_STOPWORDS_CACHE = None

def _init_df_worker(user_dict_path: str | None, stopword_paths: List[str]) -> None:
    init_jieba(user_dict_path)
    global _STOPWORDS_CACHE
    _STOPWORDS_CACHE = load_stopwords(stopword_paths)

def _worker_df(batch: List[Tuple[str, int, str]]):
    global _STOPWORDS_CACHE
    stop = _STOPWORDS_CACHE if _STOPWORDS_CACHE is not None else set()
    g = defaultdict(int)
    by_y: Dict[int, Dict[str, int]] = {}
    dpy = defaultdict(int)
    for pid, year, text in batch:
        toks = tokenize(text, stop)
        if not toks:
            continue
        dpy[year] += 1
        seen = set(toks)
        for w in seen:
            g[w] += 1
            mp = by_y.get(year)
            if mp is None:
                mp = {}
                by_y[year] = mp
            mp[w] = mp.get(w, 0) + 1
    return g, by_y, dpy

def build_vocab(cfg: Config) -> Tuple[Dict[str, int], Dict[int, int]]:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level)
    t0 = time.perf_counter()
    logger.info("初始化分词与停用词")
    init_jieba(cfg.user_dict_path)
    term_df_global: Dict[str, int] = defaultdict(int)
    term_df_by_year: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    docs_per_year: Dict[int, int] = defaultdict(int)
    total_docs = 0
    logger.info("开始扫描数据并统计DF")
    import sys
    sys.stdout.flush()
    stop_paths = cfg.stopword_paths
    batch_size = 5000
    def gen():
        b = []
        for pid, year, text in iter_clean_docs(cfg):
            b.append((pid, year, text))
            if len(b) >= batch_size:
                yield b
                b = []
        if b:
            yield b
    with Pool(processes=min(8, cpu_count()), initializer=_init_df_worker, initargs=(cfg.user_dict_path, stop_paths)) as pool:
        for g, by_y, dpy in tqdm(pool.imap(_worker_df, gen()), desc="scan(batch)"):
            for w, c in g.items():
                term_df_global[w] += c
            for y, mp in by_y.items():
                for w, c in mp.items():
                    term_df_by_year[y][w] += c
            for y, c in dpy.items():
                docs_per_year[y] += c
                total_docs += c
            if total_docs % 10000 == 0:
                logger.info(f"已处理文档: {total_docs}")
    with open(f"{cfg.artifacts_dir}/df/global_df.json", "w", encoding="utf-8") as f:
        json.dump({"total_docs": total_docs, "df": term_df_global}, f, ensure_ascii=False)
    for y, mp in term_df_by_year.items():
        with open(f"{cfg.artifacts_dir}/df/term_df_year={y}.json", "w", encoding="utf-8") as f:
            json.dump({"year": y, "df": mp, "docs": docs_per_year[y]}, f, ensure_ascii=False)
            
    # 输出每年的文档数量统计到文件
    stats_out = os.path.join(cfg.artifacts_dir, "stats")
    if not os.path.exists(stats_out):
        os.makedirs(stats_out)
    with open(os.path.join(stats_out, "docs_per_year.csv"), "w", encoding="utf-8") as f:
        f.write("year,count\n")
        for y in sorted(docs_per_year.keys()):
            f.write(f"{y},{docs_per_year[y]}\n")
    logger.info(f"每年文档数量统计已保存至: {os.path.join(stats_out, 'docs_per_year.csv')}")

    vocab = {}
    terms = []
    for term, df in term_df_global.items():
        if df < cfg.min_term_count:
            continue
        if df / max(1, total_docs) > cfg.max_doc_freq_ratio:
            continue
        terms.append(term)
    terms.sort()
    for term in terms:
        vocab[term] = len(vocab)
    with open(f"{cfg.artifacts_dir}/vocab/final_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"size": len(vocab), "vocab": vocab}, f, ensure_ascii=False)
    dt = time.perf_counter() - t0
    logger.info(f"扫描完成，总文档={total_docs}，词表大小={len(vocab)}，耗时={dt:.2f}s")
    return vocab, docs_per_year
