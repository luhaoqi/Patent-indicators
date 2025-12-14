import os
import csv
from typing import List
from .config import Config
from .log import get_logger


def assemble_final_csv(cfg: Config, output_path: str = "patent_quality_output.csv") -> None:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level)
    rows_out: List[List[str]] = []
    
    # Build header
    header = ["申请号", "申请年份", "专利名称"] + ["BS", "FS", "Quality_q"]+ cfg.extra_cols 
    rows_out.append(header)
    
    stats_dir = os.path.join(cfg.artifacts_dir, "stats")
    index_dir = os.path.join(cfg.artifacts_dir, "index")
    for name in os.listdir(stats_dir):
        if name.startswith("bsfs_year=") and name.endswith(".csv"):
            t = int(name[len("bsfs_year=") : -len(".csv")])
            stats_fp = os.path.join(stats_dir, name)
            index_fp = os.path.join(index_dir, f"year={t}.csv")
            if not os.path.exists(index_fp):
                continue
            bs_list: List[float] = []
            fs_list: List[float] = []
            with open(stats_fp, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    bs_list.append(float(row["BS"]))
                    fs_list.append(float(row["FS"]))
            with open(index_fp, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                i = 0
                for row in r:
                    pid = row["申请号"]
                    year = row["申请年份"]
                    title = row["专利名称"]
                    
                    extras = [row.get(c, "") for c in cfg.extra_cols]
                    
                    bs = bs_list[i] if i < len(bs_list) else 0.0
                    fs = fs_list[i] if i < len(fs_list) else 0.0
                    q = fs / (bs + cfg.epsilon)
                    
                    out_row = [pid, year, title] + [f"{bs:.8f}", f"{fs:.8f}", f"{q:.8f}"] + extras
                    rows_out.append(out_row)
                    i += 1
            logger.info(f"合并年份={t} 行数={len(bs_list)}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in rows_out:
            w.writerow(row)
    logger.info(f"最终CSV输出: {output_path} 总行数={len(rows_out)-1}")
