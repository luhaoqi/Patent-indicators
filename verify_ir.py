import os
import sys
import time
import cProfile
import pstats
import io
import numpy as np
import shutil
import argparse
import logging
from patent_quality.config import Config
from patent_quality.pair_compute import compute_pair_contrib
from patent_quality.postings import build_postings_for_year
from patent_quality.log import get_logger
from scipy import sparse

def run_verification(target_year: int):
    # 1. Setup Config
    # Hack: We initialize with current dir to load defaults
    cfg = Config(data_path=".") 
    
    # 获取原始 artifacts 路径
    project_root = os.getcwd()
    original_artifacts = os.path.join(project_root, "artifacts_full")
    original_vectors_filtered = os.path.join(original_artifacts, getattr(cfg, "vectors_filtered_dir", "vectors_filtered"))
    
    if not os.path.exists(original_vectors_filtered):
        print(f"Error: Original vectors not found at {original_vectors_filtered}")
        raise FileNotFoundError(f"Error: Original vectors not found")

    # 设置测试输出目录
    test_dir = os.path.join(project_root, f"test_IR_artifacts")
    os.makedirs(test_dir, exist_ok=True)

    # Clean only specific subdirectories to keep logs
    for subdir in ["pair_contrib", "postings", "stats"]:
        d_path = os.path.join(test_dir, subdir)
        if os.path.exists(d_path):
            try:
                shutil.rmtree(d_path)
            except Exception as e:
                print(f"Warning: Failed to clean {d_path}: {e}")
    
    # Update cfg to use test_dir as base for outputs
    cfg.artifacts_dir = test_dir
    cfg.ensure_dirs()
    
    # Configure Log
    cfg.log_file = os.path.join(test_dir, "run.log")
    
    # CRITICAL: Redirect vectors_filtered_dir to the ORIGINAL absolute path
    # So that _resolve_vectors_base returns the original path
    cfg.use_vectors_filtered_for_bsfs = True
    cfg.vectors_filtered_dir = original_vectors_filtered
    
    # Configure IR parameters
    cfg.block_size_docs = 10000
    cfg.postings_mmap = True
    cfg.enable_maxscore = False # Start with False for safety/correctness first
    compute_bound = 5
    
    logger = get_logger(level="INFO", log_file=cfg.log_file)
    logger.info(f"=== {target_year} IR Verification Start ===")
    logger.info(f"Input Vectors: {original_vectors_filtered}")
    logger.info(f"Test Output Dir: {test_dir}")
    logger.info(f"Log File: {cfg.log_file}")

    start_total = time.time()
    
    # 2. Determine Years
    window = cfg.window_size
    
    # Scan available years
    available_years = []
    for f in os.listdir(original_vectors_filtered):
        if f.startswith("year=") and f.endswith(".npz"):
            y = int(f.split("=")[1].split(".")[0])
            available_years.append(y)
    available_years.sort()
    
    if target_year not in available_years:
        logger.error(f"Target year {target_year} not found in vectors.")
        return

    back_years = [y for y in available_years if target_year - window <= y < target_year]
    forward_years = [y for y in available_years if target_year < y <= target_year + window]
    
    logger.info(f"Target: {target_year}")
    logger.info(f"Back Years: {back_years}")
    logger.info(f"Forward Years: {forward_years}")
    logger.info(f"Similarity Threshold: {cfg.similarity_threshold}")
    
    # 3. Compute Pairs
    pairs_computed = 0
    
    # Step 3.1: BS Pairs
    for y in back_years:
        logger.info(f"Computing Pair (BS): {y} - {target_year}")
        t_start = time.time()
        compute_pair_contrib(cfg, y, target_year)
        logger.info(f"Pair (BS) {y} - {target_year} took {time.time() - t_start:.2f}s")
        pairs_computed += 1
        if pairs_computed >= compute_bound: # Limit for profiling
            logger.info("Profiling Limit Reached (2 pairs). Stopping.")
            return
        
    # Step 3.2: FS Pairs
    for y in forward_years:
        logger.info(f"Computing Pair (FS): {target_year} - {y}")
        t_start = time.time()
        compute_pair_contrib(cfg, target_year, y)
        logger.info(f"Pair (FS) {target_year} - {y} took {time.time() - t_start:.2f}s")
        pairs_computed += 1
        if pairs_computed >= compute_bound: # Limit for profiling
            logger.info("Profiling Limit Reached (2 pairs). Stopping.")
            return
        
    # 4. Aggregate
    logger.info("Aggregating results...")
    
    # Get shape of target year
    v_path = os.path.join(original_vectors_filtered, f"year={target_year}.npz")
    M_target = sparse.load_npz(v_path)
    N_target = M_target.shape[0]
    
    bs_vec = np.zeros(N_target, dtype=np.float32)
    fs_vec = np.zeros(N_target, dtype=np.float32)
    
    pair_dir = os.path.join(test_dir, "pair_contrib")
    
    # Sum BS
    for y in back_years:
        # pair file: x=y_y=target (since y < target)
        p_path = os.path.join(pair_dir, f"x={y}_y={target_year}.npz")
        if os.path.exists(p_path):
            data = np.load(p_path)
            # In pair (y, target), target is 'y'. So we need contrib_y.
            bs_vec += data['contrib_y']
        else:
            logger.warning(f"Missing pair {p_path}")
            
    # Sum FS
    for y in forward_years:
        # pair file: x=target_y=y (since target < y)
        p_path = os.path.join(pair_dir, f"x={target_year}_y={y}.npz")
        if os.path.exists(p_path):
            data = np.load(p_path)
            # In pair (target, y), target is 'x'. So we need contrib_x.
            fs_vec += data['contrib_x']
        else:
            logger.warning(f"Missing pair {p_path}")
            
    # 5. Output
    stats_dir = os.path.join(test_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    out_csv = os.path.join(stats_dir, f"bsfs_year={target_year}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("row,BS,FS\n")
        for i in range(N_target):
            f.write(f"{i},{bs_vec[i]},{fs_vec[i]}\n")
            
    logger.info(f"Verification Done. Output at: {out_csv}")
    logger.info(f"Total Time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run IR BS/FS verification for a specific year.')
    parser.add_argument('year', type=int, nargs='?', default=2008, help='The target year to verify (default: 2008)')
    args = parser.parse_args()
    
    print(f"=== Profiling Enabled for Year {args.year} ===")
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        run_verification(args.year)
    finally:
        profiler.disable()
        
        # Save and Print Stats
        stats_file = os.path.join(os.getcwd(), f"profile_ir_{args.year}.prof")
        print(f"\nGenerating profile stats...")
        
        # Capture stats to string
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        stats.print_stats(30)
        profile_text = s.getvalue()
        
        # Print to console
        print(profile_text)
        
        # Save to file
        stats.dump_stats(stats_file)
        print(f"Profile data saved to: {stats_file}")
        
        # Append to log file
        log_file = os.path.join(os.getcwd(), "test_IR_artifacts", "run.log")
        try:
            # Try to use existing logger if available
            logger = logging.getLogger("patent_quality")
            if logger.handlers:
                logger.info("\n=== Performance Profile ===\n" + profile_text)
            else:
                # Fallback to direct file append if logger not configured
                if os.path.exists(log_file):
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n=== Performance Profile ===\n")
                        f.write(profile_text)
                        f.write("===========================\n")
        except Exception as e:
            print(f"Failed to write profile to log: {e}")
