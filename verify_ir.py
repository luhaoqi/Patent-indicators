import argparse
import cProfile
import io
import logging
import os
import pstats
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from patent_quality.config import Config
from patent_quality.log import get_logger
from patent_quality.pair_compute import compute_pair_contrib
from patent_quality.project_paths import (
    build_experiment_layout,
    infer_experiment_id_from_stage1_dir,
    resolve_project_path,
)


def run_verification(
    target_year: int,
    *,
    experiment_id: str,
    stage1_dir: Path,
    output_root: str,
    compute_bound: int,
) -> tuple[Path, Path]:
    cfg = Config(data_path=".")
    layout = build_experiment_layout(experiment_id, output_root=output_root)
    layout.ensure_verification_dirs()

    original_vectors_filtered = stage1_dir / getattr(cfg, "vectors_filtered_dir", "vectors_filtered")
    if not original_vectors_filtered.exists():
        raise FileNotFoundError(f"Original vectors not found at {original_vectors_filtered}")

    verify_dir = layout.verification_dir / "ir" / f"year={target_year}"
    verify_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ["pair_contrib", "postings", "stats"]:
        d_path = verify_dir / subdir
        if d_path.exists():
            shutil.rmtree(d_path)

    cfg.artifacts_dir = os.fspath(verify_dir)
    cfg.ensure_dirs()
    cfg.log_file = os.fspath(verify_dir / "run.log")
    cfg.use_vectors_filtered_for_bsfs = True
    cfg.vectors_filtered_dir = os.fspath(original_vectors_filtered)
    cfg.block_size_docs = 10000
    cfg.postings_mmap = True
    cfg.enable_maxscore = False

    logger = get_logger(level="INFO", log_file=cfg.log_file)
    logger.info("=== %s IR Verification Start ===", target_year)
    logger.info("Input stage1 dir: %s", stage1_dir)
    logger.info("Input vectors: %s", original_vectors_filtered)
    logger.info("Verification dir: %s", verify_dir)

    start_total = time.time()
    window = cfg.window_size

    available_years = []
    for filename in os.listdir(original_vectors_filtered):
        if filename.startswith("year=") and filename.endswith(".npz"):
            available_years.append(int(filename.split("=")[1].split(".")[0]))
    available_years.sort()

    if target_year not in available_years:
        raise ValueError(f"Target year {target_year} not found in vectors.")

    back_years = [year for year in available_years if target_year - window <= year < target_year]
    forward_years = [year for year in available_years if target_year < year <= target_year + window]
    logger.info("Back Years: %s", back_years)
    logger.info("Forward Years: %s", forward_years)

    pairs_computed = 0
    for year in back_years:
        logger.info("Computing Pair (BS): %s - %s", year, target_year)
        t_start = time.time()
        compute_pair_contrib(cfg, year, target_year)
        logger.info("Pair (BS) %s - %s took %.2fs", year, target_year, time.time() - t_start)
        pairs_computed += 1
        if pairs_computed >= compute_bound:
            logger.info("Profiling limit reached. Stop after %s pairs.", compute_bound)
            break

    if pairs_computed < compute_bound:
        for year in forward_years:
            logger.info("Computing Pair (FS): %s - %s", target_year, year)
            t_start = time.time()
            compute_pair_contrib(cfg, target_year, year)
            logger.info("Pair (FS) %s - %s took %.2fs", target_year, year, time.time() - t_start)
            pairs_computed += 1
            if pairs_computed >= compute_bound:
                logger.info("Profiling limit reached. Stop after %s pairs.", compute_bound)
                break

    matrix_path = original_vectors_filtered / f"year={target_year}.npz"
    target_matrix = sparse.load_npz(matrix_path)
    n_target = target_matrix.shape[0]

    bs_vec = np.zeros(n_target, dtype=np.float32)
    fs_vec = np.zeros(n_target, dtype=np.float32)
    pair_dir = verify_dir / "pair_contrib"

    for year in back_years:
        pair_path = pair_dir / f"x={year}_y={target_year}.npz"
        if pair_path.exists():
            data = np.load(pair_path)
            bs_vec += data["contrib_y"]

    for year in forward_years:
        pair_path = pair_dir / f"x={target_year}_y={year}.npz"
        if pair_path.exists():
            data = np.load(pair_path)
            fs_vec += data["contrib_x"]

    out_csv = verify_dir / "stats" / f"bsfs_year={target_year}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as fh:
        fh.write("row,BS,FS\n")
        for idx in range(n_target):
            fh.write(f"{idx},{bs_vec[idx]},{fs_vec[idx]}\n")

    logger.info("Verification Done. Output at: %s", out_csv)
    logger.info("Total Time: %.2fs", time.time() - start_total)
    return verify_dir, out_csv


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IR BS/FS verification for a specific year.")
    parser.add_argument("year", type=int, nargs="?", default=2008, help="The target year to verify")
    parser.add_argument("--experiment-id", help="实验 ID")
    parser.add_argument("--stage1-dir", help="第一阶段输出目录")
    parser.add_argument("--output-root", default="outputs/experiments", help="统一实验输出根目录")
    parser.add_argument("--compute-bound", type=int, default=5, help="最多计算多少个年份对")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    stage1_dir = resolve_project_path(args.stage1_dir) if args.stage1_dir else None
    experiment_id = args.experiment_id or infer_experiment_id_from_stage1_dir(stage1_dir or "baseline_1985_2025_window5")

    print(f"=== Profiling Enabled for Year {args.year} ({experiment_id}) ===")
    profiler = cProfile.Profile()
    profiler.enable()

    verify_dir = None
    try:
        verify_dir, _ = run_verification(
            args.year,
            experiment_id=experiment_id,
            stage1_dir=stage1_dir or build_experiment_layout(experiment_id, output_root=args.output_root).stage1_dir,
            output_root=args.output_root,
            compute_bound=args.compute_bound,
        )
    finally:
        profiler.disable()
        profile_text_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=profile_text_stream).sort_stats("cumtime")
        stats.print_stats(30)
        profile_text = profile_text_stream.getvalue()
        print("\nGenerating profile stats...")
        print(profile_text)

        if verify_dir is None:
            verify_dir = build_experiment_layout(experiment_id, output_root=args.output_root).verification_dir / "ir" / f"year={args.year}"
            verify_dir.mkdir(parents=True, exist_ok=True)

        stats_file = verify_dir / f"profile_ir_{args.year}.prof"
        stats.dump_stats(stats_file)
        print(f"Profile data saved to: {stats_file}")

        log_file = verify_dir / "run.log"
        try:
            logger = logging.getLogger("patent_quality")
            if logger.handlers:
                logger.info("\n=== Performance Profile ===\n%s", profile_text)
            elif log_file.exists():
                with log_file.open("a", encoding="utf-8") as fh:
                    fh.write("\n=== Performance Profile ===\n")
                    fh.write(profile_text)
                    fh.write("===========================\n")
        except Exception as exc:
            print(f"Failed to write profile to log: {exc}")
