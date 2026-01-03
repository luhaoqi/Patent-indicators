import os
from .config import Config
from .vocab import build_vocab
from .vectorizer import prepare_tokens, vectorize_by_year
from .similarity import compute_bs_fs
from .quality import assemble_final_csv
from .io_utils import load_checkpoint, save_checkpoint, clear_artifacts
from .log import get_logger
from .pruning import prune_vectors_by_year
import time


def run_all(cfg: Config) -> None:
    cfg.ensure_dirs()
    logger = get_logger(level=cfg.log_level, log_file=cfg.log_file)
    t_all = time.perf_counter()
    logger.info(f"开始流水线 (skip_if_exists={cfg.skip_if_exists})")
    if not cfg.skip_if_exists:
        logger.info("检测到 skip_if_exists=False，清理中间结果 artifacts 目录")
        clear_artifacts(cfg)
    ckpt = load_checkpoint(cfg)

    # 级联重置标志：如果前一个阶段重新执行了，后续阶段必须强制重跑（因为数据变了）
    cascade_reset = False

    # --- 阶段1: 构建词表与分年DF ---
    skip_stage1 = False
    if cfg.skip_if_exists and ckpt.get("vocab_size"):
        skip_stage1 = True
    
    if skip_stage1:
        logger.info("阶段1: 已完成，跳过")
    else:
        # 只要决定跑阶段1，就必须清除阶段1的旧状态，并标记级联重置
        cascade_reset = True
        if "vocab_size" in ckpt:
            del ckpt["vocab_size"]
            save_checkpoint(cfg, ckpt)
            
        t0 = time.perf_counter()
        logger.info("阶段1: 构建词表与分年DF")
        vocab, _ = build_vocab(cfg)
        ckpt["vocab_size"] = len(vocab)
        save_checkpoint(cfg, ckpt)
        logger.info(f"阶段1完成 词表大小={len(vocab)} 耗时={(time.perf_counter()-t0):.2f}s")

    # --- 阶段2: 准备分年tokens ---
    # 如果发生了级联重置，必须清除本阶段状态
    if cascade_reset:
        if "prepared_tokens" in ckpt:
            logger.info("因前置阶段变更,清除阶段2状态")
            del ckpt["prepared_tokens"]
            save_checkpoint(cfg, ckpt)
    
    if ckpt.get("prepared_tokens", False):
        logger.info("阶段2: 已完成，跳过")
    else:
        cascade_reset = True # 本阶段执行了，后续阶段需重置
        t1 = time.perf_counter()
        logger.info("阶段2: 准备分年tokens")
        prepare_tokens(cfg)
        ckpt["prepared_tokens"] = True
        save_checkpoint(cfg, ckpt)
        logger.info(f"阶段2完成 耗时={(time.perf_counter()-t1):.2f}s")

    # --- 阶段3: 回顾性TF-BIDF向量化 ---
    if cascade_reset:
        if "vectorized_years" in ckpt:
            logger.info("因前置阶段变更,清除阶段3状态")
            del ckpt["vectorized_years"]
            save_checkpoint(cfg, ckpt)

    if ckpt.get("vectorized_years", False):
        logger.info("阶段3: 已完成，跳过")
    else:
        cascade_reset = True
        t2 = time.perf_counter()
        logger.info("阶段3: 回顾性TF-BIDF向量化")
        vectorize_by_year(cfg)
        ckpt["vectorized_years"] = True
        save_checkpoint(cfg, ckpt)
        logger.info(f"阶段3完成 耗时={(time.perf_counter()-t2):.2f}s")

    # --- 阶段4: 向量剪枝 ---
    if cascade_reset:
        if "vectors_pruned" in ckpt:
            logger.info("因前置阶段变更,清除阶段4状态")
            del ckpt["vectors_pruned"]
            save_checkpoint(cfg, ckpt)

    if ckpt.get("vectors_pruned", False):
        logger.info("阶段4: 已完成，跳过")
    else:
        cascade_reset = True
        t3 = time.perf_counter()
        logger.info("阶段4: 向量剪枝")
        prune_vectors_by_year(cfg)
        ckpt["vectors_pruned"] = True
        save_checkpoint(cfg, ckpt)
        logger.info(f"阶段4完成 耗时={(time.perf_counter()-t3):.2f}s")

    # --- 阶段5: 计算BS/FS ---
    if cascade_reset:
        if "bsfs_years" in ckpt:
            logger.info("因前置阶段变更,清除阶段5状态")
            del ckpt["bsfs_years"]
            save_checkpoint(cfg, ckpt)

    if ckpt.get("bsfs_years", False):
        logger.info("阶段5: 已完成，跳过")
    else:
        cascade_reset = True
        t4 = time.perf_counter()
        logger.info("阶段5: 计算BS/FS")
        compute_bs_fs(cfg)
        ckpt["bsfs_years"] = True
        save_checkpoint(cfg, ckpt)
        logger.info(f"阶段5完成 耗时={(time.perf_counter()-t4):.2f}s")

    # --- 阶段6: 生成最终CSV ---
    if cascade_reset:
        if "final_csv" in ckpt:
            logger.info("因前置阶段变更,清除阶段6状态")
            del ckpt["final_csv"]
            save_checkpoint(cfg, ckpt)

    if ckpt.get("final_csv"):
        logger.info("阶段6: 已完成，跳过")
    else:
        t5 = time.perf_counter()
        logger.info("阶段6: 生成最终CSV")
        out_csv = os.path.join(cfg.artifacts_dir, "patent_quality_output.csv")
        assemble_final_csv(cfg, output_path=out_csv)
        ckpt["final_csv"] = out_csv
        save_checkpoint(cfg, ckpt)
        logger.info(f"阶段6完成 输出={out_csv} 耗时={(time.perf_counter()-t5):.2f}s")
    logger.info(f"流水线完成 总耗时={(time.perf_counter()-t_all):.2f}s")
