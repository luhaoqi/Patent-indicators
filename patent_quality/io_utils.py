import json
import os
from typing import Any, Dict, List
from .config import Config
import shutil


def load_checkpoint(cfg: Config) -> Dict[str, Any]:
    p = os.path.join(cfg.artifacts_dir, "checkpoint.json")
    if not os.path.exists(p):
        return {"prepared_tokens": False, "vectorized_years": [], "bsfs_years": [], "final_csv": False}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(cfg: Config, ckpt: Dict[str, Any]) -> None:
    with open(os.path.join(cfg.artifacts_dir, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

def clear_artifacts(cfg: Config) -> None:
    base = cfg.artifacts_dir
    subdirs = [
        "vocab",
        "df",
        "tokens",
        "vectors",
        getattr(cfg, "vectors_filtered_dir", "vectors_filtered"),
        "index",
        "stats",
        getattr(cfg, "pair_contrib_dir", "pair_contrib"),
        getattr(cfg, "postings_dir", "postings"),
    ]
    for sub in subdirs:
        d = os.path.join(base, sub)
        if not os.path.exists(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except OSError:
                pass
    ckpt = os.path.join(base, "checkpoint.json")
    if os.path.exists(ckpt):
        try:
            os.remove(ckpt)
        except OSError:
            pass
    # ensure dirs exist after cleanup
    cfg.ensure_dirs()


def years_with_tokens(cfg: Config) -> List[int]:
    base = os.path.join(cfg.artifacts_dir, "tokens")
    ys = []
    for name in os.listdir(base):
        if name.startswith("year=") and name.endswith(".jsonl"):
            ys.append(int(name[len("year=") : -len(".jsonl")]))
    ys.sort()
    return ys
