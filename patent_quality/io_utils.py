import json
import os
from typing import Any, Dict, List
from .config import Config


def load_checkpoint(cfg: Config) -> Dict[str, Any]:
    p = os.path.join(cfg.artifacts_dir, "checkpoint.json")
    if not os.path.exists(p):
        return {"prepared_tokens": False, "vectorized_years": [], "bsfs_years": [], "final_csv": False}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(cfg: Config, ckpt: Dict[str, Any]) -> None:
    with open(os.path.join(cfg.artifacts_dir, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def years_with_tokens(cfg: Config) -> List[int]:
    base = os.path.join(cfg.artifacts_dir, "tokens")
    ys = []
    for name in os.listdir(base):
        if name.startswith("year=") and name.endswith(".jsonl"):
            ys.append(int(name[len("year=") : -len(".jsonl")]))
    ys.sort()
    return ys
