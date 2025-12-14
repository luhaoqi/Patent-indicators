from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    data_path: str
    stopword_paths: List[str] = field(default_factory=list)
    user_dict_path: Optional[str] = None
    col_id: str = "申请号"
    col_date: str = "申请年份"
    col_type: str = "专利类型"
    col_text_parts: List[str] = field(default_factory=lambda: ["专利名称", "摘要文本", "主权项内容"])
    min_term_count: int = 20
    max_doc_freq_ratio: float = 0.5
    window_size: int = 5
    similarity_threshold: float = 0.05
    epsilon: float = 1e-8
    artifacts_dir: str = "artifacts"
    dtype: str = "float32"
    chunksize: int = 100000
    encoding: Optional[str] = None
    usecols: Optional[List[str]] = None
    vocab_batch_size: int = 10000
    vocab_n_jobs: Optional[int] = 12 # 默认使用cpu_count()
    token_batch_size: int = 10000
    token_n_jobs: Optional[int] = 12 # 默认使用cpu_count()
    log_level: str = "INFO"
    log_file: Optional[str] = None
    skip_if_exists: bool = True
    extra_cols: List[str] = field(default_factory=lambda: ["申请人", "申请人类型", "申请人地址", "申请人城市"])

    def ensure_dirs(self) -> None:
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "vocab"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "df"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "tokens"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "vectors"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "index"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "stats"), exist_ok=True)
