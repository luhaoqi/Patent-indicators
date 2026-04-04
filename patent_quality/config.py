from dataclasses import dataclass, field
from typing import List, Optional
import os
from pathlib import Path


@dataclass
class Config:
    data_path: str
    stopword_paths: List[str] = field(default_factory=list)
    user_dict_path: Optional[str] = None
    col_id: str = "申请号"
    col_date: str = "申请年份"
    col_type: str = "专利类型"
    col_text_parts: List[str] = field(default_factory=lambda: ["专利名称", "摘要文本", "主权项内容"])
    text_sep: str = " "
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
    manual_stopwords_path: str = "./stopword/专利停用词.txt"
    df_ratio_threshold: float = 0.20
    top_df_percent: float = 0.002
    topk_terms_per_doc: int = 30
    vectors_filtered_dir: str = "vectors_filtered"
    use_vectors_filtered_for_bsfs: bool = True
    pair_contrib_dir: str = "pair_contrib"
    postings_dir: str = "postings"
    block_size_docs: int = 10000
    postings_mmap: bool = True
    enable_maxscore: bool = False
    method_version: str = "ir_v1"
    exact_date: bool = False
    public_date_col: str = "公开公告日"
    public_year_col: str = "公开公告年份"
    public_date_ord_col: str = "公开公告日_ord"
    shared_authorized_parts_dir: Optional[str] = "outputs/shared/raw_patent_authorized_parts"

    def __post_init__(self) -> None:
        self.data_path = os.fspath(self.data_path)
        self.stopword_paths = [os.fspath(path) for path in self.stopword_paths]
        if self.user_dict_path is not None:
            self.user_dict_path = os.fspath(self.user_dict_path)
        self.artifacts_dir = os.fspath(self.artifacts_dir)
        if self.log_file is not None:
            self.log_file = os.fspath(self.log_file)
        self.manual_stopwords_path = os.fspath(self.manual_stopwords_path)
        if self.shared_authorized_parts_dir is not None:
            self.shared_authorized_parts_dir = os.fspath(self.shared_authorized_parts_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def active_year_col(self) -> str:
        return self.public_year_col if self.exact_date else self.col_date

    @property
    def input_path(self) -> str:
        if self.exact_date and self.shared_authorized_parts_dir:
            return self.shared_authorized_parts_dir
        return self.data_path

    @property
    def index_identity_columns(self) -> List[str]:
        if self.exact_date:
            return ["申请号", self.public_year_col, self.public_date_col, "专利名称"]
        return ["申请号", self.col_date, "专利名称"]

    @property
    def index_debug_columns(self) -> List[str]:
        if self.exact_date:
            return [self.public_date_ord_col]
        return []

    @property
    def token_metadata_columns(self) -> List[str]:
        if self.exact_date:
            return [self.public_year_col, self.public_date_col, self.public_date_ord_col]
        return []

    @property
    def final_output_path(self) -> Path:
        return self.artifacts_path / "patent_quality_output.csv"

    def artifacts_subdir(self, name: str) -> Path:
        return self.artifacts_path / name

    def ensure_dirs(self) -> None:
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / "vocab").mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / "df").mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / "tokens").mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / "vectors").mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / self.vectors_filtered_dir).mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / "index").mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / "stats").mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / self.pair_contrib_dir).mkdir(parents=True, exist_ok=True)
        (self.artifacts_path / self.postings_dir).mkdir(parents=True, exist_ok=True)
