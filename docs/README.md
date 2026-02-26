# 中国专利质量指标复现（Kelly et al., 2018/2021）

目标：基于中文专利文本复现 `q_j = FS_j / BS_j` 指标，严格遵守回顾性 IDF 与 5 年滑窗。

特性：

- 流式/分块处理，避免越过内存红线
- 中文分词与停用词并集，专利领域停用词内置
- 稀疏向量 + 倒排索引计算相似度（含阈值过滤与可选剪枝）
- 断点续跑与分年持久化
- 进度可视化与日志

依赖：Python、Pandas、Jieba、NumPy、SciPy、scikit‑learn、tqdm

目录：

- `patent_quality/` 模块代码
- `artifacts/` 中间产物
- `tests/` 小样本数据与脚本
- `stat/` 统计分析脚本与图表/回归探索
- `docs/` 文档

参考：

- NBER w25266（Kelly 等）与 AEA Insights 论文
- 稀疏余弦相似度文档与实践

## 详细实现流程

该项目通过 `pipeline.py` 中的 `run_all` 函数串联，核心目标是基于文本内容计算每项专利的 **BS (Backward Similarity, 后向相似度)** 和 **FS (Forward Similarity, 前向相似度)**，并据此推导出质量评分（$Q = FS / BS$）。

### 1. 核心逻辑

- **输入**: 包含专利数据的 CSV 文件（需包含申请号、申请年份、标题、摘要、权利要求书等文本字段）。
- **算法**: 使用基于时间滑窗的 TF-IDF 文本相似度计算。IDF 的计算是“回顾性”的，即计算某年专利的特征权重时，仅利用截止到该年份的历史统计数据，避免“未来信息泄露”。
- **输出**: 一个包含每项专利 BS、FS 和 Quality_q 指标的 CSV 文件。

### 2. 阶段详解

#### 阶段 1: 构建词表与分年DF (Build Vocab & DF)

- **代码**: `vocab.py` -> `build_vocab`
- **目的**: 扫描所有数据，建立全局词表，并统计每个词在每一年的文档频率 (DF)。这是为了后续计算 TF-IDF 做准备。
- **输入**: 原始专利 CSV 数据。
- **处理**:
  1. 筛选“发明授权”类型的专利。
  2. 对文本（标题+摘要+权利要求）进行分词（Jieba 分词 + 停用词）。文本拼接分隔符可通过 `Config.text_sep` 配置。
  3. 统计全局 DF 和分年 DF，并记录每年文档数。
- **输出**:
  - `artifacts/df/global_df.json`: 全局词汇文档频率统计。
  - `artifacts/df/term_df_year={YYYY}.json`: 分年的词汇统计（用于后续回顾性 IDF 计算）。
  - `artifacts/stats/docs_per_year.csv`: 每年文档数量统计。

#### 阶段 2: 准备分年 Tokens (Prepare Tokens)

- **代码**: `vectorizer.py` -> `prepare_tokens`
- **目的**: 将所有专利文本进行分词并按年份存储到磁盘，避免后续反复读取原始 CSV 和重复分词，提高后续阶段效率。
- **输入**: 原始专利 CSV 数据。
- **处理**:
  1. 读取原始数据并筛选“发明授权”。
  2. 分词并过滤停用词。
  3. 提取辅助信息（`extra_cols` 配置列，如申请人/地址等）。
- **输出**:
  - `artifacts/tokens/year={YYYY}.jsonl`: 按年存储的 JSONL 文件，每行包含专利 ID、标题、分词后的 Token 列表及辅助信息。

#### 阶段 3: 回顾性 TF-IDF 向量化 (Retrospective Vectorization)

- **代码**: `vectorizer.py` -> `vectorize_by_year`
- **目的**: 将分词后的文档转换为稀疏向量。
- **关键点**：使用“回顾性 IDF”，即计算某一年文档的向量时，仅使用该年及其之前的历史统计数据，避免未来信息泄露。
- **输入**:
  - 阶段 2 的 Token 文件 (`tokens/*.jsonl`)。
  - 阶段 1 的 DF 统计文件 (`df/*.json`)。
- **处理**:
  1. 按年份从小到大遍历。
  2. 累加历史 DF 和文档总数，计算当前 IDF。
  3. 计算 TF-IDF 权重并进行 L2 归一化。
- **输出**:
  - `artifacts/vectors/year={YYYY}.npz`: 该年专利向量稀疏矩阵。
  - `artifacts/index/year={YYYY}.csv`: 行索引映射（专利号、标题与额外列）。

#### 阶段 4: 向量剪枝 (Vector Pruning)

- **代码**: `similarity.py` -> `compute_bs_fs`
- **目的**: 降维去噪，提升倒排相似度计算效率。
- **输入**: 阶段 3 的向量文件 (`vectors/*.npz`) 与分年 DF。
- **处理**:
  1. 删除手工停用词（`manual_stopwords_path`）。
  2. 删除高 DF 比例词（`df_ratio_threshold`）。
  3. 删除每年 Top DF 百分比词（`top_df_percent`）。
  4. 每文档仅保留 TopK 权重词（`topk_terms_per_doc`）。
- **输出**:
  - `artifacts/vectors_filtered/year={YYYY}.npz`: 剪枝后的向量。

#### 阶段 5: 计算 BS/FS (Compute Similarity)

- **目的**: 计算每项专利的后向相似度 (BS) 和前向相似度 (FS)。
  - **BS**: 与过去 `window_size` 年（如5年）内专利的相似度总和。
  - **FS**: 与未来 `window_size` 年（如5年）内专利的相似度总和。
- **输入**: 阶段 3/4 的向量文件（默认使用剪枝后的 `vectors_filtered`）。
- **处理**:
  1. 为每年构建倒排索引（postings）。
  2. 逐年份对计算相似度贡献（`pair_contrib`），只累加超过阈值的相似度（`similarity_threshold`）。
  3. 汇总得到每年的 BS/FS。
- **输出**:
  - `artifacts/postings/`: 倒排索引缓存。
  - `artifacts/pair_contrib/`: 年份对贡献缓存。
  - `artifacts/pair_list.json`: 年份对清单。
  - `artifacts/stats/bsfs_year={YYYY}.csv`: 该年每项专利的 BS/FS。

#### 阶段 6: 生成最终 CSV (Assemble Final CSV)

- **代码**: `quality.py` -> `assemble_final_csv`
- **目的**: 将计算好的指标与专利元数据合并，输出最终结果。
- **输入**:
  - 阶段 4 的 BS/FS 统计 (`stats/bsfs_year=*.csv`)。
  - 阶段 3 的索引信息 (`index/year=*.csv`)。
- **处理**:
  1. 读取每年的索引和计算结果。
  2. 计算质量商数 $Q = FS / (BS + \epsilon)$。
  3. 拼接申请号、标题、BS、FS、Q 以及配置中指定的额外列（如申请人）。
- **输出**:
  - `artifacts/patent_quality_output.csv`: 最终交付的 CSV 文件。

### 总结

整个流水线是一个**线性依赖**的过程，但支持**断点续跑**（`skip_if_exists=True`）。如果中间某个阶段的数据发生了变化（如词表变了），代码中有逻辑触发 `cascade_reset`，强制重跑后续所有依赖阶段，确保数据一致性。

## 统计与诊断工具（stat/）

该目录提供对中间结果的质量检查与统计分析脚本，典型用途包括：向量稀疏度评估、年份词汇演化、DF 相关性统计、TopK 截断影响评估等。

- `calc_avg_vocab_usage.py`: 统计每年平均非零词汇数（评估稀疏度）。
- `calc_df_pair_sum.py`: 计算年份对 DF 乘积和（快速评估词汇重叠）。
- `calc_topk_df_pair_sum.py`: 模拟 TopK 截断后的 DF 与能量覆盖率统计。
- `calc_yearly_top_vocab.py`: 每年 TopK 高频词汇统计。
- `calc_yearly_vocab_size.py`: 统计每年实际使用的词汇量规模。

此外，`stat/graph/` 和 `stat/公司财务/` 包含可视化与公司层面的探索性分析笔记本，用于扩展性研究与结果展示。

## 辅助脚本

- `run_full.py` / `run_full_30years.py`: 全量运行的参数模板与入口。
- `verify_ir.py`: 抽样年份的 IR 计算验证与性能剖析。
- `profile_matrix.py`: 单年向量稀疏度与分布统计。
