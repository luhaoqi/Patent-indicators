## 变更概览

* 在阶段3（向量化）与原阶段4（BS/FS）之间新增阶段4：向量剪枝（Vector Pruning）。

* 原阶段4（BS/FS）顺延为阶段5，原阶段5（CSV）顺延为阶段6。

* 新增模块用于对每年 `.npz` CSR 向量执行三类剪枝：手工停用词、按年高 DF、文档级 Top-K。

* 修改配置以支持剪枝参数和向量目录切换；修改相似度计算默认读取 `vectors_filtered`。

* 扩展 checkpoint 与级联重置，支持按年份断点续跑与正确 reset。

## 新增模块

* 新文件：`patent_quality/pruning.py`

  * 函数：`prune_vectors_by_year(cfg: Config) -> None`

  * 入口：遍历 `artifacts/vectors/year=YYYY.npz`，按顺序执行 Step 4.1 → 4.2 → 4.3，输出到 `artifacts/vectors_filtered/year=YYYY.npz`。

  * 使用词表映射与每年 DF 文件：

    * 词表：`artifacts/vocab/final_vocab.json`（键=词，值=列索引），参见 [vocab.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/vocab.py#L96-L109)

    * 每年 DF：`artifacts/df/term_df_year=YYYY.json`，参见 [vectorizer.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/vectorizer.py#L130-L134)

## 剪枝实现细节

* 数据结构约束：全程保持稀疏，不 densify；基于 CSR 的 `data/indices/indptr` 原地置零后 `eliminate_zeros()`。

* Step 4.1 手工停用词剪枝

  * 载入 `cfg.manual_stopwords_path`（默认 `stopword/专利停用词表.txt`），先对这个文件中的重复词汇去重，然后对里面的词汇检查现有的词表，明确这些词汇对应的列是哪些（如果现在用到的词表中不包含里面的某些词汇，则忽略这些词汇），最终需要映射为列索引集合。

  * 对每年矩阵 `M_y`：逐行切片 `indices`，将命中列的 `data` 置零；调用 `eliminate_zeros()`。

  * 日志：年份、删除词数、shape、nnz before/after。

* Step 4.2 按年高 DF 剪枝

  * 载入当年 `df_y(w)` 与 `N_y`。

  * 规则1：删除满足 `df_y(w)/N_y >= cfg.df_ratio_threshold` 的列（默认 0.20）。

  * 规则2：在剩余词中，按 `df_y(w)` 降序删除 Top `cfg.top_df_percent * V_y` 的列（默认 0.002\~0.005）。

  * 同样通过逐行置零并 `eliminate_zeros()`；记录每类删除列数与 nnz before/after。

* Step 4.3 文档级 Top-K 剪枝

  * 对每行，仅保留权重最大的 `cfg.topk_terms_per_doc`（默认 30）；使用 `argpartition` 在行片段上选择 K 个最大项，其余置零。

  * 日志：年份、K、行数、nnz before/after、可选平均每行 nnz 变化。

* 输出：保存到 `artifacts/vectors_filtered/year=YYYY.npz`，维持 CSR 结构与 dtype。

## Pipeline 改造

* 修改 [pipeline.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/pipeline.py):

  * 在阶段3之后插入“阶段4：向量剪枝”，调用 `prune_vectors_by_year(cfg)`。

  * 原“阶段4：计算BS/FS”改为“阶段5”；原“阶段5：生成最终CSV”改为“阶段6”。

  * 级联重置：

    * 若阶段1/2/3执行或被强制重跑，清除 `vectors_pruned` 与按年状态，并重置后续阶段。

    * 若阶段4（剪枝）执行或被重跑，清除 `bsfs_years` 与 `final_csv`。

  * checkpoint：新增 `vectors_pruned`（布尔）与 `vectors_pruned_years`（列表，按年断点）。阶段运行完成后标记。

## 配置参数新增

* 修改 [config.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/config.py):

  * `manual_stopwords_path: str = "stopword/专利停用词表.txt"`

  * `df_ratio_threshold: float = 0.20`

  * `top_df_percent: float = 0.002`

  * `topk_terms_per_doc: int = 30`

  * `vectors_filtered_dir: str = "vectors_filtered"`（位于 `artifacts` 下）

  * `use_vectors_filtered_for_bsfs: bool = True`

  * `ensure_dirs()` 增加创建 `vectors_filtered` 子目录。

## 相似度计算切换目录

* 修改 [similarity.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/similarity.py):

  * 新增目录解析：现在的目录需要使用 config中的 `vectors_filtered_dir`。

  * `_years_with_vectors` 与向量 `load_npz` 统一使用上述目录。

## 工具与清理

* 修改 [io\_utils.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/io_utils.py):

  * `clear_artifacts` 的 `subdirs` 增加 `vectors_filtered`，以支持全量清理。

  * 保持 `checkpoint.json` 结构，新增字段由 `save_checkpoint` 直接写入。

## 日志规范与统计

* 所有子步骤按年份输出：阶段名、年份、用时、nnz/行/列等关键统计；阶段级汇总总耗时。

* 复用现有日志工具 [log.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/log.py)。

## 验证与验收

* 自检运行链路：阶段3 → 阶段4（剪枝） → 阶段5（BS/FS） → 阶段6（CSV）。

* 验证 `vectors_filtered/year=YYYY.npz` 可被 `scipy.sparse.load_npz` 正常加载，且 nnz 显著下降。

* 重跑阶段3时，确认剪枝、BS/FS、CSV 的 checkpoint 被正确 reset。

* 在大年份对（如 2008×2012）进行抽样，比较 `S.nnz` 与运行时间显著下降。

## 兼容性与性能考虑

* 所有列操作在 CSR 上按行置零，避免 CSC 转换的额外开销；对非常多列的场景可在实现中选择性使用 `tocsr()`/`eliminate_zeros()` 组合。

* Top-K 使用 `argpartition` 在行片段上选择，时间复杂度线性近似，适合大规模稀疏矩阵。

* 保持 dtype 与规范化行为与当前向量化一致（参见 [vectorizer.py](file:///d:/BaiduNetdiskDownload/工具/专利指标/patent_quality/vectorizer.py#L210-L228)），不改变指标含义。

## 后续代码改动列表（摘要）

* 新增：`patent_quality/pruning.py`（三个剪枝步骤、按年处理与日志）。

* 修改：`config.py`（新增参数与目录创建）。

