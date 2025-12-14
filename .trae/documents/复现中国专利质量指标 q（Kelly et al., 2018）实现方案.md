## 目标与交付物
- 目标：在不越过内存红线的前提下，基于中国专利（1985–2025）复现 Kelly 等提出的专利质量指标 `q_j = FS_j / BS_j`，严格执行“回顾性 IDF（TF‑BIDF）”与 5 年前后滑窗的定义。
- 交付物：
  - `CSV`：包含 `申请号`、`申请年份`、`专利名称`、`BS`、`FS`、`Quality_q`，以及必要索引字段（如行号/年份）
  - 可复运行代码（模块化，含进度条与断点续跑）
  - 文档：使用说明、配置说明、验证与性能报告模板
  - 中间产物缓存：分年 `稀疏向量矩阵（.npz）`、分年索引映射（.csv/.json）、词表与词频统计

## 规范依据与参考
- 质量度量思想与 5 年滑窗：Kelly et al., 2018/2021（Forward vs Backward Similarity，5 年窗口）[来源：NBER w25266 https://www.nber.org/system/files/working_papers/w25266/w25266.pdf；AEA Insights 2021 https://www.aeaweb.org/articles?id=10.1257%2Faeri.20190499]
- 稀疏余弦与矩阵乘法实现：`sklearn.metrics.pairwise.cosine_similarity` 支持稀疏输入；稀疏归一化 + 稀疏乘法更高效 [来源：scikit‑learn 文档 https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html；实践经验与案例：StackOverflow 讨论 https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat]
- 中文分词与词典：`jieba` 官方文档与 `load_userdict`；并行分词官方实现不支持 Windows，需自建进程池方案 [来源：jieba GitHub https://github.com/fxsjy/jieba]
- 中文停用词库：HIT/百度等合集可用作外部停用词来源 [来源：GitHub stopwords 合集 https://github.com/goto456/stopwords]

## 总体技术路线
- 两阶段流式管线，严格按年份顺序：
  1) 词表与文档频率统计（全量扫描 + 分年 DF，构建“最终词汇表”）
  2) 回顾性向量化与相似度计算（按年生成 TF‑BIDF 稀疏矩阵；分年块状稀疏乘法得到 BS/FS）
- 中间结果持久化与断点续跑：分年 `tokens`、分年 `CSR 矩阵`、分年 `index`、全局/分年 `DF` 与 `cumulative DF`、配置快照
- 大规模计算策略：稀疏矩阵、批量化/块乘、行向量归一化、阈值截断、`float32`、按年加载与释放

## 目录与模块设计
- `config.py`：`Config` 类集中所有可调参数（路径/Schema/NLP/算法）
- `data_loader.py`：分块读取 CSV/目录，类型筛选、去重、年份解析、文本拼接
- `nlp.py`：分词（加载用户词典）、停用词并集构建、清洗（去停用/非中文/数字/标点）
- `vocab.py`：
  - 第一阶段：流式统计全局 DF 与分年 DF（`min_term_count`/`max_doc_freq_ratio`）
  - 生成最终词表，保留词→列索引映射
- `vectorizer.py`：
  - 第二阶段：按年份顺序向量化（仅用历史 `<T` 的 DF/DocCount 计算 `IDF_t(w) = log(Total_Docs_<t / (1 + Doc_Count_<t(w)))`）
  - 产出分年 `CSR`，行 L2 归一化，保存 `.npz`
- `similarity.py`：
  - 分年块乘：`M_T` × `M_Y.T`（`Y ∈ [T−5, T−1]` 得到 BS；`Y ∈ [T+1, T+5]` 得到 FS）
  - 阈值截断：对乘积结果的 `data` 应用 `similarity_threshold`
  - 行求和得到每个专利的 BS/FS；合并跨年的累计值
- `quality.py`：`q = FS / (BS + epsilon)`，边界处理与结果汇总
- `io_utils.py`：持久化与加载（索引、词表、矩阵、统计）、断点续跑元数据
- `pipeline.py`：串联全流程，含进度条 `tqdm`、日志与异常重试
- `docs/`：`README.md`、`USAGE.md`、`REPORT_TEMPLATE.md`
- `tests/`：采样级单元/集成测试（小样本）

## 关键实现细节（对齐任务书）
- 配置（强制放在最前）：
  - 路径：`data_path`、`stopword_paths`、`user_dict_path`
  - Schema：`col_id`、`col_date`、`col_type`、`col_text_parts`
  - NLP：`min_term_count`、`max_doc_freq_ratio`
  - 算法：`window_size=5`、`similarity_threshold`、`epsilon`、`dtype`
- 数据清洗：
  - 保留 `专利类型=='发明授权'`，按 `申请号` 去重
  - 年份解析优先 `申请年份`，缺失则尝试 `申请日`→年；均缺失则丢弃
  - 文本拼接：`专利名称 + 摘要文本 + 主权项内容`，缺失填空串
- 中文分词与停用词：
  - `jieba` 分词，加载 `user_dict_path`
  - 停用词：外部停用词并集 + 专利领域停用词白名单（任务书列表）
  - 清洗：移除停用词、非中文、数字、标点
  - 并行策略：Windows 环境不使用 `jieba` 官方并行；采用 `ProcessPoolExecutor` 自建 worker，在进程初始化中 `jieba.initialize()` 与 `load_userdict`，批量分词
- 动态词表：
  - 第一阶段分年统计 DF：`term_df[year][term] += 1`，`docs_per_year[year] += 1`
  - 全局 `global_df[term] = Σ_year term_df[year][term]`
  - 过滤：`global_df[term] >= min_term_count` 且 `global_df[term]/TotalDocs <= max_doc_freq_ratio`
- 回顾性 TF‑BIDF：
  - 按年处理（1985→2025）：维护 `cumulative_df_before_year[term]` 与 `total_docs_so_far`
  - `IDF_t(w) = log(total_docs_so_far / (1 + cumulative_df_before_year[w]))`
  - 每专利 TF * IDF 生成 `CSR` 行；全矩阵行 L2 归一化；`astype(float32)`
- 滑窗相似度（矩阵乘法批量计算）：
  - `BS(T) = Σ_{Y∈[T−5,T−1]} sum_rows( clip(M_T × M_Y^T, thr) )`
  - `FS(T) = Σ_{Y∈[T+1,T+5]} sum_rows( clip(M_T × M_Y^T, thr) )`
  - 阈值：对乘积稀疏矩阵的 `data` 进行原位截断并 `eliminate_zeros()`
- 指标计算：`q = FS / (BS + epsilon)`；汇总到 `CSV`

## 性能与内存策略
- 流式分块：`pandas.read_csv(chunksize=...)`，`usecols` 与合理 `dtype` 降内存
- 稀疏存储：`scipy.sparse.csr_matrix`；分年 `.npz` 持久化，按需加载/释放
- 批次与块乘：跨年分块（最多 10 年窗口乘），避免一次性构建超大相似度矩阵
- 行归一化：使用 `sklearn.preprocessing.normalize(X, norm='l2', copy=False)` 后用稀疏乘法近似余弦
- 阈值截断：将小于 `similarity_threshold` 的乘积项置零，降低累加噪声与内存
- 精度与类型：优先 `float32`，在最终汇总时转 `float64`，兼顾速度与稳定性

## 中间结果与断点续跑
- `artifacts/`
  - `vocab/final_vocab.json`、`df/global_df.json`、`df/term_df_year=YYYY.json`
  - `tokens/year=YYYY.parquet`（可选）
  - `vectors/year=YYYY.npz`（CSR）与 `index/year=YYYY.csv`（行→`申请号` 映射）
  - `stats/year=YYYY.json`（行数、非零计数、窗口进度）
  - `checkpoint.json`（已完成年份、下一步计划）
- 失败恢复：读取 `checkpoint`，从上次完成的年份继续；保证幂等（重复年份覆盖同名文件）

## 验证与测试方案
- 正确性：
  - 小样本（数百条、3–4 年）端到端运行，人工抽查 BS/FS 与 q 边界/稳定性
  - 回顾性约束校验：断言 `IDF_t` 未使用 `≥t` 年数据（通过累计 DF 记录对比）
- 性能：
  - 记录每年向量化耗时、相似度块乘耗时、峰值内存（可用 `psutil`）
  - 观察 `similarity_threshold` 与非零密度的关系
- 复现实验：
  - 对比不同停用词集合、不同 `min_term_count`/`max_doc_freq_ratio` 对指标的敏感性

## 风险与应对
- Windows 并行分词：官方并行不支持 → 自建进程池，分批、进程初始化加载词典
- 编码与脏数据：优先 `utf-8`，回退 `gb18030`；异常行隔离到错误日志
- 超大年份差异：为空年份窗口跳过；确保滑窗边界处理（首尾年份）
- 稀疏乘法内存峰值：分年加载/释放、逐年累加 BS/FS，不保留全相似度矩阵
- BS≈0 的稳定性：使用 `epsilon`（如 `1e-8`）保障数值稳定

## 文档与使用说明
- `README.md`：概览、方法与假设、依赖、安装、目录结构
- `USAGE.md`：配置项说明、运行与断点续跑、常见问题（并行/编码/停用词）
- `REPORT_TEMPLATE.md`：执行报告模板（下）

## 执行报告模板（交付后填写）
- 任务完成情况：哪些步骤完成与验证、产出文件列表
- 问题与解决方案：编码异常、稀疏乘法峰值、年份缺失等的处理策略
- 性能测试结果：每年耗时、总耗时、峰值内存、非零密度、阈值敏感性
- 改进建议：
  - 语义增强（词向量/句向量）与相似度替代（限定仍遵守回顾性约束）
  - 更高效的持久化（Parquet/Arrow）、向量压缩（量化/切片）
  - 断点续跑粒度更细（按月/批次）、更丰富日志

## 下一步
- 获得确认后：按上述模块实现，先在小样本验证流程与性能，再在全量/分年批次上运行，最终生成 CSV 与完整执行报告。