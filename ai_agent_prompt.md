# AI Agent Prompt: 中国专利质量指标 (Kelly et al., 2018) 基准复现任务书

**角色设定：** 资深数据科学家 & NLP 算法工程师
**任务目标：** 基于 **96GB** 的中国专利数据（1985-2025年），复现论文 *"Measuring Technological Innovation Over the Long Run" (Kelly et al., 2018)* 中的专利质量指标 ($q$)。
**编程语言：** Python (需使用 Pandas, Jieba, Scipy, Numpy, Scikit-learn 等库)。

## 1. 核心任务定义

计算专利质量指标 $q_j$：

$$
q_j = \frac{\text{Forward Similarity (FS)}}{\text{Backward Similarity (BS)}}
$$

* **指标含义**：
  * **BS (Backward Similarity)**：衡量专利与其**过去5年**专利的相似度（越低代表越新颖）。
  * **FS (Forward Similarity)**：衡量专利与其**未来5年**专利的相似度（越高代表影响力越大）。
* **计算基础**：文本相似度基于 **TF-BIDF** (词频-回顾性逆文档频率) 向量计算。

### 核心挑战与约束 (Critical Constraints)

* **数据规模**：原始 CSV 数据量约为 **96GB**。
* **内存红线**：**严禁**一次性将全量数据读入内存。必须采用**流式/分块 (Chunking)** 处理。
* **性能要求**：
  * NLP 分词可以考虑使用 **多进程 (Multiprocessing)** 并行加速。
  * 向量存储考虑使用 **稀疏矩阵 (Sparse Matrix)**。
  * 相似度计算考虑使用 **矩阵乘法 (Matrix Multiplication)** 结合 **批处理 (Batching)**，禁止双重循环。
  * 不允许出现 $O(N^2)$ 的空间复杂度。


---

## 2. 配置优先原则 (Configuration First)

代码 **必须** 在最开头定义 `Config` 类，包含以下参数，方便我进行实验调整：

* **路径配置**：
  * `data_path`: 原始数据路径（支持文件夹或单文件）。
  * `stopword_paths`: 停用词表文件路径列表。
  * `user_dict_path`: 搜狗/行业词库路径。
* **字段映射 (Schema)**：
  * `col_id`: '申请号' (唯一标识)
  * `col_date`: '申请年份' (时间基准)
  * `col_type`: '专利类型' (筛选依据)
  * `col_text_parts`: ['专利名称', '摘要文本', '主权项内容'] (用于拼接文本)
* **NLP 参数**：
  * `min_term_count`: 20 (全局过滤低频词)
  * `max_doc_freq_ratio`: 0.5 (过滤高频通用词)
* **算法参数**：
  * `window_size`: 5 (前后5年窗口)
  * `similarity_threshold`: 0.05 (稀疏矩阵截断阈值，低于此值视为0)

其他开发过程中你认为有必要的配置也可以放在里面

---

## 3. 数据清洗与预处理 (Data Cleaning)

### A. 筛选与去重 (Filter & Deduplicate)

1. **类型筛选**：只保留 `专利类型 == '发明授权'` 的记录。剔除实用新型、外观设计及未授权申请。
2. **去重**：基于 `申请号` 去重，确保每项专利唯一。
3. **时间基准**：
   * **严格使用 `申请年份`**。
   * 如果 `申请年份` 缺失，尝试从 `申请日` 解析；如果仍缺失，丢弃该条记录。

### B. 文本构建 (Document Construction)

1. **拼接**：`text_content = 专利名称 + 摘要文本 + 主权项内容`。
2. **填充**：缺失的文本字段用空字符串替代。

### C. NLP 处理 (中文特化)

1. **分词**：使用 `jieba`，必须加载 `user_dict_path`。
2. **停用词构建**：
   * 加载并合并所有外部停用词表。
   * **强制添加**以下专利专用停用词列表到停用词集合中（去除专利八股文干扰）：
     ```python
     ['本发明', '实用新型', '外观设计', '权利要求', '特征', '涉及', '公开', 
      '提供', '实施例', '所述', '其中', '及其', '用于', '一种', '方法', 
      '装置', '系统', '组件', '结构', '步骤', '属于', '技术领域', 
      '背景技术', '优选', '位于', '连接', '附图', '示意图', '包含', 
      '包括', '使得', '能够', '相比', '现有技术', '其特征在于', '由...组成']
     ```
3. **清洗**：去除停用词、非中文字符、数字和标点。

---

## 4. 核心算法实现 (Core Algorithm)

### 步骤 A: 动态词表构建 (Dynamic Vocabulary)

* 扫描全量数据。
* 执行 `min_term_count` 和 `max_doc_freq_ratio` 过滤。
* 生成最终的词汇表 (Vocabulary)。

### 步骤 B: TF-BIDF 向量化 (The "Backward" Innovation)

* **核心逻辑**：计算 $T$ 年专利的 IDF 权重时，**严禁使用 $T$ 年之后的数据**（模拟当时的信息环境，避免后见之明）。
* **实现建议**：
  * 按年份顺序处理 (1985 -> 2025)。
  * 维护一个累积的 `global_term_counts` (每个词在历史文档中出现的文档数) 和 `total_docs_so_far`。
  * IDF 计算公式：$IDF_{t}(w) = \log(\frac{Total\_Docs_{<t}}{1 + Doc\_Count_{<t}(w)})$
  * 生成稀疏矩阵 (Sparse Matrix) 格式的向量。

### 步骤 C: 滑动窗口相似度计算 (Window Similarity)

对于每一年 $T$ 的目标专利 $j$：

1. **定义窗口**：

   * **Backward Pool**: 年份 $[T - window\_size, T - 1]$ 的所有专利。
   * **Forward Pool**: 年份 $[T + 1, T + window\_size]$ 的所有专利。
2. **计算 BS (Backward Similarity)**：

   * 计算专利 $j$ 与 Backward Pool 中所有专利向量的余弦相似度。
   * `BS_j = sum(similarities)` (注意应用 `similarity_threshold` 过滤噪音)。
3. **计算 FS (Forward Similarity)**：

   * 计算专利 $j$ 与 Forward Pool 中所有专利向量的余弦相似度。
   * `FS_j = sum(similarities)`。

### 步骤 D: 指标计算

* $q_j = FS_j / BS_j$
* *(注意处理分母为0的情况，通常加一个极小值 epsilon)*

---

## 5. 输出交付 (Deliverables)

请编写完整的 Python 代码，最终生成一个 CSV 文件，至少包含以下列：

* `申请号`
* `申请年份`
* 专利名称
* `BS` (Backward Similarity, 衡量新颖性)
* `FS` (Forward Similarity, 衡量影响力)
* `Quality_q` (最终质量指标)

其他你认为有必要的（让我确认是什么专利，或者别的重要中间结果），或者别的你认为应该输出的结果

**代码技术要求：**

1. **模块化设计**：数据加载、NLP预处理、向量化、相似度计算需分装为不同函数。
2. **内存与性能优化**：
   * 必须使用 `scipy.sparse` 存储 TF-BIDF 矩阵。
   * 计算相似度时，建议使用矩阵乘法（Matrix Multiplication）批量计算，或者分批次（Batch）处理，避免内存溢出。
3. **进度可视化**：关键步骤（如分词、向量化、相似度计算）需使用 `tqdm` 显示进度条。
4. 大型计算最好有保存中间结果，防止终止后重试花费大量时间
5. 因为最终的数据量很大，专利数据有96GB，因此过程中请适当考虑优化时间/空间性能，不过第一次以跑通全流程为第一目标
