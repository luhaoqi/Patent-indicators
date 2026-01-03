# 统计工具脚本说明

本目录包含了一系列用于分析专利指标中间结果的 Python 脚本。这些脚本主要用于验证数据质量、统计词汇分布以及计算不同年份之间的相似性指标。

## 脚本列表

### 1. `calc_avg_vocab_usage.py`

**作用**：统计每一年份所有文档中，平均每个文档包含多少个非零词汇（词表中的词）。

* **输入**：`vectors/year=*.npz` (稀疏矩阵)
* **输出**：控制台打印每年的平均非零词汇数。
* **用途**：检查向量的稀疏程度，验证分词和去停用词的效果。
* **用法**：
  ```bash
  python stat/calc_avg_vocab_usage.py artifacts_full/vectors
  ```

### 2. `calc_df_pair_sum.py`

**作用**：计算任意两年份（往前往后各5年窗口内）的词汇文档频率（DF）乘积之和。

* **公式**：$\sum_{term} (DF_{year1, term} \times DF_{year2, term})$
* **输入**：`df/term_df_year=*.json` (预计算的 DF 统计文件)
* **输出**：控制台打印满足条件的年份对的统计值。
* **用途**：快速估算年份之间的词汇重叠程度和相似性趋势（基于全局 DF，不涉及文档级 TopK）。
* **用法**：
  ```bash
  python stat/calc_df_pair_sum.py artifacts_full
  ```

### 3. `calc_topk_df_pair_sum.py`

**作用**：**核心统计脚本**。模拟“每个文档只保留 TopK 关键词”后的数据特征。

1. **L2 能量统计**：计算每一年所有文档在只保留 TopK 权重后，剩余权重的平均和以及平方和（L2能量覆盖率）。
2. **Pairwise DF Sum**：在对每个文档进行 TopK 过滤后，重新计算 DF，并计算年份对的 DF 乘积之和。

* **输入**：`vectors/year=*.npz` (原始向量矩阵)
* **输出**：文本文件（如 `result_topk_10.txt`），包含年度权重统计和年份对统计。
* **用途**：评估 TopK 截断对信息量的影响，以及在不同 K 值下的年份相似性变化。
* **用法**：
  ```bash
  # K=10, 输出到 result_topk_10.txt
  python stat/calc_topk_df_pair_sum.py --dir artifacts_full --k 10 --output stat/result_topk_10.txt
  ```

### 4. `calc_yearly_top_vocab.py`

**作用**：列出每一年份出现频率最高的前 K 个词汇。

* **统计口径**：文档频率（Document Frequency），即该词在多少个文档中出现过（每个文档只计一次）。
* **输入**：`df/term_df_year=*.json`
* **输出**：文本文件（如 `yearly_top_vocab.txt`），包含每年的 TopK 词汇列表、文档数及占比。
* **用途**：观察每年的热门技术词汇演变，检查是否有无意义的高频词（停用词遗漏）。
* **用法**：
  ```bash
  python stat/calc_yearly_top_vocab.py --dir artifacts_full --k 50 --output stat/yearly_top_vocab.txt
  ```

## 目录结构说明

* `stat/`: 存放上述脚本及生成的统计结果文件（*.txt）。
* `artifacts_full/`: (外部依赖) 存放流水线生成的中间数据，包括 `vectors` (向量), `df` (词频), `vocab` (词表) 等。

## 常见参数说明

* `--dir`: 指定 `artifacts` 根目录路径（默认为 `artifacts_full`）。
* `--k`: 指定 TopK 的数量（默认为 10 或 50）。
* `--output`: 指定输出文件路径。

可以用来构建停用词：


### Rule 1：

> **如果一个词不能回答「这是哪类技术」→ 候选删除**

### Rule 2：

> **如果一个词只描述“动作 / 效果 / 状态 / 法律结构”→ 删除**

### Rule 3：

> **如果一个词指向“物理实体 / 材料 / 化学对象 / 工程部件”→ 保留**

介于 2 和 3 之间的 → **条件保留**
