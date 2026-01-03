# 新任务提示词：在 Pipeline 中新增「向量剪枝阶段」

> **用途** ：本文件用于指导 AI Agent（代码助手）在现有专利指标 pipeline 中实现一个新的工程阶段。
>
> **目标读者** ：负责直接修改代码的 AI / 工程助手。

---

## 一、任务背景与总体目标

当前 pipeline 在 **阶段3（分年 TF-BIDF 向量化）** 与 **阶段4（BS / FS 相似度计算）** 之间，存在严重的计算瓶颈：

* 单年文档规模可达 10–50 万
* 高频模板词（如“中 / 设置 / 之间 / 技术 / 领域”等）导致候选对爆炸
* 稀疏矩阵乘法 `S = M_T · M_Y^T` 的 `nnz` 与时间成本不可接受

### 总体目标

在 **阶段3 与原阶段4 之间** 新增一个新的工程阶段：

> **阶段4：向量剪枝阶段（Vector Pruning Stage）**

该阶段对阶段3输出的分年向量（`.npz` CSR 矩阵）进行  **三类剪枝** ，显著减少后续相似度计算的候选规模，同时尽量不改变论文指标含义。

随后：

* 原阶段4 → 顺延为 **阶段5**
* 原阶段5 → 顺延为 **阶段6**

并保持 pipeline 的 **日志风格 / checkpoint / cascade reset 逻辑** 一致。

---

## 二、新阶段的输入与输出

### 输入

* 来自阶段3的分年向量文件：

```
artifacts/.../vectors/year=YYYY.npz
```

* 每个文件是一个 CSR 稀疏矩阵：
  * 行：专利文档
  * 列：词表索引
  * 值：TF-BIDF 权重

### 输出

* 新目录：

```
artifacts/.../vectors_filtered/year=YYYY.npz
```

* 同样是 CSR 稀疏矩阵
* `nnz` 应显著减少
* 后续所有相似度计算 **只使用 vectors_filtered**

---

## 三、阶段4需要执行的三类剪枝（必须按顺序）

> ⚠️ 三个步骤都基于 `.npz` 稀疏矩阵操作，不允许 densify。

### Step 4.1：手工停用词剪枝（Manual Stopwords）

#### 输入

* 文件路径（已存在）：

```
stopword/专利停用词表.txt
```

#### 要求

* 读取停用词列表，**并对词表去重**
* 使用阶段1已有的 **词 → 列索引映射**
* 对每一年矩阵 `M_y`：
  * 找到对应列
  * 将这些列的值全部置 0
  * 调用 `eliminate_zeros()`

#### 日志要求（每年）

* 年份
* 本步删除的词数
* 矩阵 `shape`
* `nnz`（before / after）

---

### Step 4.2：方法1 —— 按年高 df 词剪枝（Year-wise DF Pruning）

#### 定义

* `N_y`：该年文档数（矩阵行数）
* `df_y(w)`：该年包含词 `w` 的文档数
* `V_y`：该年出现过的词总数（`df_y(w) > 0`）

#### 剪枝规则（必须参数化）

1. **硬阈值规则**

删除所有满足：

```
df_y(w) / N_y >= df_ratio_threshold
```

* 默认：`df_ratio_threshold = 0.20`
* 参数必须写入 config

2. **Top 百分比规则（在剩余词中）**

* 按 `df_y(w)` 降序排序
* 删除 top `top_df_percent`
* 百分比基数：**当年词表大小 `V_y`，不是 40 年总词表**
* 推荐默认值：
  * `top_df_percent = 0.002 ~ 0.005`

#### 实现要求

* 对矩阵列执行置零
* 再调用 `eliminate_zeros()`

#### 日志要求（每年）

* 年份
* `N_y`, `V_y`
* 删除的列数：
  * 因 df_ratio 删除多少
  * 因 top_percent 删除多少
* `nnz`（before / after）

---

### Step 4.3：方法2 —— 文档级 Top-K 剪枝（Top-K per Document）

#### 定义

* 对每一行（每篇专利）：
  * 只保留权重最大的 K 个词
  * 其余词全部置零

#### 参数

* `topk_terms_per_doc`（必须在 config 中）
* 推荐默认：`K = 30`

#### 实现约束

* 必须在稀疏结构上完成
* 不允许 densify
* 可使用：
  * 行内 `data/indices/indptr`
  * `argpartition` 或局部选择

#### 日志要求（每年）

* 年份
* K 值
* 行数
* `nnz`（before / after）
* （可选）平均每行 nnz 变化

---

## 四、Pipeline 集成要求

### 1. 阶段顺延

* 新阶段：**阶段4（向量剪枝）**
* 原阶段4（BS/FS）→ **阶段5**
* 原阶段5（最终 CSV）→ **阶段6**

### 2. 向量加载路径修改

* 原 BS/FS 计算默认加载：

```
artifacts/.../vectors/
```

* 必须修改为可配置：

```
artifacts/.../vectors_filtered/
```

* 推荐做法：
  * `cfg.vectors_dir`
  * 或 `cfg.use_filtered_vectors = True`

### 3. Checkpoint 与 Cascade Reset

必须遵循现有 pipeline 风格：

* 新增 checkpoint key：

```
vectors_pruned_years   或   vector_pruning_done
```

* 若阶段1 / 2 / 3 / 新阶段4 任意重跑：
  * 后续阶段（剪枝 / BSFS / CSV）全部 reset

### 4. Checkpoint 粒度（建议）

* 推荐：**按年份记录剪枝完成状态**
* 允许中断恢复，避免重跑全部年份

---

## 五、Config 中需要新增的参数

```python
manual_stopwords_path = "stopword/专利停用词表.txt"
df_ratio_threshold = 0.20
top_df_percent = 0.002   # 或 0.005
topk_terms_per_doc = 30
vectors_filtered_dir = "artifacts/.../vectors_filtered"
use_vectors_filtered_for_bsfs = True
```

---

## 六、日志规范（强制）

* 每个阶段、每个子步骤必须输出：
  * 当前阶段名
  * 当前年份
  * 用时
  * 关键统计量（nnz、列数、行数等）
* 阶段级别：输出总耗时

---

## 七、验收标准（自检清单）

1. Pipeline 从头运行：
   * 阶段3 → 阶段4（剪枝）→ 阶段5（BS/FS）→ 阶段6（CSV）
2. `vectors_filtered/year=YYYY.npz`：
   * 可被 `scipy.sparse.load_npz` 正常加载
   * nnz 显著下降
3. 若阶段3重跑：
   * 剪枝、BS/FS、CSV checkpoint 被正确 reset
4. 在典型大年份对（如 2008×2012）：
   * `S.nnz` 与运行时间显著下降

---

## 八、总结

该阶段的目标不是“微调矩阵乘法”，而是 **系统性地删除几乎只贡献计算、不贡献信号的词与项** 。

这是复现大规模跨年专利相似度论文时 **必需的隐含工程步骤** ，也是整个 pipeline 能否在单机或有限资源上跑通的关键。
