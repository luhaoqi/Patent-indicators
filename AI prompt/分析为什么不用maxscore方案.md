# 一、背景与计算设定说明

## 1. 你现在在算什么？

以 **2011 年**为目标年，你在做两类计算：

* **Backward (BS)** ：

  2011 ← {2006, 2007, 2008, 2009, 2010}

* **Forward (FS)** ：

  2011 → {2012, 2013, 2014, 2015, 2016}

每一个 `(X, 2011)` 或 `(2011, Y)`，本质是：

> 对 X 年的每一篇专利文档 `i`，
>
> 计算它与 Y 年所有专利文档 `j` 的相似度，
>
> 只累加 `score(i,j) ≥ thr = 0.05` 的部分。

---

## 2. 当前使用的核心算法（不开 MaxScore）

你当前的 **不开 MaxScore** 算法是一个 **标准、非常干净的 IR / 倒排实现** ：

### 对每个查询文档 `i`：

1. 遍历它的 `n_terms ≈ 27~30` 个非零词
2. 对每个词 `w`：
   * 扫描 `postings[w]`（Y 年中包含该词的所有文档 j）
   * 做一次累加：
     <pre class="overflow-visible! px-0!" data-start="712" data-end="751"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>acc[j] += x_iw * y_jw
     </span></span></code></div></div></pre>
3. 最后筛选 `acc[j] ≥ thr`

这一步的 **真实计算成本** ，几乎完全由：

> **posting 累加次数（PostingOps）**

决定。

---

# 二、不开 MaxScore 时的真实工作量（你给出的数据）

我们直接用你 2011 年 backward 的几组日志来看（只列关键指标）。

---

## 1. 规模与工作量指标（非常关键）

以 `(2006, 2011)` 为例：

| 指标                           | 数值                  |
| ------------------------------ | --------------------- |
| X 年文档数 Nx                  | 58,597                |
| Y 年文档数 Ny                  | 167,885               |
| AvgTerms                       | 27.60                 |
| **Touched Avg**          | **11,822**      |
| **Touched P90**          | **16,549**      |
| **Touched Max**          | **27,233**      |
| **PostingOps Total**     | **813,824,896** |
| **PostingOps Avg / doc** | **13,888**      |
| KernelTime                     | **6.92 s**      |
| TotalTime                      | **7.59 s**      |

其他年份（2007–2010）的数值非常一致，只是 PostingOps 随 Nx 增大而线性上升。 07cbc38d-2829-4927-ac07-1cfa496…

---

## 2. 这些指标分别代表什么？（逐条解释）

### （1）Touched

**Touched = 对某个 X 文档 i，实际被访问到的 Y 文档 j 的数量**

直观理解：

> “因为共享词而被纳入候选的 Y 年文档数”

你这里的典型值是：

* Avg ≈ 12k
* P90 ≈ 17k
* Max ≈ 27k

这说明：

 **即便已经做过词表剪枝 + TopK，每篇专利仍会触达上万候选** 。

---

### （2）PostingOps（最重要）

**PostingOps = 实际执行 `acc[j] += ...` 的次数**

这是：

* CPU 真正干活的地方
* 与总时间 **几乎线性相关**

你这里的数量级是：

* 单 pair：**8e8 ～ 2e9 次**
* 单文档：**1.4 万次左右**

但注意一个非常关键的事实：

> **8 亿次 posting 累加只用了 ~7 秒**

这说明：

* 你的 Numba kernel
* 数据结构
* cache 行为

**已经非常高效了**

---

# 三、MaxScore 想解决什么问题？（理论目标）

## 1. MaxScore 的原始动机

MaxScore 是 IR / 搜索领域的经典剪枝技术，目标是：

> **尽早判定某些候选文档 j 不可能达到阈值 thr，从而避免后续 posting 累加**

核心思想是：

<pre class="overflow-visible! px-0!" data-start="2015" data-end="2077"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>acc</span><span>[j]</span><span> + remain_upper_bound < thr
→ j 不可能达阈 → 可以“提前判死”
</span></span></code></div></div></pre>

---

## 2. 理想情况下，MaxScore 能省什么？

**它只能省一件事：**

> 后续 term 的 posting 累加次数

也就是说，它的价值完全取决于：

* 能否 **足够早**
* 在 **posting 最长、最贵的阶段之前**
* 判死大量候选

---

# 四、结合你的数据：MaxScore 实际在干什么？

我们对比你之前给出的 **开 MaxScore 的 Diag 日志** 和现在  **不开 MaxScore 的日志** ，可以得到一个非常清晰的结论。

---

## 1. 不开 MaxScore 时的基线

* PostingOps：8e8 ～ 2e9
* KernelTime：5～10 秒 / pair
* 整体非常稳定、线性、可控

---

## 2. 开 MaxScore 时（你之前的数据）

你之前的 Diag 显示（以某 pair 为例）：

* **PrunedRatio Avg ≈ 0.84**
* **Remain 在 K10 仍 ≈ 0.5**
* thr = 0.05

这意味着什么？

---

## 3. 关键解释：为什么“剪了很多，但没省时间”

### （1）剪枝发生得 **太晚**

* 前 1 / 3 / 5 / 10 个 term（通常是 df 最大的词）：
  * `remain >> thr`
  * **必须完整扫 posting**
* 而这些 term：
  * posting 最长
  * 成本最高

👉 **最贵的工作已经做完了**

---

### （2）被剪掉的是“后半段便宜工作”

* 后面的 term：
  * posting 更短
  * 本来就不占主导成本
* 即使剪掉 80% 候选：
  * 省下的 posting_ops 占比很有限

---

### （3）但 MaxScore 本身有额外成本

即便你写在 Numba 里：

* 需要维护 `mark`
* 需要反复检查 `acc + remain`
* 需要额外的内存访问与分支

 **这些成本是“新增的”** 。

---

## 4. 所以结果是一个典型的工程现象

> **剪枝比例看起来很高（PrunedRatio 84%），
>
> 但真实工作量（PostingOps）几乎没减少，
>
> 反而多了一堆剪枝判断开销 → 总体变慢。**

---

# 五、关于你“不太懂的几个参数”的直观解释（重点）

## 1. `thr = 0.05`

* 是 **cosine similarity 阈值**
* 含义：
  > 只有技术上明显相关的专利对才被计入
  >
* 在论文和你当前数据规模下：**完全合理**

---

## 2. `remain`

* 是一个 **理论上界**
* 表示：
  > “在最乐观情况下，这个候选 j 还能再加多少分”
  >

如果：

<pre class="overflow-visible! px-0!" data-start="3315" data-end="3341"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>acc</span><span> + remain < thr
</span></span></code></div></div></pre>

才允许剪枝。

你这里的问题是：

* remain 初始值 ≈ 2
* 到 K10 仍 ≈ 0.5
* thr 只有 0.05

👉 **上界下降得太慢**

---

## 3. `Touched`

* 是“共享至少一个词的候选规模”
* 由 **高 df 词** 决定
* 在你这个语料中，它天然就很大

---

## 4. `PostingOps`

* 是判断性能的**最终裁判**
* 如果它没明显下降：
  * 剪枝就是“表面工程”

---

# 六、最终结论（可以直接写进论文/报告）

### 技术结论

> 在本数据集与阈值设定下，
>
> MaxScore 剪枝主要发生在高成本 posting 累加之后，
>
> 未能显著减少实际 posting 操作次数，
>
> 反而引入额外判断开销，因此在工程上不具备正向收益。

### 工程结论

* **不开 MaxScore** ：
* 更快
* 更稳定
* 更易维护
* 当前实现 **已经足够高效**

---

## 如果你愿意，我可以下一步帮你做的事

* 把这段分析 **改写成论文风格的一小节**
* 或者帮你设计一个：
  > “只有在 remain 很快收敛的特殊年份/参数下才启用 MaxScore”的策略
  >
* 或者直接帮你写一个：
  > `why we disable MaxScore` 的工程说明模板
  >

你现在这个插桩 + 对照实验， **已经是非常标准、非常专业的 IR 工程决策流程了** 。
