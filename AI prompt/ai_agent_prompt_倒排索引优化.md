# 给 Codex 的实现提示词（请完整阅读后开始改代码）

你将要在**现有仓库**中实现一个替代 SpGEMM 的高性能相似度计算方案，用于 pipeline 的 **BS/FS（Backward Similarity / Forward Similarity）** 计算阶段。

现状：

* `pipeline.py` 当前阶段顺序：
  * Stage1 vocab/df
  * Stage2 tokens
  * Stage3 向量化（按年输出 `artifacts/vectors/year=YYYY.npz`）
  * Stage4 向量剪枝（输出 `artifacts/vectors_filtered/year=YYYY.npz`，由 `pruning.py` 实现）
  * Stage5 计算 BS/FS（当前在 `similarity.py` 中用 `S = M_T.dot(M_Y.T)` 进行 SpGEMM，然后阈值过滤、行求和）

问题：

* 对于大年份（Nx≈10万~50万，词表≈11万，nnz≈700万 ~ 1200万/年），SpGEMM 产生的中间 `S` 极大、`S.nnz` 可到数亿，导致内存与时间无法承受。

目标：

* 用 **IR 风格候选生成（倒排索引）+ 无损阈值剪枝（MaxScore）+ 行/块级 accumulator** 替换 SpGEMM。
* 计算粒度改为  **pair（年份对）级** ，并且：
  1. `(x,y)` 只算一次；
  2. 同一次计算同时产出 x 和 y 两侧的贡献（contrib）；
  3. 每算完一个年份对立即落盘（pair checkpoint），可断点续跑。
* Stage5 的最终输出仍然是每个年份 t 的 `bsfs_year=t.csv`（row,BS,FS），但**计算过程**从矩阵乘法改为读取 pair 贡献向量汇总。

---

## 一、要实现的最终方案（核心思想）

### 1) IR 候选生成为什么能明显快过 SpGEMM

SpGEMM 会生成大量最终达不到阈值的 `S[i,j]`（共享任意词项就会产生候选），再事后 `data[data<thr]=0`。

IR 做法：固定一篇 X 文档 `i`（相当于计算矩阵的 **一行** ），通过倒排只访问共享词的候选 `j`，并且用 MaxScore 在**累加过程中**提前剔除不可能达到阈值的候选，避免生成/维护巨大 `S`。

### 2) 分块运算如何压内存、时间几乎不变甚至更快

* IR 本身是逐行（doc）计算，但工程上要按 `block_size_docs` 对 X 的行分块，做到：
  * 每块内复用 accumulator，避免频繁分配；
  * 每块输出进度日志；
  * 可选 block 级中间落盘（如果需要更细粒度恢复）。
* 关键： **永远不构造 S** ；postings 可用 `np.memmap`，避免把 1200 万 nnz 全塞进内存。

### 3) (x,y) 只算一次，同时更新两个方向

相似度（点积/余弦点积）对称：`s(i,j)=s(j,i)`。
当你用“X 查询 Y”算出所有满足 `s(i,j) >= thr` 的 `j` 以及它们的 `score` 时，同一刻就能更新：

* `contrib_x[i] += score`
* `contrib_y[j] += score`

因此 `(x,y)` 只算一次即可得到两边需要的贡献向量。

### 4) pair 级落盘 checkpoint

对每个年份对 `(x,y)` 计算完成后保存：

* `contrib_x`（长度 Nx）
* `contrib_y`（长度 Ny）
  到 `artifacts/pair_contrib/x=XXXX_y=YYYY.npz`。

断点续跑：如果该文件存在就跳过该 pair。

---

## 二、你需要在代码里新增/改造的模块与接口

你要在不破坏 Stage1-4 的情况下，改造 Stage5。

### A. 新增 postings 构建与缓存（建议新文件 `postings.py`）

 **目标** ：为每个年份 y 构建倒排索引，采用全 NumPy 扁平数组，便于 mmap。

#### 数据结构（必须按此格式实现）

* `ptr_y`: shape (V+1,), dtype=int64
* `docs_y`: shape (nnz_y,), dtype=int32
* `vals_y`: shape (nnz_y,), dtype=float32
* `max_y`: shape (V,), dtype=float32，表示每个词在 y 年份出现时的最大权重（用于 MaxScore 上界）

 **解释** ：词 w 的 posting 位于 `[ptr_y[w], ptr_y[w+1])`。

#### 建议接口

* `build_postings_for_year(cfg, year: int, vectors_base: str) -> None`
  * 输入：`year=YYYY.npz`（CSR）
  * 输出：保存到 `artifacts/postings/year=YYYY_ptr.npy` / `docs.npy` / `vals.npy` / `max.npy`
* `load_postings_for_year(cfg, year: int, mmap: bool=True) -> tuple(ptr, docs, vals, maxy)`

#### 构建方式

* 最简单：加载 CSR 后转 CSC：`M_csc = M.tocsc()`，用 `M_csc.indptr` 当作 `ptr_y`，`M_csc.indices` 当 `docs_y`，`M_csc.data` 当 `vals_y`，并对每列切片求 `max` 得到 `max_y`。
* 注意 dtype：把 indices 转成 int32，data 转 float32。
* 保存时使用 `np.save`；mmap 读取时用 `np.load(..., mmap_mode='r')`。

### B. 新增 pair 计算引擎（建议新文件 `pair_compute.py`）

#### 输出定义（pair_contrib）

对每个 pair (x,y) 输出：

* `contrib_x`: float32, shape (Nx,)
* `contrib_y`: float32, shape (Ny,)
  保存路径：`artifacts/pair_contrib/x=XXXX_y=YYYY.npz`（统一 `x<y`）

#### 关键接口

* `compute_pair_contrib(cfg, x: int, y: int) -> str`
  * 读取 `Mx`（CSR）
  * 读取 postings(y)：`ptr_y, docs_y, vals_y, max_y`
  * IR + MaxScore 计算贡献
  * 原子落盘并返回输出路径

#### IR 核心计算（必须按“accumulator + touched”实现，不要用 Python dict）

 **accumulator 定义** ：固定一行 i 时暂存 `score[j] = s(i,j)` 的容器。

推荐实现：

* `acc = np.zeros(Ny, dtype=float32)` （复用，不要每行 new）
* `mark = np.zeros(Ny, dtype=uint8)` （或 int32 seen-id）
* `touched = np.empty(<估计上限>, dtype=int32)` + `touched_len`

更新逻辑：

* 遍历 i 行的 term（indices/data）：(w, xw)
* 获取 posting：`start=ptr_y[w]; end=ptr_y[w+1]`
* 遍历 posting 的 (j, yv)：
  * if mark[j]==0: mark[j]=1; touched[touched_len]=j; touched_len += 1
  * acc[j] += xw * yv

行结束：

* 遍历 `touched[0:touched_len]`：
  * if acc[j] >= thr:
    * contrib_x[i] += acc[j]
    * contrib_y[j] += acc[j]
  * 清理：acc[j]=0; mark[j]=0
* touched_len = 0

 **MaxScore（无损阈值剪枝）** ：

* 对该行的每个 term w 计算上界：`ub = xw * max_y[w]`
* 按 ub 降序遍历 term，并维护 `remain = sum(ub)`
* 处理 term 后 `remain -= ub_term`
* 可选剪枝：对 touched 候选 j，若 `acc[j] + remain < thr`，可把 j 标记为 dead（不再更新）。

实现建议（按推进顺序）：

1. 先实现“行末阈值过滤”（正确性基线）
2. 再加 MaxScore 剪枝（加速），可先做简化版（每处理若干 term 扫一次 touched）

#### 性能热点与要求（非常重要）

* **禁止**在热循环里用 Python dict 存 score。
* 热循环推荐用 **Numba** JIT：把“遍历 posting + acc 更新 + touched 维护”编译为机器码。
  * 允许先做纯 Python 版本用于小规模验证，但最终必须切到 Numba/等效 C 路径。
* postings 必须是 NumPy 扁平数组（ptr/docs/vals/maxy），可 mmap。
* dtype 强制：
  * doc id: int32
  * ptr: int64
  * weights/acc/contrib: float32

### C. 改造 Stage5：从 SpGEMM 改为 “pair_contrib + 汇总”

当前 `similarity.py::compute_bs_fs` 对每个 t：

* 加载 M_T
* 对 back_years/forward_years 做 `S=M_T.dot(M_Y.T)`
* 阈值过滤 + 行求和

你要把它替换为：

#### Step C1：生成 pair_list（只覆盖窗口）

* 窗口为 `cfg.window_size`，只需要 pairs `(t, t±1..window)`。
* 去重并统一为 `x<y`。
* 建议保存到 `artifacts/pair_list.json`，方便复用。

#### Step C2：先确保所有 pair_contrib 都存在（或边算边汇总）

* 遍历 pair_list：
  * 如果 `pair_contrib/x=.._y=..npz` 不存在：调用 `compute_pair_contrib(cfg,x,y)` 生成
  * 已存在则跳过

#### Step C3：按年份汇总 BS/FS

对每个年份 t：

* 需要得到：
  * BS(t) = sum_{y in [t-window, t-1]} contrib_from_pair(t,y)
  * FS(t) = sum_{y in [t+1, t+window]} contrib_from_pair(t,y)

读取 pair 文件时：

* 如果 pair 文件名是 `x=t,y=y (t<y)`：t 侧贡献是 `contrib_x`
* 如果 pair 文件名是 `x=y,y=t (y<t)`：t 侧贡献是 `contrib_y`

输出仍写到：`artifacts/stats/bsfs_year={t}.csv`，格式不变。

### D. pipeline.py 的集成点（保持现有 Stage1-4，不改逻辑）

* 保留 Stage4（`pruning.py::prune_vectors_by_year`）输出的 `vectors_filtered`。
* Stage5 改为新实现：
  * 增加配置：
    * `use_vectors_filtered_for_bsfs=True`
    * `vectors_filtered_dir='vectors_filtered'`（已有）
    * `pair_contrib_dir='pair_contrib'`
    * `postings_dir='postings'`
    * `block_size_docs=10000`（可调）
    * `postings_mmap=True`
    * `enable_maxscore=True/False`
    * `method_version='ir_v1'`（用于结果失效控制）
* checkpoint 设计：
  * Stage5 不再只写 `bsfs_years=True`，建议新增：
    * `pair_contrib_done: ["x=.._y=..", ...]`（可选；文件存在即可判定完成）

---

## 三、落盘与断点续跑（必须做到）

### 1) pair_contrib 原子写入

* 写 `.../x=2008_y=2012.tmp.npz`
* 完成后 `os.replace(tmp, final)`

### 2) 存在即跳过

* `if cfg.skip_if_exists and os.path.exists(final_path): return final_path`

### 3) method_version 防错

* 在 npz 里存 meta（json str 或单独字段）：包括 `thr`, `window_size`, `method_version`, `vectors_dir`。
* 如果方法或阈值变化，旧结果应被识别为无效（最简单：method_version 变化就写到新目录名）。

---

## 四、验收标准（你写完后必须通过）

1. **正确性（小数据验证）**

* 选两个小年份（如 1985/1986），用旧 SpGEMM 跑出 BS/FS
* 用新 IR 方案跑出 BS/FS
* 两者应在浮点误差内一致（或非常接近）

2. **资源**

* 不允许出现 `S = Mx.dot(My.T)`
* 内存峰值必须明显低于 SpGEMM 版本（不依赖 swap）

3. **断点续跑**

* 中断后重跑应从已存在的 pair_contrib 继续

4. **日志**

* pair 级日志：开始/结束、Nx/Ny、nnz_x/nnz_y、耗时、处理 doc 数、候选 touched 平均/最大、命中数（>=thr 的 pair 数）

---

## 五、实现建议（按顺序推进，避免一次写太大）

1. 先实现 postings 构建与缓存（验证能生成 ptr/docs/vals/maxy）
2. 实现 pair 计算的纯 Python 版本（acc+touched，不用 dict），验证小年份正确性
3. 把热循环迁移到 Numba（或等价方式），确认性能提升
4. 改造 Stage5：生成 pair_list → 生成/读取 pair_contrib → 汇总 BS/FS 输出

> 注意：可以先不开 MaxScore，只做 IR 候选累加 + 行末阈值过滤；正确后再加 MaxScore。

---

## 六、你需要重点关注的文件

* `pipeline.py`：Stage1~6 的 orchestration（Stage5 调用点）
* `similarity.py`：现有 Stage5 的 SpGEMM 实现（需要替换/重构）
* `pruning.py`：Stage4 向量剪枝（不改，但要确保 Stage5 使用 `vectors_filtered`）

完成后：Stage5 应不再依赖 SpGEMM，并且能在大年份对上稳定运行、pair 级可续跑。
