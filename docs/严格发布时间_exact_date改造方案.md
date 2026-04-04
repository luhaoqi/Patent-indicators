# 严格发布时间 exact-date 改造方案

## 1. 背景与已确认规则

当前 `stage1` 的时间口径是：

- 只保留 `发明授权`
- 年份使用 `申请年份`
- 每个专利只和前后 `k` 年的专利计算 BS/FS
- 不计算同年份内部的 BS/FS

本次改造后，时间口径明确改为：

- 仍然只保留 `发明授权`
- 新的时间标准统一使用 `公开公告日 / 公开公告年份`
- exact 模式下，每篇专利按照“严格日期窗口”计算 BS/FS
- 同一天公开公告的两篇专利，不计入彼此前向或后向
- `stage2` 的年度含义同步切换为 `公开公告年份`

严格日期窗口的定义如下：

- 若 `k=1`，对于 `2023-03-06` 的专利：
  - 后向窗口起点为 `2022-03-06`
  - 前向窗口终点为 `2024-03-06`
- 判断窗口时按“加减自然年”处理，不使用简单的 `365 * k` 天近似
- 仅当另一篇专利的 `公开公告日` 严格早于当前专利时，才进入 `BS`
- 仅当另一篇专利的 `公开公告日` 严格晚于当前专利时，才进入 `FS`

另外，`outputs/shared/raw_patent_authorized_parts` 的输入链路已经修复，当前统计为：

- 扫描总行数：`51,358,743`
- 可用总行数：`51,256,980`
- 发明授权输出总行数：`5,201,413`

这已经回到与当前 `stage1` 产物同一量级，可以作为 exact 模式的共享输入基线。

## 2. 改造目标

本次改造的目标是：

- 在现有流程中增加一个参数，例如 `exact_date=True`
- 当 `exact_date=False` 时，完全保留现有 `申请年份 + 年窗` 逻辑
- 当 `exact_date=True` 时，切换为 `公开公告日 / 公开公告年份 + 严格日期窗`
- exact 模式的产物写入独立目录：
  - `outputs/experiments/<experiment_id>/stage1_exact/`
  - `outputs/experiments/<experiment_id>/stage2_exact/`
- 尽量复用当前 `stage1` 的稀疏向量、倒排、Numba 分块计算优化
- 避免把 exact 模式实现成“全局逐专利动态滑动窗口 + 逐条独立计算”

## 3. 不变项

为了控制改造风险，下列语义保持不变：

- 仍只处理 `发明授权`
- 分词逻辑不变，继续使用当前 `jieba + stopwords` 流程
- 词表构建、TF-IDF、向量剪枝逻辑不变
- 稀疏向量、postings、Numba block kernel 的主框架不变
- `pair_contrib` 的“一个年份对只算一次、供两边汇总复用”的思想不变
- 旧实验目录和旧结果不覆盖、不混写

## 4. 总体设计选择

### 4.1 不采用全局动态滑动窗口

虽然 exact-date 从概念上可以用“按时间排序后维护一个动态滑动窗口”来做，但本项目不采用这一方案作为主实现。

原因是：

- 当前 `stage1` 的性能优势主要来自“按年块处理 + 稀疏 postings + Numba 批量扫描”
- 如果改成全局动态窗口，需要维护动态增删的候选索引
- 会明显削弱当前对目标年 postings 的复用和 block 级批处理复用
- Python 侧状态管理复杂度会上升，性能收益不确定

本次方案仍然采用“按 `公开公告年份` 分块”的主框架，只在块内和边界年份引入精确日期过滤。

### 4.2 exact-date 的核心思路

exact 模式下，文档先按 `公开公告年份` 分桶，每个年份内部再按 `公开公告日` 严格排序。

这样可以保留当前实现的大部分结构：

- `tokens/year=YYYY.jsonl`
- `vectors/year=YYYY.npz`
- `index/year=YYYY.csv`
- `postings/year=YYYY.*`
- `pair_contrib/*.npz`

但这些文件在 exact 模式下的“年份”都解释为 `公开公告年份`。

## 5. shared parquet 改造方案

### 5.1 输入标准

exact 模式统一使用：

- `outputs/shared/raw_patent_authorized_parts/`

不再直接从原始 CSV 读取 exact 模式数据。

### 5.2 `build_raw_patent_authorized_parts.py` 的改造

在现有脚本基础上，增加以下步骤：

1. 继续按当前方式读取 CSV，修复“尾部多一个空字段”的行。
2. 继续仅保留 `发明授权`。
3. 保留原始的 `公开公告日` 与 `公开公告年份` 字段。
4. 新增一个仅用于排序和比较的辅助列，例如 `公开公告日_ord`。
   - 该列是把 `公开公告日` 解析后转换成单调可比较的整数。
   - 该列只服务于排序、二分查找、窗口边界比较。
5. 每个按年份输出的 parquet part 在最终写盘前，按以下顺序稳定排序：
   - `公开公告日_ord`
   - `申请号`
6. 将排序后的结果重新写回最终 parquet 文件。

### 5.3 shared parquet 的约束

该辅助列是 exact 模式在 shared 层新增的唯一日期辅助信息，不再额外存更多冗余日期列。

同时要求：

- 文件粒度仍按当前 `中国专利数据库YYYY年.parquet` 保持
- 该 `YYYY` 的语义视为 `公开公告年份`
- metadata 中记录：
  - `sort_by = ["公开公告日_ord", "申请号"]`
  - `date_col = "公开公告日"`
  - `year_col = "公开公告年份"`
  - `invalid_publish_date_rows`

如果某行 `公开公告日` 无法解析：

- 该行仍可保留在 parquet 中
- 但 exact `stage1` 会在读取时丢弃，并在日志中记录数量

这样 shared 层不承担“最终裁决”职责，exact 计算层负责最终过滤。

## 6. `Config` 与目录布局改造

### 6.1 `Config` 新增参数

建议在 `patent_quality/config.py` 中新增以下参数：

- `exact_date: bool = False`
- `public_date_col: str = "公开公告日"`
- `public_year_col: str = "公开公告年份"`
- `shared_authorized_parts_dir: Optional[str] = "outputs/shared/raw_patent_authorized_parts"`

语义如下：

- `exact_date=False`
  - 继续使用当前 `data_path`、`申请年份`、CSV 读取逻辑
- `exact_date=True`
  - 输入切换到 `shared_authorized_parts_dir`
  - 年份切换到 `public_year_col`
  - 日期窗口切换到 `public_date_col`

### 6.2 实验目录布局

建议在 `patent_quality/project_paths.py` 中扩展 experiment layout，使其支持：

- `stage1_dir`
- `stage2_dir`
- `stage1_exact_dir`
- `stage2_exact_dir`
- `verification_dir`

具体目录为：

- `outputs/experiments/<experiment_id>/stage1/`
- `outputs/experiments/<experiment_id>/stage2/`
- `outputs/experiments/<experiment_id>/stage1_exact/`
- `outputs/experiments/<experiment_id>/stage2_exact/`
- `outputs/experiments/<experiment_id>/verification/`

当 `exact_date=True` 时：

- `stage1` 输出写入 `stage1_exact`
- `stage2` 输出写入 `stage2_exact`
- checkpoint、日志、中间文件全部和旧模式隔离

## 7. stage1 exact 改造方案

### 7.1 数据读取层

新增一条 exact 模式专用的数据读取路径：

- 从 `outputs/shared/raw_patent_authorized_parts/*.parquet` 读取
- 读取后按 `公开公告年份` 分桶
- 每个年份内部严格保持 shared parquet 已经排好的顺序

exact 读取层仍保留以下语义：

- 仍按 `申请号` 做去重保护
- `公开公告日` 无法解析的文档直接跳过
- 文本拼接字段仍沿用现有 `col_text_parts`
- 额外字段仍沿用现有 `extra_cols`

### 7.2 token / 向量 / index 的语义变化

exact 模式下，`tokens/year=YYYY.jsonl`、`vectors/year=YYYY.npz`、`index/year=YYYY.csv` 中的 `year` 统一解释为 `公开公告年份`。

同时要求：

- `tokens/year=YYYY.jsonl` 的行顺序必须等于该年份按 `公开公告日_ord, 申请号` 排序后的顺序
- `vectors/year=YYYY.npz` 的 row 顺序必须与 tokens 一致
- `index/year=YYYY.csv` 必须显式记录：
  - `row`
  - `申请号`
  - `公开公告年份`
  - `公开公告日`
  - `专利名称`
  - `extra_cols`

必要时可附加 `公开公告日_ord` 列，便于调试与验证。

### 7.3 exact `k=1` 的计算结构

当 `k=1` 且 `exact_date=True` 时，只计算两类年份块：

- 同年块：`(t, t)`
- 相邻年块：`(t, t+1)`

不再沿用当前“只算前后年、不算同年”的设定。

### 7.4 exact 一般 `k` 的计算结构

当 `k>1` 时：

- `|year_diff| < k` 的内部年份块可以继续整块计算
- `|year_diff| = k` 的边界年份块才需要精确日期过滤
- 同年块 `(t, t)` 始终需要单独处理方向性

因此 exact 模式仍然适合保留 year-pair 主框架，不需要退化成逐专利全局滑动窗口。

## 8. exact 模式下的 BS/FS 计算方式

### 8.1 仍保留当前主优化结构

当前 `stage1` 的性能核心不是普通稠密矩阵乘法，而是：

- 稀疏 TF-IDF 向量
- term postings
- block 级 query 批处理
- Numba kernel

exact 模式继续保留这套结构。

需要避免的是：

- 把每篇专利拆成一次独立 kernel 调用
- 对同一目标年份反复初始化工作数组
- 把目标 postings 的复用打散

### 8.2 为什么“日期过滤本身”不贵

单个日期比较本身非常便宜，真正昂贵的是：

- 扫 postings
- 对候选文档做 `acc[j] += xw * yw`

因此 exact 模式的性能关键不是“多了日期比较”，而是：

- 日期过滤是在候选贡献都算完以后才做
- 还是在 postings 扫描前就把非法目标区间裁掉

### 8.3 推荐实现：前置区间裁剪

推荐做法是：

1. 每个年份内的 row 已经按 `公开公告日_ord` 排序。
2. 对于每个 query 文档，先基于 `公开公告日` 计算其合法窗口边界。
3. 在目标年中用二分查找得到合法 row 区间。
4. postings 扫描时只遍历这个合法 row 区间内的目标文档。

这样：

- 相邻年份中大量非法目标文档不会进入累加
- `k=1 exact-date` 的开销会接近当前 `k=1`
- 不会退化成“当前相邻年全算 + 同年再额外算一遍”的高膨胀方案

### 8.4 不推荐实现：后置过滤

不推荐只在最终累加阶段增加一个日期 `if`。

原因是：

- 非法候选的 postings 扫描成本已经发生
- 无法真正省掉相邻年份的大量无效计算
- `k=1` 时更容易接近“当前方案 + 新增同年计算”的成本

## 9. `pair_contrib` 在 exact 模式下的设计

### 9.1 `pair_contrib` 的作用

`pair_contrib` 的作用不是让“单次相似度计算”更快，而是让“一个年份对的昂贵计算只做一次”。

以跨年 `(2020, 2021)` 为例：

- `2020` 的结果会贡献给 `FS`
- `2021` 的结果会贡献给 `BS`

如果不保存年份对结果，那么同一对年份的相似度关系会被重复计算。

保存 `pair_contrib` 后：

- `2020-2021` 只算一次
- 之后汇总 `BS/FS` 时直接读取该结果相加
- 还可以支持断点续跑与定位慢年份对

### 9.2 exact 模式下的拆分方式

exact 模式继续保留跨年 `pair_contrib`，但要新增同年定向贡献文件。

建议：

- 跨年文件继续放在 `pair_contrib/`
  - 文件名沿用当前风格，例如 `x=2020_y=2021.npz`
  - 只累计满足 exact 日期条件的合法 pair
- 同年文件也放在 `pair_contrib/`
  - 文件名为 `same_year=2020.npz`
  - 内容不再是对称贡献，而是：
    - `bs_same`
    - `fs_same`

### 9.3 同年方向性的原因

同年 `(t, t)` 和跨年 `(t, t+1)` 的最大不同是：

- 跨年方向天然由年份顺序决定
- 同年内部必须依赖精确日期判断方向

对于同年一对专利 `(i, j)`：

- 若 `date_i < date_j`
  - 该对相似度进入 `FS[i]`
  - 该对相似度进入 `BS[j]`
- 若 `date_i > date_j`
  - 方向相反
- 若 `date_i == date_j`
  - 两边都不计

因此同年不能复用当前“单个对称贡献向量”的形式，必须单独保存方向性结果。

## 10. stage2 exact 改造方案

### 10.1 语义切换

当 `exact_date=True` 时，`stage2` 的年度含义统一切换为 `公开公告年份`。

这意味着：

- 企业-年份创新指标的年份维度改为 `公开公告年份`
- 各类按年导出的 top patents、统计表、面板表，都以 `公开公告年份` 为准

### 10.2 输出目录

exact 模式下，`stage2` 结果写入：

- `outputs/experiments/<experiment_id>/stage2_exact/`

### 10.3 代码层的修改原则

`stage2` 脚本修改时遵循以下原则：

1. 读取 exact 模式的 `stage1_exact` 产物。
2. 把使用 `申请年份` 的逻辑改为 `公开公告年份`。
3. 对外导出的年度字段优先显式命名为 `公开公告年份`。
4. 如果某些表为了兼容需要保留 `year` 列，则 `year` 的语义也必须明确视为 `公开公告年份`。

重点涉及的脚本包括但不限于：

- `analysis/run_stage2_pipeline.py`
- `analysis/run_stage2_experiment.py`
- `analysis/run_stage2_batch.py`
- `analysis/build_firm_year_innovation.py`
- `analysis/export_top_patents_by_year.py`

## 11. 参数与入口层方案

### 11.1 入口脚本

建议仍以现有入口为主，不单独发明一套全新主流程：

- `run_full.py`
- `analysis/run_stage2_pipeline.py`

只是在配置或 CLI 中增加 `exact_date` 开关。

### 11.2 推荐行为

- `exact_date=False`
  - 输出到 `stage1 / stage2`
  - 读取原始 CSV
  - 仍按 `申请年份`
- `exact_date=True`
  - 输出到 `stage1_exact / stage2_exact`
  - 读取 `outputs/shared/raw_patent_authorized_parts`
  - 按 `公开公告日 / 公开公告年份`

如果需要便于人工使用，可以额外提供：

- `run_full_exact.py`

但这只是便捷入口，不应成为唯一实现方式，核心逻辑仍应由 `Config.exact_date` 驱动。

## 12. 性能预期

### 12.1 exact `k=1` 的重点

用户当前最关心的是 `k=1`。

在该场景下：

- 同年计算是全新的
- 相邻年计算也不再能直接复用当前“整年全量”的逻辑

但只要 exact 模式采用“先按 `公开公告日_ord` 排序，再做前置合法区间裁剪”的方案，性能仍然有望保持在当前 `k=1` 的同一量级。

### 12.2 性能边界判断

可接受方案：

- 保留 year-block
- 保留 postings
- 保留 block 批处理
- 对 exact 边界做前置区间裁剪

不接受方案：

- 全局逐专利滑动窗口
- 每篇专利单独发起一次完整计算
- 先全算完候选，再在最后阶段简单做日期过滤

## 13. 实施步骤

### 步骤1：shared parquet 增加排序辅助列并重写输出

修改：

- `analysis/build_raw_patent_authorized_parts.py`

完成后应满足：

- 每个 parquet part 仍是一个年份文件
- 该年份语义为 `公开公告年份`
- 文件内部按 `公开公告日_ord, 申请号` 排序

### 步骤2：扩展 `Config` 与 experiment layout

修改：

- `patent_quality/config.py`
- `patent_quality/project_paths.py`

完成后应满足：

- 支持 `exact_date=True`
- 支持 `stage1_exact / stage2_exact`

### 步骤3：新增 exact 模式数据读取与 index 语义

修改：

- `patent_quality/data_loader.py`
- `patent_quality/vectorizer.py`

完成后应满足：

- exact 模式从 shared parquet 读取
- tokens / vectors / index 的 row 顺序完全基于 `公开公告日` 排序

### 步骤4：实现 exact BS/FS 计算

修改：

- `patent_quality/similarity.py`
- `patent_quality/pair_compute.py`
- `patent_quality/postings.py`

完成后应满足：

- 支持同年 `(t, t)` 计算
- 支持跨年 `(t, t+1)` 的 exact 日期过滤
- 支持同一天不计
- 保留跨年 `pair_contrib`
- 新增同年 `same_year=YYYY.npz`

### 步骤5：调整 stage1 最终汇总

修改：

- `patent_quality/quality.py`

完成后应满足：

- final CSV 使用 exact 模式的年份与日期口径
- `BS/FS` 正确叠加同年和跨年贡献

### 步骤6：调整 stage2

修改：

- `analysis/run_stage2_pipeline.py`
- `analysis/run_stage2_experiment.py`
- `analysis/run_stage2_batch.py`
- `analysis/build_firm_year_innovation.py`
- `analysis/export_top_patents_by_year.py`

完成后应满足：

- exact 模式使用 `stage1_exact`
- 所有按年统计逻辑转向 `公开公告年份`

### 步骤7：验证与回归

先跑小样本，再跑全量。

## 14. 验证方案

至少验证以下内容：

### 14.1 shared 层验证

- `rows_written_total` 与当前 stage1 量级一致
- 每个 parquet part 内 `公开公告日_ord` 单调不下降
- 同一天时 `申请号` 排序稳定

### 14.2 exact 逻辑验证

抽样验证以下规则：

- 同年但更早日期进入 `BS`
- 同年但更晚日期进入 `FS`
- 同一天不进入任何一边
- `k=1` 时边界日期按自然年精确生效
- 相邻年只统计落在合法日期区间内的 pair

### 14.3 回归验证

- `exact_date=False` 时，旧结果完全不变
- `exact_date=True` 时，所有输出都落到 `stage1_exact / stage2_exact`
- checkpoint 不混用
- 日志和统计文件能明确识别 exact 模式

## 15. 风险点与控制措施

### 风险1：shared parquet 排序后破坏行顺序一致性

控制措施：

- exact 模式明确以“发布时间排序后的顺序”为唯一 row 顺序来源
- old mode 完全不读取该顺序

### 风险2：同年方向性实现错误

控制措施：

- 同年贡献单独落盘，不与跨年贡献共用同一套对称结构
- 对 hand-crafted 小样本做逐对验证

### 风险3：stage2 混入旧年份语义

控制措施：

- exact 输出目录单独隔离
- exact 模式下优先显式使用 `公开公告年份`

### 风险4：日期过滤做成后置过滤，性能恶化

控制措施：

- 设计上明确要求“合法 row 区间前置裁剪”
- 不接受只在最终累加处增加日期判断的简化实现

## 16. 最终定稿结论

本次改造的最终方案确定为：

1. 时间标准统一切换到 `公开公告日 / 公开公告年份`。
2. 只在 `exact_date=True` 时启用新逻辑。
3. exact 模式只使用 `outputs/shared/raw_patent_authorized_parts` 作为输入。
4. shared parquet 每个年份文件内部按 `公开公告日_ord, 申请号` 排序。
5. `stage1` exact 仍保留 year-block + postings + Numba 的主框架。
6. `k=1` exact 计算同年块与相邻年块，不采用全局动态滑动窗口。
7. 跨年继续使用 `pair_contrib`，同年新增方向性 `same_year=YYYY.npz`。
8. 同一天公开公告的两篇专利，不计入彼此前后向。
9. exact 结果写入 `stage1_exact / stage2_exact`，与旧模式彻底隔离。
10. `stage2` 全链路的年度含义同步切换为 `公开公告年份`。
