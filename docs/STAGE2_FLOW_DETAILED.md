# Stage2 详细流程说明

本文档按当前仓库实现，记录 `stage2` 的总控流程、各阶段输入输出，以及每一步实际做了什么，便于后续回顾。

当前 `stage2` 总控入口位于 [analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py)。

## 1. 总控结构

`run_stage2()` 会把第二阶段拆成 7 个顺序步骤：

1. `diagnostics`
2. `build_main_enriched`
3. `analyze_quality_basic`
4. `analyze_special_firms`
5. `build_ucc_panel`
6. `build_firm_year_innovation`
7. `run_regressions`

执行时会统一写入：

```text
outputs/experiments/<experiment_id>/stage2/
  metadata/
  data/
  diagnostics/
  figures/
  tables/
  logs/
```

其中：

- `metadata/stage2_config.json` 记录 stage2 参数快照
- `metadata/run_stage2_pipeline.json` 记录阶段执行摘要
- `logs/stage2.log` 记录总控日志
- 各阶段自己的日志写在 `logs/*.log`

## 2. 参数结构

stage2 参数结构定义在 [analysis/common/config.py](../analysis/common/config.py)。

参数分为几组：

- `inputs`
- `diagnostics`
- `build_main_enriched`
- `analyze_quality_basic`
- `analyze_special_firms`
- `build_firm_year_innovation`
- `run_regressions`

这些参数会在运行开始时被序列化进 `metadata/stage2_config.json`。

## 3. 第一步：diagnostics

入口函数：

- [analysis/common/diagnostics.py](../analysis/common/diagnostics.py)
- `run_diagnostics()`

### 输入

- `stage1/vectors/year=*.npz`
- `stage1/df/term_df_year=*.json`
- `stage1/vocab/final_vocab.json`
- 参数：
  - `topk_values`
  - `yearly_top_vocab_k`
  - `max_year_gap`

### 具体做的事情

这一阶段主要是对 `stage1` 产物做诊断性统计，不改写 `stage1`，只生成一批检查用表。

#### 3.1 `avg_vocab_usage.csv`

逐年读取 `vectors/year=*.npz` 稀疏矩阵，计算：

- 当年文档数 `n_docs`
- 每篇文档平均保留的非零词项数 `avg_nonzero_terms`

#### 3.2 `df_pair_sum.csv`

逐年读取 `df/term_df_year=*.json`，把每个年份的 DF 字典映射到最终词表，然后构造每年的稀疏 DF 向量。  
接着计算年份与年份之间的 DF 向量内积：

- `year_x`
- `year_y`
- `sum_df_product`

只保留 `|year_x - year_y| <= max_year_gap` 的年份组合。

#### 3.3 `yearly_top_vocab_topK.csv`

逐年读取 `df/term_df_year=*.json`，对每年的词按文档频次从高到低排序，取前 `K` 个，输出：

- `year`
- `rank`
- `word`
- `doc_count`
- `doc_ratio`

#### 3.4 `yearly_vocab_size.csv`

逐年读取 `df/term_df_year=*.json`，计算：

- 每年有效词汇量 `unique_vocab_size`
- 每年文档数 `total_docs`
- 跨年累计词汇并集大小 `vocab_union_size`

#### 3.5 `topk_weight_stats_k{topk}.csv` 和 `topk_df_pair_sum_k{topk}.csv`

对每个 `topk`：

- 逐年读取向量矩阵
- 对每篇文档只保留权重最高的前 `topk` 个词项
- 统计当年：
  - `n_docs`
  - `avg_weight_sum`
  - `avg_squared_weight_sum`
- 再把每个年份的 topk 二值 DF 向量做年份两两内积

### 输出目录

写入：

```text
stage2/diagnostics/
```

## 4. 第二步：build_main_enriched

入口函数：

- [analysis/build_main_enriched.py](../analysis/build_main_enriched.py)
- `build_main_enriched()`

### 输入

- `stage1/patent_quality_output.csv`
- 原始专利目录 `data/raw/中国专利分年份保存数据1985-2025`
- 参数：
  - `chunksize`

### 具体做的事情

这一阶段的目标是把 stage1 的核心指标结果，和原始专利表中的辅助字段拼回到一起，形成一个后续分析统一使用的专利主表。

#### 4.1 读取 stage1 主结果

先读入 `patent_quality_output.csv`，以 `申请号` 为关键键值。

#### 4.2 从原始专利表回捞对应申请号

遍历原始专利目录中的 CSV 文件，按块读取。  
对每个 chunk：

- 取出 `申请号`
- 判断是否出现在 stage1 主结果中
- 如果命中，则把整行原始数据保留下来

#### 4.3 对回捞的原始记录去重

回捞之后，同一个 `申请号` 可能会出现多行，因此按 `申请号` 做去重。

去重规则：

- 若存在 `专利类型` 列，则优先保留 `发明授权`
- 如果某列全是数值，则取最大值
- 如果是文本列，则取第一个非空值

#### 4.4 把原始字段回填进 stage1 主结果

将去重后的原始表按 `申请号` 左连接回 stage1 主结果。  
如果主结果某一列为空，而原始表里对应列非空，则用原始值补齐。

### 输出

写入 `stage2/data/`：

- `main.parquet`
- `extra_all_dedup.parquet`
- `main_enriched.parquet`

## 5. 第三步：analyze_quality_basic

入口函数：

- [analysis/analyze_quality_basic.py](../analysis/analyze_quality_basic.py)
- `analyze_quality_basic()`

### 输入

- `stage2/data/main_enriched.parquet`
- 参数：
  - `exclude_years`
  - `quality_min`
  - `bs_min`
  - `quality_desc_threshold`

### 具体做的事情

这一阶段做的是最基础的专利层统计分析和图表输出。

#### 5.1 样本过滤

过滤逻辑定义在 [analysis/common/analysis.py](../analysis/common/analysis.py) 的 `filter_patents()`。

过滤规则包括：

- 排除指定年份
- 只保留 `Quality_q >= quality_min`
- 只保留 `BS >= bs_min`

如果没有 `被引证次数` 列，则补 0。

#### 5.2 描述统计表

基于过滤后的专利样本，分别输出：

- 全样本 `Quality_q`
- 高质量子样本 `Quality_q >= quality_desc_threshold`
- `被引证次数`

的描述统计表。

#### 5.3 质量与被引证次数关系图

构造：

- 横轴 `log(1 + Quality_q)`
- 纵轴 `log(1 + 被引证次数)`

先画散点图，再做一元线性拟合，输出：

- 散点图
- 带拟合线的图
- 回归结果表

#### 5.4 质量分布图

对 `Quality_q` 取 `log(1 + Quality_q)`，画频数直方图，并将纵轴设为对数尺度。

#### 5.5 年度均值与年度阈值计数

按 `申请年份` 统计：

- 每年平均 `Quality_q`
- 若干阈值下每年的高质量专利数

默认阈值列表包括：

- `0.5`
- `1.0`
- `1.5`
- `2.0`
- `2.5`
- `3.0`

### 输出

表格写到 `stage2/tables/`，主要包括：

- `tbl_desc_patent_quality.csv/.tex`
- `tbl_quality_citation_ols.csv/.tex`
- `tbl_yearly_mean_quality.csv`
- `tbl_yearly_high_q_counts.csv`

图写到 `stage2/figures/`，主要包括：

- `fig_quality_vs_citations_logq_logcite.png`
- `fig_quality_vs_citations_fit_logq_logcite.png`
- `fig_quality_distribution_log1p_logy.png`
- `fig_yearly_mean_quality.png`
- `fig_yearly_high_q_counts.png`

## 6. 第四步：analyze_special_firms

入口函数：

- [analysis/analyze_special_firms.py](../analysis/analyze_special_firms.py)
- `analyze_special_firms()`

核心辅助逻辑位于：

- [analysis/common/analysis.py](../analysis/common/analysis.py)

### 输入

- `stage2/data/main_enriched.parquet`
- 特殊企业名单 `special_list_path`
- 参数：
  - `exclude_years`
  - `quality_min`
  - `bs_min`
  - `quality_threshold`
  - `policy_start_year`
  - `event_window`

### 具体做的事情

这一阶段可以拆成 5 层。

#### 6.1 静态企业层分组

先对专利表按统一社会信用代码聚合成公司层，计算：

- `total_patents`
- `high_q_count`
- `mean_quality`
- `max_quality`
- `log_total_patents`

然后根据特殊企业名单判断企业是否属于特殊企业，输出“特殊企业 vs 其他企业”的静态对比表。

对应函数：

- `build_company_special_panel()`

#### 6.2 firm-year 动态特殊企业标记

先把特殊企业名单转成 `UCC-year` 面板。

判定规则：

- 如果名单中存在 `科创企业称号总数`，则 `> 0` 视为特殊企业年份
- 否则在若干候选标识列上求和，大于 0 即视为特殊企业年份

之后把这个 `UCC-year` 标签表左连接回专利表，给每篇专利打上 `is_special_year`。

同时只保留：

- `policy_start_year` 及以后
- 满足 `exclude_years`、`quality_min`、`bs_min` 条件的样本

对应函数：

- `build_firm_year_special_panel()`
- `attach_special_year_labels()`

#### 6.3 公司年层 special-year 对比

在专利已经带有 `is_special_year` 标签后，再按 `公司-UCC x 年份` 聚合，计算：

- `total_patents`
- `high_q_count`
- `mean_quality`
- `max_quality`
- `is_special_year`
- `log_total_patents`

然后生成：

- firm-year 层面对比表
- 年度趋势表
- 年度趋势图

对应函数：

- `build_company_year_special_panel()`

#### 6.4 A/B/C 三组设计

在 firm-year 层面定义三组：

- `A_treated_year`
- `B_same_firm_other_year`
- `C_never_treated`

含义分别是：

- 企业曾经被处理，且当前年份就是 special-year
- 企业曾经被处理，但当前年份不是 special-year
- 企业从未进入 treated 状态

之后在公司年层面继续统计：

- `total_patents`
- `high_q_count`
- `mean_quality`
- `max_quality`
- `ever_special`
- `firm_group_3`

并输出：

- A/B/C 描述统计表
- 专利层质量分布图
- 公司年平均质量分布图
- 年度平均质量趋势图
- 年度高质量占比趋势图
- A vs B 箱线图
- 总体对比柱状图

对应函数：

- `build_company_year_abc_panel()`
- `build_abc_summary_table()`

#### 6.5 事件研究式趋势

对 `ever_special == 1` 的企业：

- 找到首次进入 `is_special_year == 1` 的年份，记为 `t0`
- 定义 `event_time = year - t0`
- 截取 `[-event_window, +event_window]`
- 按 `event_time` 计算平均 `mean_quality`

对应函数：

- `build_event_study_frame()`

### 输出

数据写到 `stage2/data/`：

- `company_special_panel.parquet`
- `firm_year_special_labels.parquet`
- `patents_special_year.parquet`
- `company_year_special.parquet`
- `company_year_abc.parquet`

表写到 `stage2/tables/`，主要包括：

- `tbl_firm_compare.csv/.tex`
- `tbl_firmyear_compare.csv/.tex`
- `tbl_patent_special_year_quality_summary.csv/.tex`
- `tbl_special_year_trend.csv`
- `tbl_firm_year_abc_desc.csv/.tex`
- `tbl_abc_yearly_mean_quality.csv`
- `tbl_abc_yearly_high_q_share.csv`
- `tbl_abc_overall_compare.csv`
- `tbl_event_study_mean_quality.csv`

图写到 `stage2/figures/`，主要包括：

- `fig_special_vs_other_hist_log1p.png`
- `fig_special_year_vs_other_year_trend.png`
- `fig_abc_patent_quality_distribution.png`
- `fig_abc_firm_year_mean_quality_distribution.png`
- `fig_abc_yearly_mean_quality.png`
- `fig_abc_yearly_high_q_share.png`
- `fig_abc_ab_boxplot.png`
- `fig_abc_overall_compare.png`
- `fig_event_study_mean_quality.png`

## 7. 第五步：build_ucc_panel

入口函数：

- [analysis/build_ucc_panel.py](../analysis/build_ucc_panel.py)
- `build_ucc_panel()`

### 输入

- 母公司统一社会信用代码表 `parent_csv_path`
- 子公司名称到统一社会信用代码映射表 `subsidiary_mapping_path`
- 上市公司子公司联营合营明细表 `subjoint_csv_path`
- 参数：
  - `chunksize`

### 具体做的事情

这一阶段的目标是构造“上市公司及其子公司在每一年对应哪些统一社会信用代码”的面板。

#### 7.1 读取母公司表

要求包含列：

- `stkid`
- `shortname`
- `SocialCreditCode`
- `FirstYear`
- `LastYear`

这张表提供：

- 上市公司证券 ID
- 公司简称
- 母公司的统一社会信用代码
- 公司存续年份范围

#### 7.2 读取子公司名称映射表

要求包含列：

- `企业名称`
- `统一社会信用代码`

读入后按企业名称聚合，形成：

- `企业名称 -> UCC列表`

#### 7.3 扫描联营合营明细表

按块读取 `subjoint_csv_path`，使用列：

- `Symbol`
- `EndDate`
- `RalatedParty`
- `Relationship`

处理步骤：

- 从 `EndDate` 提取年份
- 用 `RalatedParty` 匹配子公司 UCC
- 累积成 `(证券ID, 年份) -> 子公司 UCC 串`

#### 7.4 展开成年份面板

对每家上市公司：

- 从 `FirstYear` 到 `LastYear` 逐年展开
- 取母公司自身 UCC
- 拼上该公司该年的子公司 UCC 串
- 合成 `统一社会信用代码列表`

### 输出

写入：

- `stage2/data/ucc_panel.csv`

列为：

- `证券ID`
- `公司简称`
- `年份`
- `统一社会信用代码列表`

## 8. 第六步：build_firm_year_innovation

入口函数：

- [analysis/build_firm_year_innovation.py](../analysis/build_firm_year_innovation.py)
- `build_firm_year_innovation()`

### 输入

- `stage2/data/main_enriched.parquet`
- `ucc_panel.csv`
- 参数：
  - `top_k`
  - `quality_cap`

### 具体做的事情

这一阶段把专利层创新指标映射到上市公司年度层面，构造成 firm-year 创新指数。

#### 8.1 处理 UCC 面板

读取 `ucc_panel.csv` 后，把：

- `证券ID` / `stkid`
- `公司简称` / `shortname`
- `年份`
- `统一社会信用代码列表`

规范化为：

- `Stkid`
- `ShortName`
- `year`
- `UCC_list`

然后把 `UCC_list` 按 `;` 拆开并 `explode`，形成逐行的：

- `Stkid`
- `ShortName`
- `year`
- `UCC`

#### 8.2 读取并清洗专利主表

从 `main_enriched.parquet` 中只取：

- `申请年份`
- `统一社会信用代码`
- `Quality_q`

然后转换为：

- `year`
- `UCC`
- `Quality_q`

过滤规则：

- `year` 有效
- `UCC` 非空
- `Quality_q` 非空
- `Quality_q <= quality_cap`

#### 8.3 按 UCC 和年份匹配到上市公司

按 `[UCC, year]` 把专利表与 UCC 面板做内连接。  
这样每篇专利会映射到对应上市公司及年份。

#### 8.4 按公司-年份聚合创新指数

按 `[Stkid, ShortName, year]` 聚合：

- `PatentCount`
  该公司该年的匹配专利数
- `Innovation_raw`
  该公司该年内 `Quality_q` 最大的前 `top_k` 篇专利的均值

然后去掉 `Innovation_raw <= 0` 的记录，并记录方法名：

- `Method = Top{top_k}Mean`

#### 8.5 横截面标准化

再对每一个年份：

- 计算所有公司 `Innovation_raw` 的均值 `mu`
- 计算标准差 `sigma`
- 构造 `Innovation_z = (Innovation_raw - mu) / sigma`

如果该年 `sigma = 0` 或缺失，则 `Innovation_z = NaN`。

### 输出

写入：

- `stage2/data/firm_year_innovation.parquet`

主要列包括：

- `Stkid`
- `ShortName`
- `year`
- `PatentCount`
- `Innovation_raw`
- `Innovation_z`
- `Method`

## 9. 第七步：run_regressions

入口函数：

- [analysis/run_regressions.py](../analysis/run_regressions.py)
- `run_regressions()`

### 输入

- `stage2/data/firm_year_innovation.parquet`
- 财务面板 `financial_data_path`
- 参数：
  - `year_min`
  - `year_max`

### 具体做的事情

这一阶段把 firm-year 创新指数与财务面板对齐，构造固定效应回归所需的公司年面板，并运行若干组模型。

#### 9.1 规范化 firm-year 创新指标

先读取 `firm_year_innovation.parquet`，统一字段名到：

- `stkcd`
- `year`
- `Innovation_raw`
- `Innovation_z`
- `PatentCount`

#### 9.2 读取并清洗财务面板

读入财务 `dta` 后，要求至少存在：

- `stkcd`
- `Accper`
- `roa`
- `roe`
- `tq`
- `asset`
- `liability`
- `finlev`
- `gassets`
- `soe`

然后进行处理：

- 将 `Accper` 转换为日期
- 只保留 `12月31日` 的年报口径
- 只保留 `[year_min, year_max]` 之间的年份
- 将 `stkcd` 标准化为 6 位字符串
- 同一公司同一年如有多条，保留最后一条

#### 9.3 财务指标与创新指标合并

按 `[stkcd, year]` 内连接财务数据和创新指标。

随后构造衍生变量：

- `ln_asset = log(asset)`
- `lev_ratio = liability / asset`
- `soe`
- 如果存在 `研发费用`，则构造 `rd_intensity = 研发费用 / asset`

#### 9.4 构造滞后变量

按公司排序后，构造：

- `Innovation_z_lag1`
- `Innovation_z_lag2`
- `PatentCount_lag1`

并将这一份回归用面板先写出：

- `stage2/data/regression_panel.parquet`

#### 9.5 运行固定效应回归

使用 `linearmodels.panel.PanelOLS`，回归中包含：

- 企业固定效应 `EntityEffects`
- 年份固定效应 `TimeEffects`
- 聚类标准误 `cluster_entity=True`

当前代码中定义的模型包括：

- `ROA Baseline`
- `ROA + Controls`
- `ROA Lag1`
- `ROE + Controls`
- 若有研发强度，则额外加入：
  - `ROA + RD`

每个模型都会输出：

- 系数
- 标准误
- t 值
- p 值
- 样本量
- within R²
- 文本版回归摘要

#### 9.6 回归汇总与系数图

最后把所有成功运行的模型整理成：

- 回归汇总表 `tbl_regression_summary.csv/.tex`
- 每个模型的文本摘要 `reg_*.txt`
- 系数误差棒图 `fig_regression_coefficients.png`

### 输出

数据：

- `stage2/data/regression_panel.parquet`

表：

- `stage2/tables/tbl_regression_summary.csv`
- `stage2/tables/tbl_regression_summary.tex`
- `stage2/tables/reg_*.txt`

图：

- `stage2/figures/fig_regression_coefficients.png`

## 10. 各阶段输入输出关系

按依赖顺序，可以把 stage2 理解为：

```text
stage1 输出
  -> diagnostics
  -> build_main_enriched

main_enriched
  -> analyze_quality_basic
  -> analyze_special_firms
  -> build_firm_year_innovation

上市公司/子公司映射源数据
  -> build_ucc_panel
  -> build_firm_year_innovation

firm_year_innovation + 财务数据
  -> run_regressions
```

其中：

- `build_main_enriched` 是专利层统一主表的构造阶段
- `analyze_quality_basic` 和 `analyze_special_firms` 主要负责描述性分析与图表输出
- `build_ucc_panel` 和 `build_firm_year_innovation` 负责把专利层指标映射到公司年层
- `run_regressions` 负责最终公司财务层面的固定效应回归
