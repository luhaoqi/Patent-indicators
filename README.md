# 中国专利指标计算与统计分析项目说明

## 1. 项目在做什么

本项目围绕“中文专利文本能否构造出可用于企业创新研究的数量化指标”这一问题，分成两个连续阶段：

1. **第一阶段（stage1）**  
   根据中文专利文本计算专利层创新指标，核心输出是每篇专利的：
   - `BS`
   - `FS`
   - `Quality_q = FS / (BS + epsilon)`

2. **第二阶段（stage2）**  
   以第一阶段结果为输入，继续做：
   - 指标诊断
   - 专利层图表与描述统计
   - 特殊企业 / 专精特新企业对比
   - 公司年度创新指数构造
   - 与上市公司财务面板的固定效应回归

项目的核心目标不是只跑出一个指标，而是形成一套**可重复、可比较、可批量运行**的实验流程。  
每组参数对应一个独立实验目录，方便后续比较不同窗口、阈值、TopK 和样本口径。

---

## 2. 仓库结构

```text
patent_quality/          第一阶段主流程代码
run_full.py              第一阶段单实验入口
run_stage2.py            第二阶段单实验一键入口

analysis/                    第二阶段脚本与公共模块
analysis/common/             stage2 的路径、IO、分析、表格、绘图公共逻辑
analysis/run_stage2_pipeline.py   第二阶段单实验总控
analysis/run_stage2_batch.py      第二阶段批量入口

docs/                    文档
tests/                   小样本测试
outputs/                 统一实验输出目录

data/raw/                原始专利数据
stopword/                停用词
user_dict/               用户词典
```

---

## 3. 第一阶段：专利指标计算

### 3.1 第一阶段做了什么

第一阶段核心入口是 `patent_quality.pipeline.run_all(cfg)`，流程分为 6 个顺序步骤：

1. **构建词表与分年 DF**  
   输入原始专利 CSV，筛选“发明授权”，分词后统计：
   - 全局词表
   - 分年份 DF
   - 每年文档数

2. **准备分年 tokens**  
   把文本按年份切分并持久化，避免后续重复分词。

3. **回顾性 TF-IDF 向量化**  
   只使用当年及以前的信息构造 IDF，避免未来信息泄露。

4. **向量剪枝**  
   去掉停用词和高 DF 词，并按文档保留 TopK 权重项。

5. **计算 BS / FS**  
   基于年份滑动窗口和倒排索引，计算专利的相似度贡献，输出每年每篇专利的 `BS`、`FS`。

6. **组装最终结果**  
   汇总所有年份结果，输出 `patent_quality_output.csv`。

### 3.2 第一阶段输入

- 原始专利数据：`data/raw/...`
- 停用词：`stopword/`
- 用户词典：`user_dict/`

### 3.3 第一阶段输出

以 `outputs/experiments/<experiment_id>/stage1/` 为根目录，典型输出包括：

- `patent_quality_output.csv`
- `df/global_df.json`
- `df/term_df_year=YYYY.json`
- `tokens/year=YYYY.jsonl`
- `vectors/year=YYYY.npz`
- `vectors_filtered/year=YYYY.npz`
- `postings/`
- `pair_contrib/`
- `stats/bsfs_year=YYYY.csv`
- `logs/<experiment_id>.log`

### 3.4 第一阶段运行方式

- 小样本 smoke test：
  ```bash
  python tests/test_small.py
  ```

- 正式运行：
  ```bash
  python run_full.py
  ```

---

## 4. 第二阶段：统计分析与回归

第二阶段现在已经完成脚本化，不再依赖 notebook 手工逐段执行。  
主入口是：

- 单实验一键运行：`run_stage2.py`
- 单实验命令行入口：`analysis/run_stage2_pipeline.py`
- 多实验批量入口：`analysis/run_stage2_batch.py`

### 4.1 第二阶段总流程

`stage2` 的总控流程一共 7 步：

1. `diagnostics`
2. `build_main_enriched`
3. `analyze_quality_basic`
4. `analyze_special_firms`
5. `build_ucc_panel`
6. `build_firm_year_innovation`
7. `run_regressions`

下面分别说明每一步做什么、输入是什么、输出是什么。

---

## 5. 第二阶段每一步说明

### 5.1 `diagnostics`

对应脚本：
- [analysis/run_diagnostics.py](analysis/run_diagnostics.py)
- [analysis/common/diagnostics.py](analysis/common/diagnostics.py)

作用：
- 读取第一阶段中间产物
- 检查词表、向量稀疏度、年份间 DF 重叠
- 帮助理解不同 TopK 和窗口参数下的指标行为

输入：
- `stage1/vectors/`
- `stage1/df/`
- `stage1/vocab/`

输出：
- `stage2/diagnostics/avg_vocab_usage.csv`
- `stage2/diagnostics/df_pair_sum.csv`
- `stage2/diagnostics/topk_df_pair_sum_k*.csv`
- `stage2/diagnostics/topk_weight_stats_k*.csv`
- `stage2/diagnostics/yearly_top_vocab_top*.csv`
- `stage2/diagnostics/yearly_vocab_size.csv`

---

### 5.2 `build_main_enriched`

对应脚本：
- [analysis/build_main_enriched.py](analysis/build_main_enriched.py)

作用：
- 从 `stage1/patent_quality_output.csv` 中提取申请号
- 回到原始按年专利 CSV 中回捞额外字段
- 去重并与主表左连接
- 构造第二阶段主分析表 `main_enriched.parquet`

输入：
- `stage1/patent_quality_output.csv`
- `data/raw/中国专利分年份保存数据1985-2025/*.csv`

输出：
- `stage2/data/patent_quality_output.csv`
- `stage2/data/main.parquet`
- `stage2/data/extra_all_dedup.parquet`
- `stage2/data/main_enriched.parquet`

---

### 5.3 `analyze_quality_basic`

对应脚本：
- [analysis/analyze_quality_basic.py](analysis/analyze_quality_basic.py)

作用：
- 做专利层基础图表和描述统计
- 输出 `Quality_q` 与被引证次数关系
- 输出 `Quality_q` 分布
- 输出年度均值趋势
- 输出按不同阈值的年度高质量专利数量

输入：
- `stage2/data/main_enriched.parquet`

输出：
- `stage2/figures/fig_quality_vs_citations_logq_logcite.png`
- `stage2/figures/fig_quality_vs_citations_fit_logq_logcite.png`
- `stage2/figures/fig_quality_distribution_log1p_logy.png`
- `stage2/figures/fig_yearly_mean_quality.png`
- `stage2/figures/fig_yearly_high_q_counts.png`
- `stage2/tables/tbl_desc_patent_quality.csv`
- `stage2/tables/tbl_desc_patent_quality.tex`
- `stage2/tables/tbl_quality_citation_ols.csv`
- `stage2/tables/tbl_yearly_mean_quality.csv`
- `stage2/tables/tbl_yearly_high_q_counts.csv`

---

### 5.4 `analyze_special_firms`

对应脚本：
- [analysis/analyze_special_firms.py](analysis/analyze_special_firms.py)

作用：
- 读取特殊企业 / 科创企业名单
- 先做静态口径的企业层对比
- 再构造 `firm-year` 动态标签 `is_special_year`
- 输出专利层、公司层、公司年层对比
- 输出 A/B/C 分组：
  - `A_treated_year`
  - `B_same_firm_other_year`
  - `C_never_treated`
- 输出事件研究风格的趋势图

输入：
- `stage2/data/main_enriched.parquet`
- 特殊企业名单 `.dta`

输出：
- 中间数据：
  - `stage2/data/company_special_panel.parquet`
  - `stage2/data/firm_year_special_labels.parquet`
  - `stage2/data/patents_special_year.parquet`
  - `stage2/data/company_year_special.parquet`
  - `stage2/data/company_year_abc.parquet`
- 表格：
  - `stage2/tables/tbl_firm_compare.csv`
  - `stage2/tables/tbl_firm_compare.tex`
  - `stage2/tables/tbl_firmyear_compare.csv`
  - `stage2/tables/tbl_firmyear_compare.tex`
  - `stage2/tables/tbl_patent_special_year_quality_summary.csv`
  - `stage2/tables/tbl_firm_year_abc_desc.csv`
- 图像：
  - `stage2/figures/fig_special_vs_other_hist_log1p.png`
  - `stage2/figures/fig_special_year_vs_other_year_trend.png`
  - `stage2/figures/fig_abc_patent_quality_distribution.png`
  - `stage2/figures/fig_abc_firm_year_mean_quality_distribution.png`
  - `stage2/figures/fig_abc_yearly_mean_quality.png`
  - `stage2/figures/fig_abc_yearly_high_q_share.png`
  - `stage2/figures/fig_abc_ab_boxplot.png`
  - `stage2/figures/fig_abc_overall_compare.png`
  - `stage2/figures/fig_event_study_mean_quality.png`

---

### 5.5 `build_ucc_panel`

对应脚本：
- [analysis/build_ucc_panel.py](analysis/build_ucc_panel.py)

作用：
- 根据母公司统一社会信用代码表
- 子公司名称到统一社会信用代码映射表
- 上市公司子公司联营合营明细表
- 构造“上市公司（包括所有子公司）各年度的统一社会信用代码列表”

输入：
- `上市公司统一社会信用代码.csv`
- `上市公司子公司对应统一社会信用代码.csv`
- `STK_NotesSubJoint_merged.csv`

输出：
- `stage2/data/ucc_panel.csv`

说明：
- 如果你已经有现成的 `ucc_panel.csv`，可以直接提供，不必重新构造。

---

### 5.6 `build_firm_year_innovation`

对应脚本：
- [analysis/build_firm_year_innovation.py](analysis/build_firm_year_innovation.py)

作用：
- 将 `main_enriched.parquet` 与 `UCC` 面板连接
- 在公司-年份层面对专利创新质量做聚合
- 当前默认方法是：
  - `TopK` 专利的 `Quality_q` 均值
  - 再按年份做横截面标准化，得到 `Innovation_z`

输入：
- `stage2/data/main_enriched.parquet`
- `stage2/data/ucc_panel.csv` 或外部提供的 UCC 面板

输出：
- `stage2/data/firm_year_innovation.parquet`

关键字段：
- `Stkid`
- `ShortName`
- `year`
- `PatentCount`
- `Innovation_raw`
- `Innovation_z`
- `Method`

---

### 5.7 `run_regressions`

对应脚本：
- [analysis/run_regressions.py](analysis/run_regressions.py)

作用：
- 将公司年度创新指数和财务面板数据连接
- 构造控制变量
- 生成滞后项
- 运行公司固定效应 + 年固定效应的面板回归

当前脚本会跑的主要规格包括：
- `ROA ~ Innovation_z + FE`
- `ROA ~ Innovation_z + controls + FE`
- `ROA ~ L1(Innovation_z) + controls + FE`
- `ROE ~ Innovation_z + controls + FE`
- 如果有研发费用列，还会加 `ROA + RD`

输入：
- `stage2/data/firm_year_innovation.parquet`
- 上市公司财务面板 `.dta`

输出：
- `stage2/data/regression_panel.parquet`
- `stage2/tables/tbl_regression_summary.csv`
- `stage2/tables/tbl_regression_summary.tex`
- `stage2/tables/reg_*.txt`
- `stage2/figures/fig_regression_coefficients.png`

---

## 6. 第二阶段输入依赖

第二阶段除了依赖 `stage1` 结果，还可能依赖以下外部数据：

1. **原始专利主表**
   - 用于 `build_main_enriched`

2. **特殊企业名单**
   - `.dta`
   - 用于 `analyze_special_firms`

3. **UCC 面板或其原始构造输入**
   - 现成 `ucc_panel.csv`
   - 或：
     - 母公司统一社会信用代码表
     - 子公司名称映射表
     - 子公司联营合营明细表

4. **上市公司财务面板**
   - `.dta`
   - 用于 `run_regressions`

如果某些可选输入缺失，总控会自动跳过对应步骤：

- 没有 `special_list_path`：跳过特殊企业分析
- 没有 `ucc_panel` 且也没有其原始构造输入：跳过公司创新指数和回归
- 没有 `financial_data_path`：跳过回归

---

## 7. 第二阶段输出目录

第二阶段所有结果统一写到：

```text
outputs/experiments/<experiment_id>/stage2/
```

目录结构如下：

```text
stage2/
  data/           中间数据和最终分析表
  diagnostics/    指标诊断结果
  figures/        图像
  tables/         CSV / LaTeX / 文本结果表
  logs/           各步骤日志与总日志
  metadata/       配置快照、步骤元信息
```

其中最重要的元数据文件是：

- `stage2/metadata/stage2_config.json`

它会把本次 `stage2` 使用的关键参数完整落盘，而且是按步骤分组保存：

- `inputs`
- `diagnostics`
- `build_main_enriched`
- `analyze_quality_basic`
- `analyze_special_firms`
- `build_firm_year_innovation`
- `run_regressions`

这意味着后续对比实验时，不需要回忆“某张图是用什么参数跑出来的”，直接看对应实验目录里的配置快照即可。

---

## 8. 第二阶段运行方式

### 8.1 单实验一键运行

入口文件：
- [run_stage2.py](run_stage2.py)

先在文件顶部改实验名和分组配置，然后执行：

```bash
python run_stage2.py
```

这个入口适合：
- 固定一组实验参数
- 希望像 `run_full.py` 一样一键运行完整 `stage2`

### 8.2 单实验命令行运行

入口文件：
- [analysis/run_stage2_pipeline.py](analysis/run_stage2_pipeline.py)

示例：

```bash
python analysis/run_stage2_pipeline.py \
  --experiment-id 标题_摘要_window5 \
  --stage1-dir outputs/experiments/标题_摘要_window5/stage1 \
  --raw-patent-dir data/raw/中国专利分年份保存数据1985-2025 \
  --special-list-path analysis/graph/科创企业名单2024.dta \
  --financial-data-path analysis/公司财务/数据/上市公司财务数据/上市公司财务数据.dta \
  --ucc-panel-path "analysis/公司财务/数据/上市公司（包括所有子公司）各年度的统一社会信用代码列表.csv"
```

这个入口适合：
- 临时指定实验
- 从命令行覆盖路径和参数

### 8.3 多实验批量运行

入口文件：
- [analysis/run_stage2_batch.py](analysis/run_stage2_batch.py)

示例：

```bash
python analysis/run_stage2_batch.py --manifest path/to/stage2_manifest.yaml
```

manifest 可以为多个实验分别指定：
- `stage1_dir`
- `raw_patent_dir`
- `special_list_path`
- `financial_data_path`
- `ucc_panel_path`
- 各步骤参数

适合：
- 批量比较多组实验参数
- 统一管理一批 `stage2` 任务

---

## 9. 典型完整数据流

```text
原始专利 CSV
  -> stage1
  -> outputs/experiments/<experiment_id>/stage1/patent_quality_output.csv

patent_quality_output.csv + 原始专利附加字段
  -> build_main_enriched
  -> stage2/data/main_enriched.parquet

main_enriched.parquet
  -> analyze_quality_basic
  -> 专利层图表与描述统计

main_enriched.parquet + 特殊企业名单
  -> analyze_special_firms
  -> firm / firm-year / A-B-C 对比图表与表格

main_enriched.parquet + UCC 面板
  -> build_firm_year_innovation
  -> stage2/data/firm_year_innovation.parquet

firm_year_innovation.parquet + 财务面板
  -> run_regressions
  -> 回归结果表、系数图、回归面板数据
```

---

## 10. 你应该先看哪些文件

建议按下面顺序阅读：

1. [run_full.py](run_full.py)
2. [run_stage2.py](run_stage2.py)
3. [patent_quality/pipeline.py](patent_quality/pipeline.py)
4. [analysis/run_stage2_pipeline.py](analysis/run_stage2_pipeline.py)
5. [analysis/common/config.py](analysis/common/config.py)
6. [analysis/README.md](analysis/README.md)
7. [docs/STAT_改造说明.md](docs/STAT_改造说明.md)

---

## 11. 当前状态

当前项目状态可以概括为：

- `stage1` 已经是稳定的可复现 pipeline
- `stage2` 已完成脚本化总控
- notebook 中的关键逻辑已经迁移到 Python 脚本
- 各实验结果按目录隔离，支持单实验与批量运行
- 关键参数会随实验结果一并落盘

也就是说，现在项目已经从“研究 notebook 集合”改造成了“围绕 experiment 目录的一套可重复实验流程”。
