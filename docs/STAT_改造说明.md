# 第二阶段统计分析改造说明

## 1. 改造目标

当前第二阶段的核心问题不是“分析逻辑不够”，而是“流程还没有工程化”。后续目标应明确为：

1. 给定任意一组第一阶段结果目录，自动完成第二阶段全部分析；
2. 每组参数的第二阶段结果单独落盘，互不覆盖；
3. notebook 中已稳定的逻辑迁移到 Python 脚本；
4. 图、表、日志、主分析数据表全部有固定输出路径和有意义的文件名；
5. 支持一次性批量比较多组参数实验。

---

## 2. 当前第二阶段的实际组成

建议把现有 `stat/` 内容拆成 4 类任务：

### A. 第一阶段诊断

输入：`artifacts_dir`

代表文件：

- `calc_avg_vocab_usage.py`
- `calc_df_pair_sum.py`
- `calc_topk_df_pair_sum.py`
- `calc_yearly_top_vocab.py`
- `calc_yearly_vocab_size.py`

这类脚本已经比较接近可复用 CLI，只需要把输出从“打印 / txt”进一步标准化即可。

### B. 主分析表构造

输入：

- 第一阶段 `patent_quality_output.csv`
- 原始专利年度 CSV

输出：

- `main.parquet`
- `extra_all_dedup.parquet`
- `main_enriched.parquet`

来源 notebook：

- `stat/graph/合并数据.ipynb`

### C. 图表与企业分组分析

输入：

- `main_enriched.parquet`
- 专精特新 / 科创企业名单

输出：

- 指标分布图
- 年度趋势图
- 专精特新 vs 非专精特新对比图
- `firm_compare.tex`
- `firmyear_compare.tex`

来源 notebook：

- `stat/graph/graph.ipynb`
- `stat/graph/graph_with_firmyear_special.ipynb`

### D. 公司口径面板与回归

输入：

- `main_enriched.parquet`
- 上市公司及子公司统一社会信用代码表
- 财务面板数据

输出：

- `firm_year_innovation.parquet`
- 回归结果表
- 回归配图或摘要表

来源 notebook：

- `stat/公司财务/上市公司子公司.ipynb`
- `stat/公司财务/公司创新指数计算.ipynb`
- `stat/公司财务/reg.ipynb`

---

## 3. 推荐的目标目录结构

后续不要再把第二阶段结果直接写回 `stat/graph/` 或 `stat/公司财务/`。建议改成：

```text
outputs/
  analysis/
    <experiment_id>/
      metadata/
        stage1_config.yaml
        stage2_config.yaml
      data/
        patent_quality_output.csv
        main.parquet
        extra_all_dedup.parquet
        main_enriched.parquet
        firm_year_innovation.parquet
      diagnostics/
        avg_vocab_usage.csv
        df_pair_sum.csv
        topk_df_pair_sum_k10.csv
        yearly_top_vocab_top50.csv
      figures/
        fig_quality_vs_citations_logq_logcite.png
        fig_quality_distribution_logx_logy.png
        fig_yearly_mean_quality.png
        fig_yearly_high_q_counts_thr_0p5_to_3p0.png
        fig_special_vs_other_hist_log1p.png
        fig_special_year_vs_other_year_trend.png
      tables/
        tbl_desc_patent_quality.csv
        tbl_firm_compare.tex
        tbl_firmyear_compare.tex
        tbl_regression_summary.csv
      logs/
        stage2.log
```

这里最关键的是：**实验隔离依赖目录，而不是依赖文件名本身**。文件名保持稳定、语义清晰，参数区分由 `<experiment_id>` 目录承担。

---

## 4. `experiment_id` 的命名建议

建议每组实验都固定一个短标签，例如：

- `window3_thr005_topk30`
- `window5_thr003_topk50`
- `baseline_1985_2025`
- `title_abstract_only_w3`

如果第一阶段已经把结果写在独立目录中，那么第二阶段直接复用该标签即可，不要在第二阶段再重新发明一套命名。

---

## 5. 推荐的脚本拆分方案

### 5.1 先抽公共层

建议新建：

```text
stat/
  common/
    paths.py
    io.py
    filters.py
    plotting.py
    tables.py
```

其中建议统一沉淀以下共用逻辑：

- 中文字体设置
- `filter_patents()` 一类的样本过滤规则
- 读取 `main_enriched.parquet`
- 输出目录创建
- 保存图片 / 表格 / LaTeX
- 日志记录

### 5.2 再拆业务脚本

建议新增或逐步迁移为以下脚本：

1. `stat/build_main_enriched.py`
   从第一阶段输出和原始专利数据生成 `main_enriched.parquet`。
2. `stat/build_ucc_panel.py`
   从母公司、子公司、联营公司数据生成统一社会信用代码面板。
3. `stat/build_firm_year_innovation.py`
   从 `main_enriched.parquet` 和 UCC 面板生成 `firm_year_innovation.parquet`。
4. `stat/analyze_quality_basic.py`
   输出散点图、分布图、年度均值、阈值计数图等基础图表。
5. `stat/analyze_special_firms.py`
   输出专精特新企业相关图表和对比表。
6. `stat/run_regressions.py`
   读取 `firm_year_innovation.parquet` 和财务数据，输出回归摘要。
7. `stat/run_diagnostics.py`
   包装现有 `calc_*.py`，把输出改成可落盘的 CSV / TXT。

---

## 6. 每个脚本应统一支持的参数

建议所有第二阶段脚本至少支持以下 CLI 参数：

- `--experiment-id`
- `--stage1-dir` 或 `--stage1-output`
- `--output-dir`
- `--raw-patent-dir`
- `--special-list-path`
- `--ucc-panel-path`
- `--financial-data-path`
- `--quality-threshold`
- `--exclude-years`
- `--bs-min`

这样后续做批量比较时，不需要改脚本，只改配置文件。

---

## 7. 图片与表格命名规范

### 图片命名

统一使用：

```text
fig_<主题>_<样本口径>_<可选参数>.png
```

示例：

- `fig_quality_vs_citations_logq_logcite.png`
- `fig_quality_distribution_bsmin1e-6_logy.png`
- `fig_yearly_mean_quality_excl1985_1986.png`
- `fig_special_vs_other_hist_log1p.png`
- `fig_special_year_vs_other_year_trend.png`

### 表格命名

统一使用：

```text
tbl_<主题>.csv
tbl_<主题>.tex
```

示例：

- `tbl_desc_quality.csv`
- `tbl_firm_compare.tex`
- `tbl_firmyear_compare.tex`
- `tbl_regression_summary.csv`

原则是：**文件名表达“内容”，实验目录表达“参数版本”**。

---

## 8. 批量运行的推荐入口

建议最终增加一个总控脚本，例如：

```text
stat/run_stage2_batch.py
```

输入一个 manifest 文件，例如：

```yaml
experiments:
  - id: window3_thr005_topk30
    stage1_dir: data/result/window3_thr005_topk30
    raw_patent_dir: data/raw/中国专利分年份保存数据1985-2025
  - id: window5_thr005_topk30
    stage1_dir: data/result/window5_thr005_topk30
    raw_patent_dir: data/raw/中国专利分年份保存数据1985-2025

shared:
  special_list_path: stat/graph/科创企业名单2024.dta
  financial_data_path: stat/公司财务/数据/上市公司财务数据/上市公司财务数据.dta
  output_root: outputs/analysis
```

批处理逻辑建议如下：

1. 为每个 `experiment_id` 创建独立输出目录；
2. 复制或记录该实验的第一阶段配置；
3. 依次运行：
   - diagnostics
   - build_main_enriched
   - build_ucc_panel（如未已有）
   - build_firm_year_innovation
   - analyze_quality_basic
   - analyze_special_firms
   - run_regressions
4. 每一步写日志，并记录成功 / 失败状态。

---

## 9. 推荐的迁移优先级

不要一次性重写所有 notebook。建议按稳定度和收益排序：

### 第一优先级

1. `stat/公司财务/公司创新指数计算.ipynb`
   这个 notebook 已经非常接近标准脚本，最容易先迁移。
2. `stat/graph/合并数据.ipynb`
   它决定第二阶段主表，是后续所有分析的入口。

### 第二优先级

3. `stat/graph/graph_with_firmyear_special.ipynb`
   先拆成公共函数 + 两个分析脚本：
   - 基础指标图表
   - 专精特新企业对比

### 第三优先级

4. `stat/公司财务/reg.ipynb`
   等 `firm_year_innovation.parquet` 路径和字段稳定后再脚本化。
5. `stat/公司财务/上市公司子公司.ipynb`
   这部分步骤较杂，建议最后整理。

### 保留为探索工具

- `stat/graph/read.ipynb`
- `stat/graph/1.ipynb`

这类 notebook 可以继续保留为检查或临时读取工具，不必强行流水线化。

---

## 10. 你后续实际最值得先做的最小改造

如果目标是尽快支持“比较多组参数结果”，建议先完成下面 3 件事：

1. **把 `合并数据.ipynb` 改成脚本**
   让任何一组 `patent_quality_output.csv` 都能自动产出 `main_enriched.parquet`。
2. **把 `graph_with_firmyear_special.ipynb` 的函数部分抽到 `.py`**
   至少实现固定保存图片和表格，不再只 `show()`。
3. **引入实验输出目录**
   即使暂时还没有 batch runner，只要能通过
   `--output-dir outputs/analysis/<experiment_id>` 写结果，就已经解决了 80% 的参数比较问题。

---

## 11. 最终目标状态

理想状态下，第二阶段应该从“几个 notebook 的人工串行操作”变成：

```text
给定第一阶段结果目录
  -> 自动生成 main_enriched.parquet
  -> 自动生成 diagnostics / figures / tables / firm_year_innovation
  -> 自动输出到该实验专属目录
  -> 多组参数实验可以批量运行
```

一旦做到这一步，你就可以稳定地比较不同窗口、阈值、分词口径、剪枝参数带来的指标差异，而不用每次手动改 notebook 路径和导出文件名。

