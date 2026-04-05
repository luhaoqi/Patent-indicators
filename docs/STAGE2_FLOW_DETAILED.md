# Stage2 详细流程说明

本文档记录**当前实现**下的第二阶段实际流程。  
当前 stage2 已经拆成两层：

1. `shared prep`
2. `per-experiment stage2`

如果你只想快速运行，请优先看根目录 [README.md](../README.md)。

---

## 1. 当前入口

### shared prep 入口

- [analysis/run_shared_prep.py](../analysis/run_shared_prep.py)
- [analysis/verify_shared_prep.py](../analysis/verify_shared_prep.py)

### 单实验 stage2 入口

- [analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py)
- [analysis/run_stage2_experiment.py](../analysis/run_stage2_experiment.py)
- [run_stage2.py](../run_stage2.py)

### 批量 stage2 入口

- [analysis/run_stage2_batch.py](../analysis/run_stage2_batch.py)

---

## 2. 运行前提

当前 stage2 是**严格模式**，运行前需要满足：

1. 某个实验的 `stage1` 已完成
2. `outputs/shared/` 下的共享产物已完成

也就是说，stage2 不再：

- 扫描原始专利目录
- 直接读取原始财务 `dta`
- 在实验目录内生成 UCC 面板
- 在实验目录内重新生成特殊企业标签底座

---

## 3. shared prep 流程

shared prep 的目标是把与实验参数无关的静态底座单独落盘，统一写到：

```text
outputs/shared/
```

目录结构：

```text
outputs/shared/
  patent_master/
  special_firm_labels/
  ucc_mapping/
  financial_panel/
  metadata/
  logs/
```

### 3.1 patent_master

对应逻辑：
- [analysis/build_main_enriched.py](../analysis/build_main_enriched.py) 中的 `build_patent_master()`

输入：
- 原始专利 CSV 目录

处理：
- 扫描原始专利 CSV
- 按 `申请号` 去重
- 保留后续分析需要的静态专利字段

输出：
- `outputs/shared/patent_master/patent_master.parquet`
- `outputs/shared/patent_master/metadata.json`

### 3.2 special_firm_labels

对应逻辑：
- [analysis/shared_prep.py](../analysis/shared_prep.py) 中的 `build_special_firm_labels()`

输入：
- 特殊企业名单 `.dta`

处理：
- 清洗企业名单
- 生成 `special_ucc_set`
- 生成 `firm_year_special_labels`

输出：
- `outputs/shared/special_firm_labels/special_panel_clean.parquet`
- `outputs/shared/special_firm_labels/special_ucc_set.parquet`
- `outputs/shared/special_firm_labels/firm_year_special_labels.parquet`
- `outputs/shared/special_firm_labels/metadata.json`

### 3.3 ucc_mapping

对应逻辑：
- [analysis/build_ucc_panel.py](../analysis/build_ucc_panel.py) 中的 `build_ucc_mapping()`

输入：
- 母公司统一社会信用代码表
- 子公司名称到统一社会信用代码映射表
- 上市公司子公司联营合营明细表

处理：
- 构造年度公司 UCC 列表
- explode 成 `Stkid-Year-UCC` 明细映射

输出：
- `outputs/shared/ucc_mapping/ucc_panel.csv`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`
- `outputs/shared/ucc_mapping/metadata.json`

### 3.4 financial_panel

对应逻辑：
- [analysis/shared_prep.py](../analysis/shared_prep.py) 中的 `build_financial_annual_panel()`

输入：
- 原始财务面板 `.dta`

处理：
- `Accper` 转日期
- 只保留 12 月 31 日年报
- `stkcd` 标准化为 6 位
- 同公司同年保留最后一条

输出：
- `outputs/shared/financial_panel/financial_annual_clean.parquet`
- `outputs/shared/financial_panel/metadata.json`

### 3.5 shared prep metadata

shared prep 会额外写：

- `outputs/shared/metadata/run_shared_prep.json`
- `outputs/shared/metadata/verify_shared_prep.json`

---

## 4. per-experiment stage2 流程

每个实验的结果统一写到：

```text
outputs/experiments/<experiment_id>/stage2/
```

目录结构：

```text
stage2/
  data/
  diagnostics/
  figures/
  tables/
  logs/
  metadata/
```

[analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py) 当前按 6 步执行。

### 4.1 diagnostics

对应逻辑：
- [analysis/common/diagnostics.py](../analysis/common/diagnostics.py)

输入：
- `stage1/df/`
- `stage1/vectors/`
- `stage1/vocab/`

输出：
- `stage2/diagnostics/*.csv`

作用：
- 生成词表、DF、向量使用等 diagnostics 输出

### 4.2 build_experiment_patent_panel

对应逻辑：
- [analysis/build_main_enriched.py](../analysis/build_main_enriched.py) 中的 `build_experiment_patent_panel()`

输入：
- `stage1/patent_quality_output.csv`
- `outputs/shared/patent_master/patent_master.parquet`

处理：
- 按 `申请号` 将当前实验的 `BS / FS / Quality_q` 与静态专利字段 join

输出：
- `stage2/data/patent_quality_output.csv`
- `stage2/data/main.parquet`
- `stage2/data/experiment_patent_panel.parquet`
- `stage2/metadata/build_experiment_patent_panel.json`

### 4.3 analyze_quality_basic

对应逻辑：
- [analysis/analyze_quality_basic.py](../analysis/analyze_quality_basic.py)

输入：
- `stage2/data/experiment_patent_panel.parquet`

输出：
- 专利层基础图表
- 描述统计表
- `Quality_q` 与被引证次数关系表

典型输出位置：
- `stage2/figures/fig_quality_*.png`
- `stage2/tables/tbl_desc_patent_quality.*`
- `stage2/tables/tbl_quality_citation_ols.*`

### 4.4 analyze_special_firms

对应逻辑：
- [analysis/analyze_special_firms.py](../analysis/analyze_special_firms.py)

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/special_firm_labels/firm_year_special_labels.parquet`
- `outputs/shared/special_firm_labels/special_ucc_set.parquet`

处理：
- company 口径对比
- firm-year 口径对比
- A/B/C 分组
- 事件研究风格趋势

典型输出：
- `stage2/data/company_special_panel.parquet`
- `stage2/data/patents_special_year.parquet`
- `stage2/data/company_year_special.parquet`
- `stage2/data/company_year_abc.parquet`
- `stage2/tables/tbl_firm_compare.*`
- `stage2/tables/tbl_firmyear_compare.*`
- `stage2/figures/fig_special_*.png`
- `stage2/figures/fig_abc_*.png`

### 4.5 build_firm_year_innovation

对应逻辑：
- [analysis/build_firm_year_innovation.py](../analysis/build_firm_year_innovation.py)

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`

处理：
- 按 `[UCC, year]` 将专利匹配到上市公司
- 在公司年层面对 `Quality_q` 做聚合
- 默认方法为 `TopK Mean + 年内标准化`

输出：
- `stage2/data/firm_year_innovation.parquet`
- `stage2/metadata/build_firm_year_innovation.json`

关键字段：
- `Stkid`
- `ShortName`
- `year`
- `PatentCount`
- `Innovation_raw`
- `Innovation_z`

### 4.6 run_regressions

对应逻辑：
- [analysis/run_regressions.py](../analysis/run_regressions.py)

输入：
- `stage2/data/firm_year_innovation.parquet`
- `outputs/shared/financial_panel/financial_annual_clean.parquet`

处理：
- 合并创新指标与共享财务面板
- 构造控制变量
- 构造滞后项
- 运行固定效应回归

输出：
- `stage2/data/regression_panel.parquet`
- `stage2/tables/tbl_regression_summary.*`
- `stage2/tables/回归分析/regressions/current/.../reg_*.txt`
- `stage2/tables/回归分析/regressions/future/.../reg_*.txt`
- `stage2/figures/fig_regression_coefficients.png`
- `stage2/metadata/run_regressions.json`

---

## 5. 关键 metadata 文件

### shared prep

- `outputs/shared/metadata/run_shared_prep.json`
- `outputs/shared/metadata/verify_shared_prep.json`

### per-experiment stage2

- `stage2/metadata/stage2_config.json`
- `stage2/metadata/run_stage2_pipeline.json`
- `stage2/metadata/build_experiment_patent_panel.json`
- `stage2/metadata/analyze_quality_basic.json`
- `stage2/metadata/analyze_special_firms.json`
- `stage2/metadata/build_firm_year_innovation.json`
- `stage2/metadata/run_regressions.json`

---

## 6. 推荐运行顺序

### 6.1 共享预处理

如果使用仓库默认数据位置，可以直接运行：

```bash
python analysis/run_shared_prep.py
python analysis/verify_shared_prep.py
```

### 6.2 单实验 stage2

```bash
python analysis/run_stage2_pipeline.py \
  --experiment-id 标题_摘要_window3 \
  --stage1-dir outputs/experiments/标题_摘要_window3/stage1 \
  --shared-root outputs/shared
```

或使用根目录一键入口：

```bash
python run_stage2.py
```

### 6.3 批量 stage2

```bash
python analysis/run_stage2_batch.py --manifest path/to/stage2_manifest.yaml
```

---

## 7. 数据依赖关系

```text
原始专利 CSV
  -> run_full.py
  -> outputs/experiments/<experiment_id>/stage1/patent_quality_output.csv

原始专利 CSV
特殊企业名单
UCC 原始映射数据
财务原始数据
  -> run_shared_prep.py
  -> outputs/shared/*

stage1/patent_quality_output.csv + outputs/shared/patent_master
  -> experiment_patent_panel

experiment_patent_panel
  -> analyze_quality_basic
  -> analyze_special_firms

experiment_patent_panel + outputs/shared/ucc_mapping/ucc_exploded
  -> firm_year_innovation

firm_year_innovation + outputs/shared/financial_panel/financial_annual_clean
  -> run_regressions
```
