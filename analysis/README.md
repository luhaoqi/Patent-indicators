# 第二阶段脚本说明

`analysis/` 目录现在承担的是**第二阶段（stage2）完整脚本化流程**，不再只是 notebook 辅助目录。

第二阶段的目标是：  
给定某个实验的 `stage1` 输出目录，自动完成：

- diagnostics
- 主分析表构造
- 专利层基础图表与描述统计
- 特殊企业 / 专精特新企业对比
- `UCC` 面板构造
- 公司年度创新指数构造
- 财务面板固定效应回归

所有结果统一写到：

```text
outputs/experiments/<experiment_id>/stage2/
```

---

## 1. 第二阶段主入口

### 单实验总控

- [analysis/run_stage2_pipeline.py](./run_stage2_pipeline.py)

作用：
- 串行调用完整 `stage2`
- 自动记录 `logs/` 和 `metadata/`
- 根据可用输入自动跳过某些可选步骤

### 批量入口

- [analysis/run_stage2_batch.py](./run_stage2_batch.py)

作用：
- 读取 manifest
- 批量对多个 `experiment_id` 运行 `stage2`
- 每个实验独立写入自己的 `stage2/` 目录

### 单实验一键入口

仓库根目录下还有：

- [run_stage2.py](../run_stage2.py)

这个入口适合像 `run_full.py` 一样在顶部集中改参数后直接运行。

---

## 2. 公共模块

### `analysis/common/paths.py`

作用：
- 统一 experiment 目录定位
- 解析仓库内相对路径
- 生成 `stage2/data`、`figures`、`tables` 等目录对象

### `analysis/common/io.py`

作用：
- 带编码回退的 CSV 读取
- 日志构建
- JSON metadata 写入

### `analysis/common/config.py`

作用：
- 统一 `stage2` 配置结构
- 按任务分组保存参数：
  - `inputs`
  - `diagnostics`
  - `build_main_enriched`
  - `analyze_quality_basic`
  - `analyze_special_firms`
  - `build_firm_year_innovation`
  - `run_regressions`

### `analysis/common/analysis.py`

作用：
- 样本过滤
- 描述统计
- 特殊企业静态 / 动态标签构造
- 公司层、公司年层、A/B/C 分组聚合
- 分组对比表构造

### `analysis/common/plotting.py`

作用：
- 中文字体设置
- 图片保存

### `analysis/common/tables.py`

作用：
- 表格格式化
- `csv` / `tex` 双落盘

### `analysis/common/diagnostics.py`

作用：
- 封装第一阶段 diagnostics 计算逻辑

---

## 3. 第二阶段每个脚本做什么

### `run_diagnostics.py`

输入：
- `stage1/df/`
- `stage1/vectors/`
- `stage1/vocab/`

输出：
- `stage2/diagnostics/*.csv`

作用：
- 统一运行 diagnostics，并标准化保存输出。

### `build_main_enriched.py`

输入：
- `stage1/patent_quality_output.csv`
- 原始按年专利 CSV

输出：
- `stage2/data/patent_quality_output.csv`
- `stage2/data/main.parquet`
- `stage2/data/extra_all_dedup.parquet`
- `stage2/data/main_enriched.parquet`

作用：
- 把第一阶段结果和原始专利补充字段拼接成第二阶段主表。

### `analyze_quality_basic.py`

输入：
- `stage2/data/main_enriched.parquet`

输出：
- `stage2/figures/fig_quality_*.png`
- `stage2/tables/tbl_desc_patent_quality.*`
- `stage2/tables/tbl_quality_citation_ols.*`

作用：
- 输出专利层基础图表和描述统计。

### `analyze_special_firms.py`

输入：
- `stage2/data/main_enriched.parquet`
- 特殊企业名单 `.dta`

输出：
- `stage2/data/company_special_panel.parquet`
- `stage2/data/company_year_special.parquet`
- `stage2/data/company_year_abc.parquet`
- `stage2/tables/tbl_firm_compare.*`
- `stage2/tables/tbl_firmyear_compare.*`
- `stage2/figures/fig_special_*.png`
- `stage2/figures/fig_abc_*.png`

作用：
- 输出静态企业口径、动态 `firm-year` 口径和 A/B/C 分组分析结果。

### `build_ucc_panel.py`

输入：
- 母公司统一社会信用代码表
- 子公司名称映射表
- 联营合营明细表

输出：
- `stage2/data/ucc_panel.csv`

作用：
- 生成上市公司（包含子公司）的年度统一社会信用代码面板。

### `build_firm_year_innovation.py`

输入：
- `stage2/data/main_enriched.parquet`
- `stage2/data/ucc_panel.csv` 或外部 UCC 面板

输出：
- `stage2/data/firm_year_innovation.parquet`

作用：
- 计算公司-年份创新指标，当前默认方法是 `TopK Mean + 年内标准化`。

### `run_regressions.py`

输入：
- `stage2/data/firm_year_innovation.parquet`
- 财务面板 `.dta`

输出：
- `stage2/data/regression_panel.parquet`
- `stage2/tables/tbl_regression_summary.*`
- `stage2/tables/reg_*.txt`
- `stage2/figures/fig_regression_coefficients.png`

作用：
- 运行固定效应回归并输出摘要结果。

---

## 4. diagnostics 独立脚本

以下脚本仍然保留，也可以单独运行：

- `calc_avg_vocab_usage.py`
- `calc_df_pair_sum.py`
- `calc_topk_df_pair_sum.py`
- `calc_yearly_top_vocab.py`
- `calc_yearly_vocab_size.py`

这些脚本主要直接读取 `stage1` 中间产物，用于做诊断、停用词检查和 TopK 影响分析。  
但在常规使用中，一般不需要单独手动调用，因为 `run_stage2_pipeline.py` 已经会统一调用它们。

---

## 5. 目录结构

```text
outputs/experiments/<experiment_id>/stage2/
  data/
  diagnostics/
  figures/
  tables/
  logs/
  metadata/
```

其中最重要的 metadata 文件包括：

- `stage2/metadata/stage2_config.json`
- `stage2/metadata/build_main_enriched.json`
- `stage2/metadata/analyze_quality_basic.json`
- `stage2/metadata/analyze_special_firms.json`
- `stage2/metadata/build_firm_year_innovation.json`
- `stage2/metadata/run_regressions.json`

---

## 6. 常用运行方式

### 一键运行单实验

```bash
python run_stage2.py
```

### 命令行运行单实验

```bash
python analysis/run_stage2_pipeline.py \
  --experiment-id 标题_摘要_window5 \
  --stage1-dir outputs/experiments/标题_摘要_window5/stage1 \
  --raw-patent-dir data/raw/中国专利分年份保存数据1985-2025
```

### 批量运行多个实验

```bash
python analysis/run_stage2_batch.py --manifest path/to/stage2_manifest.yaml
```

---

## 7. 参数组织方式

`stage2` 的参数现在按任务分组组织，而不是全部平铺：

- `inputs`
- `diagnostics`
- `build_main_enriched`
- `analyze_quality_basic`
- `analyze_special_firms`
- `build_firm_year_innovation`
- `run_regressions`

这意味着：
- 看入口文件时，能直接知道某个参数控制哪个步骤
- 看实验目录时，能直接从 `stage2_config.json` 判断每张图、每张表对应的参数口径

---

## 8. notebook 的角色

`analysis/graph/` 和 `analysis/公司财务/` 中的 notebook 现在主要保留作：

- 历史分析记录
- 临时探索
- 结果检查

当前常规研究流程不再依赖这些 notebook 作为主入口。  
正式生成结果应以脚本和 experiment 目录下的输出为准。
