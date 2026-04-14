# Stage2 重构方案

> 状态说明
> 本文档是本次 shared prep 改造的设计方案文档，主要用于记录重构目标、边界和拆分思路。
> 当前实现已经基本按本文方案落地。
> 文中提到的 `build_main_enriched`、`build_ucc_panel` 等“当前实现”描述，指的是**重构前的旧版 stage2**。
> 如果你要看现在的实际运行流程，请优先阅读 [README.md](../README.md)。

本文档用于定义 `stage2` 的重构目标、边界与实施计划。  
核心目标是把当前第二阶段严格拆成两部分：

1. 一次性共享预处理 pipeline
2. 每实验 stage2 分析 pipeline

共享预处理只处理与实验参数无关的数据清洗、整理、映射构造和静态字段拼接。  
每实验 stage2 只处理与当前实验 `stage1` 输出直接相关的 join、聚合、统计分析、图表、表格和回归。

## 1. 重构目标

当前 `stage2` 在一个总控里混合了两类工作：

- 与实验参数无关的基础数据整理
- 与当前实验 `BS / FS / Quality_q` 结果直接相关的分析

这导致只要 `stage1` 结果不同，`stage2` 就会把很多原本静态的数据准备工作重复做一遍。  
例如：

- 每个实验都重新扫描原始专利 CSV
- 每个实验都重新整理企业名单标签
- 每个实验都可能重新生成 UCC 面板
- 每个实验都重新清洗财务原始面板

重构后的目标是：

- 所有静态准备工作放到单独 pipeline 中，只运行一次
- 所有实验相关计算留在 stage2，每个实验单独运行
- 共享产物与实验产物在目录、命名、职责上严格分离

## 2. 重构原则

- 共享预处理产物不能放在 `outputs/experiments/<experiment_id>/stage2/` 下。
- 共享预处理不能依赖某个实验的 `BS / FS / Quality_q`。
- 每实验 stage2 只能依赖：
  - 当前实验 `stage1` 输出
  - 共享预处理产物
- 所有共享产物必须有独立 metadata，记录：
  - 输入源路径
  - 关键参数
  - 生成时间
  - 行数
  - 关键字段
- 重构后应保持分析结果口径不变，先兼容旧入口，再逐步迁移。

## 3. 目标架构

建议新增一个专门目录承载一次性预处理逻辑，例如：

```text
analysis_shared/
```

或者：

```text
prep/
```

共享产物统一写到：

```text
outputs/shared/
  patent_master/
  special_firm_labels/
  ucc_mapping/
  financial_panel/
  metadata/
  logs/
```

每实验结果仍然写到：

```text
outputs/experiments/<experiment_id>/stage2/
```

## 4. 共享预处理 pipeline 的职责

共享预处理 pipeline 只处理静态底座，不涉及实验参数。

### 4.1 patent_master

目标：

- 从原始专利 CSV 中抽取并去重
- 生成统一的专利静态主表

应包含的字段至少包括：

- `申请号`
- `申请年份`
- `统一社会信用代码`
- `被引证次数`
- `专利类型`
- 其他后续分析需要的原始字段

来源逻辑：

- 当前 [analysis/build_main_enriched.py](../analysis/build_main_enriched.py) 中“扫描原始专利目录 + 去重 + 提取辅助字段”的部分

建议输出：

- `outputs/shared/patent_master/patent_master.parquet`
- `outputs/shared/patent_master/metadata.json`

### 4.2 special_firm_labels

目标：

- 从特殊企业名单中一次性生成静态标签和 `firm-year` 标签

应包含：

- `special_ucc_set`
- `firm_year_special_labels`
- 可选的清洗后企业名单

来源逻辑：

- 当前 [analysis/analyze_special_firms.py](../analysis/analyze_special_firms.py)
- 当前 [analysis/common/analysis.py](../analysis/common/analysis.py)
  中的：
  - `load_special_panel`
  - `compute_special_ucc_set`
  - `build_firm_year_special_panel`

建议输出：

- `outputs/shared/special_firm_labels/firm_year_special_labels.parquet`
- `outputs/shared/special_firm_labels/special_ucc_set.parquet`
- `outputs/shared/special_firm_labels/metadata.json`

### 4.3 ucc_mapping

目标：

- 一次性生成上市公司及其子公司年度 UCC 面板
- 同时生成 explode 后的明细映射表

来源逻辑：

- 当前 [analysis/build_ucc_panel.py](../analysis/build_ucc_panel.py)

建议输出：

- `outputs/shared/ucc_mapping/ucc_panel.csv`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`
  列建议至少包括：
  - `Stkid`
  - `ShortName`
  - `year`
  - `UCC`
- `outputs/shared/ucc_mapping/metadata.json`

### 4.4 financial_panel

目标：

- 一次性清洗财务面板
- 只保留回归需要的年报口径公司年数据

来源逻辑：

- 当前 [analysis/run_regressions.py](../analysis/run_regressions.py)
  中“读入财务数据到与创新指标合并前”的部分

应包含的处理：

- `Accper` 转日期
- 只保留 12 月 31 日年报
- `stkcd` 标准化为 6 位
- 同公司同年保留最后一条
- 可选地提前保留回归所需字段

建议输出：

- `outputs/shared/financial_panel/financial_annual_clean.parquet`
- `outputs/shared/financial_panel/metadata.json`

## 5. 每实验 stage2 的职责

重构后的每实验 stage2 只保留与当前实验数值直接相关的部分。

### 5.1 diagnostics

保留在 stage2。

原因：

- 直接依赖当前实验 `stage1` 的 `vectors / df / vocab`

对应代码：

- [analysis/common/diagnostics.py](../analysis/common/diagnostics.py)

### 5.2 experiment_patent_panel

新增一个每实验轻量拼接步骤，替代当前 `build_main_enriched` 的重扫描逻辑。

输入：

- 当前实验 `stage1/patent_quality_output.csv`
- 共享 `patent_master.parquet`

处理：

- 按 `申请号` 进行 join
- 保留当前实验的 `BS / FS / Quality_q`
- 拼上静态专利字段

建议输出：

- `stage2/data/experiment_patent_panel.parquet`

### 5.3 analyze_quality_basic

保留在 stage2。

输入改为：

- `experiment_patent_panel.parquet`

原因：

- 图表与统计直接依赖当前实验 `Quality_q / BS`

### 5.4 analyze_special_firms

保留在 stage2，但不再自己清洗企业名单。

输入改为：

- `experiment_patent_panel.parquet`
- 共享 `firm_year_special_labels.parquet`
- 共享 `special_ucc_set`

保留的内容：

- 将标签与实验专利表 join
- company / firm-year / A-B-C 聚合
- 作图做表

### 5.5 build_firm_year_innovation

保留在 stage2，但不再读取原始 `ucc_panel.csv` 后自己 explode。

输入改为：

- `experiment_patent_panel.parquet`
- 共享 `ucc_exploded.parquet`

保留的内容：

- 与 UCC 映射表 join
- 按公司年聚合创新指标
- 按年份标准化生成 `Innovation_z`

### 5.6 run_regressions

保留在 stage2，但不再直接清洗原始财务 `dta`。

输入改为：

- `firm_year_innovation.parquet`
- 共享 `financial_annual_clean.parquet`

保留的内容：

- 合并创新指标与财务面板
- 生成滞后项
- 跑回归
- 输出表格与图

## 6. 对现有模块的拆分建议

### 6.1 替换 build_main_enriched

当前：

- [analysis/build_main_enriched.py](../analysis/build_main_enriched.py)

应拆成：

- `build_patent_master()`
  一次性共享产物
- `build_experiment_patent_panel()`
  每实验轻量 join

### 6.2 拆分 analyze_special_firms 的输入准备

当前：

- [analysis/analyze_special_firms.py](../analysis/analyze_special_firms.py)

应迁出的逻辑：

- `load_special_panel`
- `compute_special_ucc_set`
- `build_firm_year_special_panel`

保留在 stage2 的逻辑：

- 与实验专利表 join
- 动态分组
- 聚合
- 图表与表格输出

### 6.3 拆分 build_ucc_panel

当前：

- [analysis/build_ucc_panel.py](../analysis/build_ucc_panel.py)

共享 pipeline 应负责生成：

- `ucc_panel.csv`
- `ucc_exploded.parquet`

### 6.4 拆分 run_regressions

当前：

- [analysis/run_regressions.py](../analysis/run_regressions.py)

应迁出的逻辑：

- 原始财务数据读取
- 年报筛选
- 主键标准化
- 公司年去重

保留在 stage2 的逻辑：

- 合并实验创新指标
- 构造滞后项
- 固定效应回归

### 6.5 更新 run_stage2_pipeline

当前：

- [analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py)

重构后建议顺序：

1. `diagnostics`
2. `build_experiment_patent_panel`
3. `analyze_quality_basic`
4. `analyze_special_firms`
5. `build_firm_year_innovation`
6. `run_regressions`

不再在 stage2 中直接做：

- 原始专利扫描
- 特殊企业名单标准化
- UCC 面板构造
- 财务原始数据清洗

## 7. 建议新增的入口

### 7.1 `run_shared_prep.py`

职责：

- 一次性生成所有共享底座产物

建议顺序：

1. `build_patent_master`
2. `build_special_firm_labels`
3. `build_ucc_mapping`
4. `build_financial_panel`

### 7.2 `run_stage2_experiment.py`

职责：

- 运行单个实验的 stage2

建议输入：

- `experiment_id`
- `stage1_dir`
- `shared_root`

### 7.3 `verify_shared_prep.py`

职责：

- 验证共享产物是否完整、字段是否齐全、主键是否唯一、缺失率是否异常

## 8. 迁移顺序

建议按以下顺序实施：

1. 先定义共享目录结构与 metadata schema
2. 优先落地 `patent_master`
3. 落地 `ucc_mapping`
4. 落地 `financial_panel`
5. 落地 `special_firm_labels`
6. 修改 stage2，使其读取共享产物
7. 保留旧 CLI 一段时间，内部转调新函数
8. 新旧结果核对通过后，再删除旧的重复清洗路径

## 9. 验收标准

重构完成后，应满足：

1. 共享 pipeline 跑完后，不依赖任何 `experiment_id`
2. 每实验 stage2 不再扫描原始专利目录
3. 每实验 stage2 不再直接清洗原始财务 `dta`
4. 每实验 stage2 不再重新生成 `ucc_panel`
5. 每实验 stage2 不再重新构造特殊企业标签底座

同时应核对新旧流程在同一实验上的关键结果一致，包括但不限于：

- `experiment_patent_panel` 行数
- `firm_year_innovation` 行数与关键统计量
- 回归样本量与回归结果表
- 基础图表和特殊企业分析表中的核心统计值

## 10. 推荐的实施任务拆分

后续 agent 可以按以下任务顺序实施：

1. 设计共享产物目录和 metadata schema
2. 实现 `build_patent_master()`，并替换 `build_main_enriched()`
3. 实现 `build_special_firm_labels()`，并改造 `analyze_special_firms()`
4. 实现 `build_ucc_mapping()` 和 `ucc_exploded.parquet`
5. 实现 `build_financial_annual_panel()`，并改造 `run_regressions()`
6. 重写 `run_stage2_pipeline()`，只保留实验相关步骤
7. 增加对比验证脚本，核对新旧流程输出

## 11. 结论

本次重构不是简单“给 stage2 加缓存”，而是要从职责上重新划分：

- 共享预处理层负责所有静态底座
- 每实验 stage2 只负责真正与实验参数和实验结果有关的分析

这样可以显著减少重复 IO、重复清洗和重复映射构造，也能让后续代码结构更清晰，更利于维护和扩展。
