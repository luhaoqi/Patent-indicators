# 第二阶段脚本说明

`analysis/` 目录现在承担的是第二阶段的两层脚本化流程：

- **shared prep**
  一次性生成 `outputs/shared/*` 下的静态底座
- **per-experiment stage2**
  对每个实验消费 `stage1` 结果和共享产物，输出到 `outputs/experiments/<experiment_id>/stage2/`

如果和旧文档冲突，请以根目录 [README.md](../README.md) 和当前脚本实现为准。

---

## 1. 当前主入口

### shared prep

- [analysis/run_shared_prep.py](./run_shared_prep.py)
  一次性生成：
  - `outputs/shared/patent_master/`
  - `outputs/shared/raw_patent_authorized_parts/`
  - `outputs/shared/special_firm_labels/`
  - `outputs/shared/ucc_mapping/`
  - `outputs/shared/financial_panel/`

- [analysis/verify_shared_prep.py](./verify_shared_prep.py)
  校验共享产物是否存在、字段是否齐全、主键是否唯一

### 单实验 stage2

- [analysis/run_stage2_pipeline.py](./run_stage2_pipeline.py)
  当前单实验 stage2 主入口

- [analysis/run_stage2_experiment.py](./run_stage2_experiment.py)
  对 `run_stage2_pipeline.py` 的薄包装

- [run_stage2.py](../run_stage2.py)
  根目录的一键单实验入口，适合手改顶部参数后直接运行

### 批量 stage2

- [analysis/run_stage2_batch.py](./run_stage2_batch.py)
  按 manifest 批量跑多个实验

---

## 2. stage2 当前总流程

[analysis/run_stage2_pipeline.py](./run_stage2_pipeline.py) 现在按 7 步执行：

1. `diagnostics`
2. `build_experiment_patent_panel`
3. `export_top_patents_by_year`
4. `analyze_quality_basic`
5. `analyze_special_firms`
6. `build_firm_year_innovation`
7. `run_regressions`

这是**严格 shared-root 模式**：

- 必须存在 `stage1/patent_quality_output.csv`
- 必须预先存在 `outputs/shared/*`
- stage2 内部不再扫描原始专利目录
- stage2 内部不再清洗原始财务数据
- stage2 内部不再重建 UCC 面板
- stage2 内部不再重建特殊企业标签底座

---

## 3. shared prep 子脚本

### `build_main_enriched.py`

当前承担两件事：

- `build_patent_master()`
  从原始专利 CSV 生成共享 `patent_master`
- `build_experiment_patent_panel()`
  将 `stage1/patent_quality_output.csv` 与共享 `patent_master` 轻量拼接

关键输出：

- `outputs/shared/patent_master/patent_master.parquet`
- `outputs/experiments/<experiment_id>/stage2/data/experiment_patent_panel.parquet`

### `build_ucc_panel.py`

当前提供：

- `build_ucc_mapping()`
  生成共享：
  - `outputs/shared/ucc_mapping/ucc_panel.csv`
  - `outputs/shared/ucc_mapping/ucc_exploded.parquet`

### `shared_prep.py`

当前提供：

- `build_special_firm_labels()`
- `build_financial_annual_panel()`
- `verify_shared_prep()`

### `build_raw_patent_authorized_parts.py`

当前提供：

- `build_raw_patent_authorized_parts()`
  将 `data/raw/中国专利分年份保存数据1985-2025/*.csv` 按原文件拆分转换为 parquet
  - 保留原始全部列
  - 仅保留 `专利类型 == 发明授权`
  - 输出到 `outputs/shared/raw_patent_authorized_parts/*.parquet`
  - `run_shared_prep.py` 会自动调用这一步

---

## 4. per-experiment stage2 子脚本

### `analyze_quality_basic.py`

输入：
- `stage2/data/experiment_patent_panel.parquet`

输出：
- 专利层基础图表和描述统计

### `export_top_patents_by_year.py`

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`
- `outputs/shared/raw_patent_authorized_parts/*.parquet`

输出：
- `stage2/tables/top_patents_by_year/*.csv`
- 按 `申请年份` 导出 `Quality_q` 排名前 `top_n` 的专利明细
- 默认先对 `experiment_patent_panel.parquet` 做流式 top_n 筛选，再补公司名与原始明细
- 原始明细回查现在只走 `outputs/shared/raw_patent_authorized_parts/*.parquet`
- 原始明细回查仅用于补充摘要、申请日、公告日等字段；如果只需要排名结果，可用 `--skip-raw-lookup`
- UCC 公司名映射仅用于补 `公司名称/证券ID`；如果只接受申请人回退，可用 `--skip-company-lookup`

### `analyze_special_firms.py`

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/special_firm_labels/firm_year_special_labels.parquet`
- `outputs/shared/special_firm_labels/special_ucc_set.parquet`

输出：
- 特殊企业静态、动态、A/B/C 分组分析结果

### `build_firm_year_innovation.py`

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`

输出：
- `stage2/data/firm_year_innovation.parquet`

### `run_regressions.py`

输入：
- `stage2/data/firm_year_innovation.parquet`
- `outputs/shared/financial_panel/financial_annual_clean.parquet`

输出：
- `stage2/data/regression_panel.parquet`
- 回归表和系数图

---

## 5. 公共模块

### `analysis/common/paths.py`

作用：
- 统一 experiment 路径
- 统一 shared 路径
- 解析仓库内相对路径

### `analysis/common/io.py`

作用：
- CSV 读取
- 日志构建
- JSON metadata 写入

### `analysis/common/config.py`

作用：
- 统一 stage2 配置结构
- 按任务分组保存参数：
  - `inputs`
  - `diagnostics`
  - `build_experiment_patent_panel`
  - `analyze_quality_basic`
  - `analyze_special_firms`
  - `build_firm_year_innovation`
  - `run_regressions`

### `analysis/common/analysis.py`

作用：
- 样本过滤
- 特殊企业标签构造
- 公司层、公司年层、A/B/C 分组聚合

### `analysis/common/diagnostics.py`

作用：
- stage1 diagnostics 输出逻辑

---

## 6. 常用运行方式

先生成共享产物：

```bash
python analysis/run_shared_prep.py
python analysis/verify_shared_prep.py
```

再跑单实验 stage2：

```bash
python analysis/run_stage2_pipeline.py \
  --experiment-id 标题_摘要_window5 \
  --stage1-dir outputs/experiments/标题_摘要_window5/stage1 \
  --shared-root outputs/shared
```

批量跑：

```bash
python analysis/run_stage2_batch.py --manifest path/to/stage2_manifest.yaml
```

---

## 7. 独立运行子脚本

每个步骤脚本也支持独立运行，可用 `python analysis/<脚本名>.py --help` 查看参数，方便调试单步。

## 8. 历史文档说明

- [docs/STAGE2_REFACTOR_PLAN.md](../docs/STAGE2_REFACTOR_PLAN.md) 是 shared prep 改造方案文档，记录了重构设计思路。
