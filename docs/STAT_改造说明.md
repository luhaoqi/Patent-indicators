# 第二阶段统计分析改造说明

本文档用于说明第二阶段（`stage2`）的改造背景、目标和当前落地结果。

结论先说：  
第二阶段已经从“多个 notebook 的人工串行操作”改造成“围绕 experiment 目录的一套脚本化流程”。

---

## 1. 改造前的问题

原先的第二阶段主要存在这些问题：

1. 图表、对比分析和回归主体集中在 notebook 中
2. 输出经常写回 `analysis/graph/` 或 `analysis/公司财务/`，容易互相覆盖
3. 不方便比较多组实验参数
4. 图片、表格、日志缺少统一命名和固定目录
5. 参数分散在 notebook 单元格中，不利于追踪和复现

这些问题的核心不是“分析逻辑不够”，而是“流程没有工程化”。

---

## 2. 改造目标

本次改造的目标是：

1. 给定任意一组 `stage1` 结果目录，自动完成 `stage2`
2. 每组实验结果独立落盘，互不覆盖
3. 将 notebook 中稳定的逻辑迁移为 Python 脚本
4. 图、表、日志、中间数据都进入固定 experiment 目录
5. 支持一键单实验运行和多实验批量运行
6. 参数按任务分组保存，便于复现和对比

---

## 3. 改造后的第二阶段结构

### 3.1 主入口

现在第二阶段主入口有三类：

- 单实验一键入口：
  - [run_stage2.py](../run_stage2.py)
- 单实验脚本总控：
  - [analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py)
- 多实验批量入口：
  - [analysis/run_stage2_batch.py](../analysis/run_stage2_batch.py)

### 3.2 公共层

新增的公共模块包括：

- [analysis/common/paths.py](../analysis/common/paths.py)
- [analysis/common/io.py](../analysis/common/io.py)
- [analysis/common/config.py](../analysis/common/config.py)
- [analysis/common/analysis.py](../analysis/common/analysis.py)
- [analysis/common/plotting.py](../analysis/common/plotting.py)
- [analysis/common/tables.py](../analysis/common/tables.py)
- [analysis/common/diagnostics.py](../analysis/common/diagnostics.py)

这些公共层统一承担：

- experiment 目录管理
- 路径解析
- 日志
- 配置快照
- 图片 / 表格保存
- 样本过滤
- 分组统计与对比表构造

---

## 4. 改造后的第二阶段步骤

现在 `stage2` 总控包含 7 个顺序步骤：

1. `diagnostics`
2. `build_main_enriched`
3. `analyze_quality_basic`
4. `analyze_special_firms`
5. `build_ucc_panel`
6. `build_firm_year_innovation`
7. `run_regressions`

### 4.1 diagnostics

脚本：
- [analysis/run_diagnostics.py](../analysis/run_diagnostics.py)

来源：
- 原 `calc_*.py` 脚本

状态：
- 已脚本化并已接入总控

### 4.2 主分析表构造

脚本：
- [analysis/build_main_enriched.py](../analysis/build_main_enriched.py)

来源：
- 原 `analysis/graph/合并数据.ipynb`

状态：
- 已脚本化并已接入总控

### 4.3 基础图表与描述统计

脚本：
- [analysis/analyze_quality_basic.py](../analysis/analyze_quality_basic.py)

来源：
- 原 `analysis/graph/graph.ipynb`

状态：
- 已脚本化并已接入总控

### 4.4 特殊企业 / firm-year / A-B-C 分组分析

脚本：
- [analysis/analyze_special_firms.py](../analysis/analyze_special_firms.py)

来源：
- 原 `analysis/graph/graph.ipynb`
- 原 `analysis/graph/graph_with_firmyear_special.ipynb`

状态：
- 已脚本化并已接入总控

### 4.5 UCC 面板构造

脚本：
- [analysis/build_ucc_panel.py](../analysis/build_ucc_panel.py)

来源：
- 原 `analysis/公司财务/上市公司子公司.ipynb`

状态：
- 已脚本化并已接入总控

### 4.6 公司年创新指数

脚本：
- [analysis/build_firm_year_innovation.py](../analysis/build_firm_year_innovation.py)

来源：
- 原 `analysis/公司财务/公司创新指数计算.ipynb`

状态：
- 已脚本化并已接入总控

### 4.7 财务回归

脚本：
- [analysis/run_regressions.py](../analysis/run_regressions.py)

来源：
- 原 `analysis/公司财务/reg.ipynb`

状态：
- 已脚本化并已接入总控

---

## 5. 输出目录改造结果

过去第二阶段常把结果写回 `analysis/graph/` 或 `analysis/公司财务/`。  
现在统一写入：

```text
outputs/experiments/<experiment_id>/stage2/
```

目录结构为：

```text
stage2/
  data/
  diagnostics/
  figures/
  tables/
  logs/
  metadata/
```

这带来三个直接好处：

1. 不同实验结果天然隔离
2. 文件名可以保持稳定，不需要把参数塞进每个文件名
3. 后续做实验对比时，只需要比较不同 experiment 目录

---

## 6. 参数管理改造结果

过去 `stage2` 参数通常散在 notebook 的不同单元格里。  
现在参数已经改为**按任务分组**管理：

- `inputs`
- `diagnostics`
- `build_main_enriched`
- `analyze_quality_basic`
- `analyze_special_firms`
- `build_firm_year_innovation`
- `run_regressions`

对应配置结构见：

- [analysis/common/config.py](../analysis/common/config.py)

并且每次运行都会自动落盘：

```text
outputs/experiments/<experiment_id>/stage2/metadata/stage2_config.json
```

这样做的意义是：

- 一眼看出某个参数属于哪个步骤
- 方便后续按任务排查结果差异
- 方便比较两个 experiment 的配置快照

---

## 7. 当前第二阶段的运行方式

### 单实验一键

```bash
python run_stage2.py
```

### 单实验命令行

```bash
python analysis/run_stage2_pipeline.py \
  --experiment-id <experiment_id> \
  --stage1-dir outputs/experiments/<experiment_id>/stage1 \
  --raw-patent-dir data/raw/中国专利分年份保存数据1985-2025
```

### 多实验批量

```bash
python analysis/run_stage2_batch.py --manifest path/to/manifest.yaml
```

---

## 8. 现在 notebook 的角色

改造完成后，notebook 不再承担正式流程入口。

它们现在主要保留作：

- 历史分析记录
- 口径检查
- 临时探索

正式结果以脚本总控输出为准。

---

## 9. 当前状态总结

第二阶段当前已经达到以下状态：

- 可以给定某个 `stage1` 目录自动完成完整 `stage2`
- 可以把图、表、数据、日志统一输出到 experiment 目录
- 可以记录完整参数快照
- 可以按实验批量比较结果

也就是说，第二阶段已经从“研究型 notebook 集合”变成了“可重复实验 pipeline”。

剩下的工作主要不再是流程工程化，而是：

- 根据研究需要继续增加新的分析规格
- 优化图表风格或表格格式
- 在已有脚本框架上扩展新的对比口径
