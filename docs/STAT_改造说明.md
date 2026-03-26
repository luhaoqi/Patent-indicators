# 第二阶段统计分析改造说明

> 注意
> 本文档保留的是 stage2 从 notebook 向脚本化迁移的**历史背景**。
> 当前实现已经进一步重构为“共享预处理层 + 每实验 stage2 分析层”。
> 如果你要看现在实际可运行的入口、目录结构和依赖关系，请优先阅读：
> [README.md](../README.md)、[analysis/README.md](../analysis/README.md)、[STAGE2_FLOW_DETAILED.md](./STAGE2_FLOW_DETAILED.md)。

## 1. 这份文档记录什么

这份文档记录的是一个历史阶段：

- 第二阶段最初主要靠多个 notebook 人工串行执行
- 后来逐步迁移到围绕 experiment 目录的脚本化流程

它的价值主要在于解释：

- 当时为什么要做脚本化
- 为什么统一 experiment 输出目录、日志和 metadata
- 为什么把参数改成按任务分组保存

它**不再**用于描述当前 stage2 的具体运行步骤。

---

## 2. 当时要解决的问题

早期 notebook 流程主要有这些问题：

1. 图表、对比分析和回归散落在不同 notebook 中
2. 输出经常写回 `analysis/graph/` 或 `analysis/公司财务/`，容易互相覆盖
3. 不方便比较多组实验参数
4. 图、表、日志和中间数据缺少固定目录
5. 参数散落在 notebook 单元格中，不利于追踪和复现

这些问题推动了 stage2 的第一次脚本化改造。

---

## 3. 第一次脚本化改造留下了什么

这个阶段沉淀下来的核心结果包括：

### 3.1 experiment 目录结构

统一使用：

```text
outputs/experiments/<experiment_id>/stage2/
```

并固定拆分为：

- `data/`
- `diagnostics/`
- `figures/`
- `tables/`
- `logs/`
- `metadata/`

### 3.2 公共模块层

逐步沉淀出：

- `analysis/common/paths.py`
- `analysis/common/io.py`
- `analysis/common/config.py`
- `analysis/common/analysis.py`
- `analysis/common/plotting.py`
- `analysis/common/tables.py`
- `analysis/common/diagnostics.py`

这些模块把路径、日志、配置、表格、绘图和统计逻辑统一抽了出来。

### 3.3 脚本入口层

逐步形成了：

- [run_stage2.py](../run_stage2.py)
- [analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py)
- [analysis/run_stage2_batch.py](../analysis/run_stage2_batch.py)

这为后续 shared prep 重构打下了入口层基础。

### 3.4 metadata 快照

每次运行会把参数快照落盘到：

```text
outputs/experiments/<experiment_id>/stage2/metadata/stage2_config.json
```

这是后续可复现和实验对比的重要基础。

---

## 4. 和当前实现的关系

当前实现是在“第一次脚本化改造”的基础上继续往前走了一步：

- 把原先混在 stage2 里的静态准备工作拆成了 shared prep
- 把每实验 stage2 收紧成只做与当前实验直接相关的分析

现在实际应理解为：

```text
shared prep
  -> outputs/shared/*

per-experiment stage2
  -> outputs/experiments/<experiment_id>/stage2/
```

---

## 5. 当前应参考哪些文档

如果你要了解当前实现，请按这个顺序看：

1. [README.md](../README.md)
2. [analysis/README.md](../analysis/README.md)
3. [STAGE2_FLOW_DETAILED.md](./STAGE2_FLOW_DETAILED.md)
4. [STAGE2_REFACTOR_PLAN.md](./STAGE2_REFACTOR_PLAN.md)

如果你想理解“为什么当初要把 notebook 流程脚本化”，再回来看这份文档即可。
