# 文档索引

本目录用于说明项目结构、运行方式和设计思路。  
项目现在已经形成两阶段的完整实验流程：

- `stage1`：专利文本指标计算
- `stage2`：共享预处理 + 每实验统计分析、企业分组、公司年创新指数与回归

如果你只想快速理解项目，建议先读根目录的 [README.md](../README.md)。

---

## 建议阅读顺序

1. [README.md](../README.md)
   - 项目整体目标
   - stage1 / stage2 完整流程
   - 输入输出和实验目录结构

2. [USAGE.md](./USAGE.md)
   - 第一阶段最小运行示例

3. [STRUCTURE_GUIDE.md](./STRUCTURE_GUIDE.md)
   - 数据目录和输出目录推荐结构

4. [STAGE2_REFACTOR_PLAN.md](./STAGE2_REFACTOR_PLAN.md)
   - 第二阶段重构方案：共享预处理层与每实验分析层拆分

5. [严格发布时间_exact_date改造方案.md](./严格发布时间_exact_date改造方案.md)
   - `stage1_exact / stage2_exact` 方案，按 `公开公告日 / 公开公告年份` 计算严格日期窗口

6. [analysis/README.md](../analysis/README.md)
   - 第二阶段当前脚本入口和 shared prep / stage2 结构

7. [阶段1_阶段2提速改造方案.md](./阶段1_阶段2提速改造方案.md)
   - 性能优化与验证思路

---

## 当前项目状态

### 第一阶段

第一阶段已经是稳定的 pipeline，入口是：

- [run_full.py](../run_full.py)

输出写到：

```text
outputs/experiments/<experiment_id>/stage1/
```

### 第二阶段

第二阶段当前入口分成 shared prep 和 per-experiment stage2 两类：

- shared prep：
  - [analysis/run_shared_prep.py](../analysis/run_shared_prep.py)
  - [analysis/verify_shared_prep.py](../analysis/verify_shared_prep.py)

- per-experiment stage2：
  - [run_stage2.py](../run_stage2.py)
  - [analysis/run_stage2_pipeline.py](../analysis/run_stage2_pipeline.py)
  - [analysis/run_stage2_experiment.py](../analysis/run_stage2_experiment.py)
  - [analysis/run_stage2_batch.py](../analysis/run_stage2_batch.py)

共享产物写到：

```text
outputs/shared/
```

输出写到：

```text
outputs/experiments/<experiment_id>/stage2/
```

每实验 `stage2` 会自动生成：

- `data/`
- `diagnostics/`
- `figures/`
- `tables/`
- `logs/`
- `metadata/`

其中：
- `metadata/stage2_config.json` 会记录完整参数快照
- 参数按任务分组保存，而不是全部平铺

---

## 这份文档目录不再承担什么

过去 `docs/` 和 `analysis/` 中很多说明默认假设第二阶段主要靠人工执行，或者默认使用重构前的 7 步旧流程。现在这个前提已经改变：

- 正式结果以脚本输出为准
- notebook 主要作为历史记录和探索入口保留
- 批量比较实验应使用 experiment 目录、shared prep 和脚本总控

所以如果某个旧文档和根目录 `README.md` 冲突，请以根目录 `README.md` 和当前脚本实现为准。
