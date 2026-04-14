# 项目脚本盘点与整理建议

本文档盘点项目中所有可独立运行的 Python 脚本（含 `if __name__ == "__main__"` 的文件），标注其在文档中的覆盖情况、当前状态和整理建议。

> 纯库模块（`patent_quality/*.py`、`analysis/common/*.py`、`analysis/shared_prep.py`、`analysis/special_firm_regressions.py` 等无 `__main__` 的文件）不在本文档范围内。

---

## 1. 状态说明

| 标记 | 含义 |
|------|------|
| **当前** | 功能正常、仍被主流程或日常使用 |
| **已整合** | 功能已被 pipeline 总控脚本整合，独立运行的需求很小 |
| **一次性** | 为特定分析场景编写，路径硬编码，不属于常规流程 |
| **可废弃** | 引用旧路径或已被完全取代 |

---

## 2. README.md 已记录的入口（13 个）

均为当前状态，文档覆盖完整，无需补充。

| 脚本 | 用途 | README 章节 |
|------|------|------------|
| `run_full.py` | stage1 主入口 | §2.1, §4.4 |
| `run_stage2.py` | stage2 单实验一键入口 | §2.3, §6.3 |
| `analysis/run_shared_prep.py` | shared prep 入口 | §2.2, §5.2 |
| `analysis/verify_shared_prep.py` | 共享产物校验 | §2.2, §5.2 |
| `analysis/run_stage2_pipeline.py` | stage2 单实验总控 | §2.3, §6.3 |
| `analysis/run_stage2_experiment.py` | stage2 薄包装入口 | §2.3 |
| `analysis/run_stage2_batch.py` | stage2 批量入口 | §2.3 |
| `verify_ir.py` | 抽样验证 IR/相似度计算结果 | §2.4 |
| `inspect_patent_similarity_case.py` | exact 模式单专利相似度拆解 | §2.4, §4.5 |
| `profile_matrix.py` | 检查稀疏矩阵规模与分布 | §2.4 |
| `search_exact_time_patents.py` | exact 实验批量排名查询 | §4.6 |
| `tests/test_small.py` | stage1 小样本 smoke test | §2.4 |
| `tests/test_stage2_refactor.py` | shared prep 与 stage2 链路测试 | §2.4 |

---

## 3. USAGE.md 提到但 README.md 遗漏的（1 个）

| 脚本 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `inspect_patent_case.py` | 非 exact 模式的单专利个例分析：追踪某条专利在 stage1 中哪些词被保留/舍弃、各词贡献多少。支持单条运行和 `--cases-manifest` 批量运行 | **当前** | 补充到 README §2.4；它是 `inspect_patent_similarity_case.py` 的姐妹脚本，面向非 exact 实验 |

---

## 4. 未被任何文档记录的独立入口

### 4.1 根目录脚本（3 个）

| 脚本 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `compare_stage1_outputs.py` | 对比新旧 stage1 输出，默认比较 `outputs/experiments/标题_摘要_window3/stage1` vs 旧的 `data/result/标题+摘要版本+window_size=3`，生成 markdown 对比报告到 `outputs/comparisons/` | **可废弃** — 默认路径引用已废弃的 `data/result/`，是迁移验证用的一次性工具 | 如果不再需要对比旧产物可以删除；保留则需更新默认路径 |
| `verify_patent_exact_time.py` | 从头对单条专利重跑 exact 模式全流程（分词→向量化→剪枝→相似度），并与 stage1 已有结果逐步比对，验证正确性。输出到 `outputs/tests/verify_patent_exact_time/` | **当前** — 有用的端到端验证工具 | 补充到 README §2.4 |
| `aggregate_award_exact_time_top_rank_percents.py` | 汇总多个专利金奖 `*_exact_time_lookup.csv` 查询结果，提取每个文件最小排名百分比 Top N，导出全局排名 | **一次性** — 配合 `search_exact_time_patents.py` 后处理用 | 如果还会做金奖专利分析就保留并补充文档；否则可删除 |

### 4.2 analysis/ 下的独立 diagnostics 小工具（6 个）

这些脚本分别从 `common.diagnostics` 调用单个诊断函数，允许单独运行某一项 diagnostics 指标。**它们的全部功能已被 `run_stage2_pipeline.py` 的 diagnostics 步骤整合**。

| 脚本 | 计算内容 | 状态 | 建议 |
|------|---------|------|------|
| `analysis/run_diagnostics.py` | 完整 diagnostics 并标准化输出到 experiment 目录 | **已整合** | 可废弃——pipeline 已整合 |
| `analysis/calc_avg_vocab_usage.py` | 平均词汇使用率 | **已整合** | 同上 |
| `analysis/calc_df_pair_sum.py` | DF pair sum | **已整合** | 同上 |
| `analysis/calc_topk_df_pair_sum.py` | Top-K pair sum | **已整合** | 同上 |
| `analysis/calc_yearly_top_vocab.py` | 年度高频词 | **已整合** | 同上 |
| `analysis/calc_yearly_vocab_size.py` | 年度词表大小 | **已整合** | 同上 |

### 4.3 analysis/ 下的一次性提取脚本（1 个）

| 脚本 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `analysis/extract_exacttime_regression_results.py` | 提取 window_1 和 window_3 两个 exact 实验的回归结果，生成对比 CSV 和 markdown 到 `outputs/experiments/标题_摘要_ExactTime_window_1_vs_3_regression_extract/` | **一次性** — 路径全部硬编码 | 保留作结果导出参考，不需写入 README |

---

## 5. 被 pipeline 调用、也有 `__main__` 的步骤模块（8 个）

这些是 `run_shared_prep.py` 或 `run_stage2_pipeline.py` 内部调用的函数模块，同时也支持独立运行（带 argparse）。正常流程中不需要用户直接调用，但调试时可以单步运行某一步。

| 脚本 | 被谁调用 | 独立运行用途 |
|------|---------|------------|
| `analysis/build_main_enriched.py` | `run_shared_prep.py` + `run_stage2_pipeline.py` | 单独构建 patent_master 或 experiment_patent_panel |
| `analysis/build_raw_patent_authorized_parts.py` | `run_shared_prep.py` | 单独构建授权专利 parquet |
| `analysis/build_ucc_panel.py` | `run_shared_prep.py` | 单独构建 UCC 面板 |
| `analysis/analyze_quality_basic.py` | `run_stage2_pipeline.py` | 单独运行专利质量描述统计 |
| `analysis/export_top_patents_by_year.py` | `run_stage2_pipeline.py` | 单独导出年度 Top 专利 |
| `analysis/analyze_special_firms.py` | `run_stage2_pipeline.py` | 单独运行特殊企业分析 |
| `analysis/build_firm_year_innovation.py` | `run_stage2_pipeline.py` | 单独构建公司年创新指数 |
| `analysis/run_regressions.py` | `run_stage2_pipeline.py` | 单独运行回归 |

**建议：** 不需要在 README 中逐个列出，可在 `analysis/README.md` 补一句"每个步骤脚本也支持 `python analysis/xxx.py --help` 独立运行"。

---

## 6. 测试文件

README 已记录 `test_small.py` 和 `test_stage2_refactor.py`。以下 8 个测试未在文档中提及：

| 脚本 | 测试内容 |
|------|---------|
| `tests/test_exact_date.py` | exact 模式流程正确性 |
| `tests/test_patent_case_analysis.py` | 专利个例分析逻辑 |
| `tests/test_diagnostics_multi_topk.py` | diagnostics 多 TopK 值 |
| `tests/test_special_firms_equivalence.py` | 特殊企业分析等价性 |
| `tests/test_special_firm_regressions.py` | 特殊企业回归 |
| `tests/test_stage2_regressions.py` | stage2 回归 |
| `tests/test_similarity_case_analysis.py` | 相似度拆解分析 |
| `tests/test_search_exact_time_patents.py` | 批量查询脚本 |

**建议：** README 中补一句"完整测试套件见 `tests/` 目录，运行 `python -m pytest tests/` 可执行全部测试"即可。

---

## 7. 整理行动清单

### 7.1 需要补充到 README.md 的（2 个）

- [ ] `inspect_patent_case.py` — 补充到 §2.4 "验证与辅助入口"
- [ ] `verify_patent_exact_time.py` — 补充到 §2.4 "验证与辅助入口"

### 7.2 建议删除的脚本（8 个）

- [ ] `compare_stage1_outputs.py` — 引用已废弃的 `data/result/` 旧路径
- [ ] `analysis/run_diagnostics.py` — 功能已被 pipeline 整合
- [ ] `analysis/calc_avg_vocab_usage.py` — 同上
- [ ] `analysis/calc_df_pair_sum.py` — 同上
- [ ] `analysis/calc_topk_df_pair_sum.py` — 同上
- [ ] `analysis/calc_yearly_top_vocab.py` — 同上
- [ ] `analysis/calc_yearly_vocab_size.py` — 同上
- [ ] `analysis/extract_exacttime_regression_results.py` — 硬编码路径的一次性结果提取

### 7.3 可留可删的脚本（1 个）

- [ ] `aggregate_award_exact_time_top_rank_percents.py` — 如果还会做金奖专利分析就保留

### 7.4 README.md 中关于测试的补充

- [ ] 补充一句"完整测试套件见 `tests/` 目录"
