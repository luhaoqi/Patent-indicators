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
   - diagnostics
   - 专利层图表与描述统计
   - 特殊企业 / 专精特新企业对比
   - 公司年度创新指数构造
   - 与上市公司财务面板的固定效应回归

当前第二阶段已经按职责拆成两层：

- **共享预处理层**
  一次性生成 `outputs/shared/*` 下的静态底座
- **每实验分析层**
  每个实验只消费 `stage1` 结果和共享产物，输出到 `outputs/experiments/<experiment_id>/stage2/`

---

## 2. 当前主程序入口

### 2.1 stage1 入口

- [run_full.py](run_full.py)

作用：
- 运行单个 stage1 实验
- 输出到 `outputs/experiments/<experiment_id>/stage1/`

说明：
- 仓库当前**没有** `run_full_30years.py`

### 2.2 shared prep 入口

- [analysis/run_shared_prep.py](analysis/run_shared_prep.py)

作用：
- 一次性生成第二阶段共享产物
- 输出到 `outputs/shared/`

会生成的共享目录：
- `outputs/shared/patent_master/`
- `outputs/shared/special_firm_labels/`
- `outputs/shared/ucc_mapping/`
- `outputs/shared/financial_panel/`
- `outputs/shared/metadata/`
- `outputs/shared/logs/`

- [analysis/verify_shared_prep.py](analysis/verify_shared_prep.py)

作用：
- 检查共享产物是否存在
- 检查关键字段是否齐全
- 检查主键是否唯一

### 2.3 stage2 入口

- [run_stage2.py](run_stage2.py)
  适合在文件顶部集中修改参数后直接运行单实验

- [analysis/run_stage2_pipeline.py](analysis/run_stage2_pipeline.py)
  单实验 stage2 主入口

- [analysis/run_stage2_experiment.py](analysis/run_stage2_experiment.py)
  对 `run_stage2_pipeline.py` 的薄包装入口

- [analysis/run_stage2_batch.py](analysis/run_stage2_batch.py)
  按 manifest 批量运行多个实验

### 2.4 验证与辅助入口

- [verify_ir.py](verify_ir.py)
  抽样验证 IR / 相似度计算结果

- [inspect_patent_similarity_case.py](inspect_patent_similarity_case.py)
  exact 模式下按单个专利展开词项贡献与前后窗口相似度明细

- [profile_matrix.py](profile_matrix.py)
  检查稀疏矩阵规模与分布

- [tests/test_small.py](tests/test_small.py)
  stage1 小样本 smoke test

- [tests/test_stage2_refactor.py](tests/test_stage2_refactor.py)
  shared prep 与新 stage2 输入链路测试

---

## 3. 仓库结构

```text
patent_quality/              第一阶段主流程代码
run_full.py                  第一阶段主入口
run_stage2.py                第二阶段单实验一键入口

analysis/                    第二阶段脚本与公共模块
analysis/common/             stage2 的路径、IO、分析、表格、绘图公共逻辑
analysis/run_shared_prep.py  共享预处理总入口
analysis/verify_shared_prep.py   共享产物校验入口
analysis/run_stage2_pipeline.py  第二阶段单实验总控
analysis/run_stage2_batch.py     第二阶段批量入口

docs/                        文档
tests/                       小样本测试与链路测试
outputs/                     统一输出目录

data/raw/                    原始专利数据
stopword/                    停用词
user_dict/                   用户词典
```

---

## 4. 第一阶段：专利指标计算

### 4.1 第一阶段做了什么

第一阶段核心入口是 `patent_quality.pipeline.run_all(cfg)`，流程分为 6 个顺序步骤：

1. 构建词表与分年 DF
2. 准备分年 tokens
3. 回顾性 TF-IDF 向量化
4. 向量剪枝
5. 计算 BS / FS
6. 组装最终结果

### 4.2 第一阶段输入

- 原始专利数据：`data/raw/...`
- 停用词：`stopword/`
- 用户词典：`user_dict/`

### 4.3 第一阶段输出

以 `outputs/experiments/<experiment_id>/stage1/` 为根目录，典型输出包括：

- `patent_quality_output.csv`
- `df/`
- `tokens/`
- `vectors/`
- `vectors_filtered/`
- `postings/`
- `pair_contrib/`
- `stats/`
- `logs/`

### 4.4 第一阶段运行方式

小样本 smoke test：

```bash
python tests/test_small.py
```

正式运行：

```bash
python run_full.py
```

### 4.5 exact 模式单专利验证脚本

如果你已经跑完 exact 实验，并且想检查某一篇专利：

- stage1 最终分词后到底哪些词参与了计算
- 每个词对前后窗口相似度累计贡献了多少
- 前 `k` 年 / 后 `k` 年里哪些专利和它最相似

可以使用：

```bash
python inspect_patent_similarity_case.py \
  --experiment-id 标题_摘要_ExactTime_window_1 \
  --application-no CN201110047803.9 \
  --year 2020 \
  --date 2020-01-03
```

推荐直接分析 exact 实验，也就是 `outputs/experiments/<experiment_id>/stage1_exact/`。

脚本输入：

- `--application-no`：申请号，必填
- `--year`：公开公告年份，必填
- `--date`：公开公告日，选填
- `--stage1-dir` 或 `--experiment-id`：二选一

默认行为：

- 不传 `--stage1-dir` 时，会根据 `--experiment-id` 自动定位 `stage1_exact`
- 不传 `--k` 时，会优先从 `pair_contrib/*.npz` 或 `pair_list.json` 推断窗口大小；如果推断失败，回退到 `Config.window_size` 默认值 `5`
- 不传 `--similarity-threshold` 时，会优先从 `pair_contrib/*.npz` 里的 `meta_json` 推断阈值；如果推断失败，回退到 `Config.similarity_threshold` 默认值 `0.05`
- 不传 `--output-dir` 时，会输出到
  `outputs/experiments/<experiment_id>/verification/patent_similarity_case/<case_name>/`
- `--top-n` 默认 100，`--bottom-n` 默认 10；当窗口内候选很多时，只保留相似度前 100 和最后 10 条

输出文件：

- `term_contribution.csv`
  目标专利每个词的 stage1 词频、是否进入最终向量、最终权重、向前/向后原始贡献、计入 BS/FS 的贡献
- `backward_similarity.csv`
  往前窗口内逐专利相似度，按相似度降序，只保留前 100 和后 10
- `forward_similarity.csv`
  往后窗口内逐专利相似度，按相似度降序，只保留前 100 和后 10
- `summary.json`
  目标专利信息、窗口参数、候选数量、贡献汇总、输出路径

关于 `--date`：

- 大多数时候只给 `申请号 + 公开公告年份` 就够了
- 如果 exact 数据里某个申请号在同一年对应了多条记录，脚本就无法知道你想看哪一条
- 这时再补 `--date YYYY-MM-DD`，就是告诉脚本“我要的是这个公开公告日对应的那条记录”
- `--title` 也是同样用途，只是用专利标题来辅助唯一定位

### 4.6 exact 实验批量排名查询脚本

如果你已经有一批专利申请号，想批量查询它们在：

- `outputs/experiments/标题_摘要_ExactTime_window_1/stage1_exact/`
- `outputs/experiments/标题_摘要_ExactTime_window_3/stage1_exact/`

中的年内排名、排名百分比和 `quantity_q`，使用：

- [search_exact_time_patents.py](search_exact_time_patents.py)

这个脚本默认查询两个 exact 实验，并输出一个新的 CSV。

输入支持两种形式：

1. 只给 `申请号`
   脚本会先去 `outputs/shared/raw_patent_authorized_parts/*.parquet` 中查这个申请号实际出现过的 `公开公告年份`，再按这些年份到两个实验里查询；如果同一个申请号对应多个公开年份，输出会展开成多行。

2. 给 `申请号 + 公开年份`
   脚本先按输入年份查；如果共享授权数据表明实际公开年份不同，也会继续按实际公开年份补查。

最常用命令：

```bash
python search_exact_time_patents.py \
  "outputs/第二十四届中国专利金奖.csv" \
  "outputs/第二十四届中国专利金奖_exact_time_lookup.csv" \
  --raw-lookup-mode auto
```

如果你只想依赖共享授权 parquet，不去原始 CSV 回查缺失原因：

```bash
python search_exact_time_patents.py \
  "outputs/第二十四届中国专利金奖.csv" \
  "outputs/第二十四届中国专利金奖_exact_time_lookup.csv" \
  --raw-lookup-mode skip
```

输出表会保留原始输入列，并补充：

- `查询公开年份`
- `<experiment_id>_状态`
- `<experiment_id>_命中公开年份`
- `<experiment_id>_排名`
- `<experiment_id>_年内专利数`
- `<experiment_id>_排名百分比`
- `<experiment_id>_quantity_q`
- `<experiment_id>_原因`

`--raw-lookup-mode` 含义：

- `skip`
  只查共享授权 parquet 与实验产物，不查原始 `data/raw/*.csv`
- `auto`
  优先用 `rg` 回查原始 CSV，补充“不是发明授权”“公开公告日无效”等原因
- `scan`
  逐行扫描原始 CSV，最慢但最彻底

---

## 5. 第二阶段：共享预处理层

共享预处理只处理与实验参数无关的静态底座，不读取任何某个实验的 `BS / FS / Quality_q`。

### 5.1 共享产物

#### patent_master

位置：
- `outputs/shared/patent_master/patent_master.parquet`
- `outputs/shared/patent_master/metadata.json`

来源脚本：
- [analysis/build_main_enriched.py](analysis/build_main_enriched.py)
  中的 `build_patent_master()`

作用：
- 扫描原始专利 CSV
- 按 `申请号` 去重
- 保留后续 stage2 需要的静态专利字段

#### special_firm_labels

位置：
- `outputs/shared/special_firm_labels/special_panel_clean.parquet`
- `outputs/shared/special_firm_labels/firm_year_special_labels.parquet`
- `outputs/shared/special_firm_labels/special_ucc_set.parquet`
- `outputs/shared/special_firm_labels/metadata.json`

来源脚本：
- [analysis/shared_prep.py](analysis/shared_prep.py)
  中的 `build_special_firm_labels()`

作用：
- 从特殊企业名单一次性生成静态标签和 `firm-year` 标签

#### ucc_mapping

位置：
- `outputs/shared/ucc_mapping/ucc_panel.csv`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`
- `outputs/shared/ucc_mapping/metadata.json`

来源脚本：
- [analysis/build_ucc_panel.py](analysis/build_ucc_panel.py)
  中的 `build_ucc_mapping()`

作用：
- 一次性生成上市公司及其子公司年度 UCC 面板
- 同时生成 explode 后的 `Stkid-Year-UCC` 明细表

#### financial_panel

位置：
- `outputs/shared/financial_panel/financial_annual_clean.parquet`
- `outputs/shared/financial_panel/metadata.json`

来源脚本：
- [analysis/shared_prep.py](analysis/shared_prep.py)
  中的 `build_financial_annual_panel()`

作用：
- 一次性清洗财务面板
- 保留年报口径公司年数据

### 5.2 共享预处理运行方式

如果使用仓库默认数据位置，可以直接运行：

```bash
python analysis/run_shared_prep.py
```

如需覆盖默认路径，再显式传参：

```bash
python analysis/run_shared_prep.py \
  --raw-patent-dir data/raw/中国专利分年份保存数据1985-2025 \
  --special-list-path analysis/graph/科创企业名单2024.dta \
  --financial-data-path analysis/公司财务/数据/上市公司财务数据/上市公司财务数据.dta \
  --listedco-parent-path analysis/公司财务/数据/上市公司基本信息年度表/上市公司统一社会信用代码.csv \
  --subsidiary-mapping-path analysis/公司财务/数据/爱企查结果/上市公司子公司对应统一社会信用代码.csv \
  --subjoint-csv-path analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint_merged.csv
```

验证共享产物：

```bash
python analysis/verify_shared_prep.py
```

---

## 6. 第二阶段：每实验分析层

当前 stage2 是**严格模式**：

- 必须提供 `stage1/patent_quality_output.csv`
- 必须预先存在 `outputs/shared/*`
- stage2 内部不再扫描原始专利目录
- stage2 内部不再清洗原始财务数据
- stage2 内部不再重建 UCC 面板
- stage2 内部不再重建特殊企业标签底座

### 6.1 stage2 总流程

[analysis/run_stage2_pipeline.py](analysis/run_stage2_pipeline.py) 当前按 6 步执行：

1. `diagnostics`
2. `build_experiment_patent_panel`
3. `analyze_quality_basic`
4. `analyze_special_firms`
5. `build_firm_year_innovation`
6. `run_regressions`

### 6.2 每一步对应脚本

#### diagnostics

相关脚本：
- `analysis/common/diagnostics.py`

输入：
- `stage1/df/`
- `stage1/vectors/`
- `stage1/vocab/`

输出：
- `outputs/experiments/<experiment_id>/stage2/diagnostics/*.csv`

#### build_experiment_patent_panel

相关脚本：
- [analysis/build_main_enriched.py](analysis/build_main_enriched.py)

输入：
- `stage1/patent_quality_output.csv`
- `outputs/shared/patent_master/patent_master.parquet`

输出：
- `stage2/data/patent_quality_output.csv`
- `stage2/data/main.parquet`
- `stage2/data/experiment_patent_panel.parquet`

#### analyze_quality_basic

相关脚本：
- [analysis/analyze_quality_basic.py](analysis/analyze_quality_basic.py)

输入：
- `stage2/data/experiment_patent_panel.parquet`

输出：
- `stage2/figures/fig_quality_*.png`
- `stage2/tables/tbl_desc_patent_quality.*`
- `stage2/tables/tbl_quality_citation_ols.*`

#### analyze_special_firms

相关脚本：
- [analysis/analyze_special_firms.py](analysis/analyze_special_firms.py)

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/special_firm_labels/firm_year_special_labels.parquet`
- `outputs/shared/special_firm_labels/special_ucc_set.parquet`

输出：
- `stage2/data/company_special_panel.parquet`
- `stage2/data/patents_special_year.parquet`
- `stage2/data/company_year_special.parquet`
- `stage2/data/company_year_abc.parquet`
- `stage2/tables/tbl_firm_compare.*`
- `stage2/tables/tbl_firmyear_compare.*`
- `stage2/figures/fig_special_*.png`
- `stage2/figures/fig_abc_*.png`

#### build_firm_year_innovation

相关脚本：
- [analysis/build_firm_year_innovation.py](analysis/build_firm_year_innovation.py)

输入：
- `stage2/data/experiment_patent_panel.parquet`
- `outputs/shared/ucc_mapping/ucc_exploded.parquet`

输出：
- `stage2/data/firm_year_innovation.parquet`

#### run_regressions

相关脚本：
- [analysis/run_regressions.py](analysis/run_regressions.py)

输入：
- `stage2/data/firm_year_innovation.parquet`
- `outputs/shared/financial_panel/financial_annual_clean.parquet`

输出：
- `stage2/data/regression_panel.parquet`
- `stage2/tables/tbl_regression_summary.*`
- `stage2/tables/reg_*.txt`
- `stage2/figures/fig_regression_coefficients.png`

### 6.3 stage2 运行方式

单实验命令行入口：

```bash
python analysis/run_stage2_pipeline.py \
  --experiment-id 标题_摘要_window5 \
  --stage1-dir outputs/experiments/标题_摘要_window5/stage1 \
  --shared-root outputs/shared
```

单实验一键入口：

```bash
python run_stage2.py
```

批量运行：

```bash
python analysis/run_stage2_batch.py --manifest path/to/stage2_manifest.yaml
```

---

## 7. 推荐运行顺序

```text
run_full.py
  -> outputs/experiments/<experiment_id>/stage1/

run_shared_prep.py
  -> outputs/shared/

verify_shared_prep.py
  -> 检查 shared 产物完整性

run_stage2_pipeline.py
  -> outputs/experiments/<experiment_id>/stage2/
```

如果只做单实验，最常用顺序是：

1. `python run_full.py`
2. `python analysis/run_shared_prep.py`
3. `python analysis/verify_shared_prep.py`
4. `python analysis/run_stage2_pipeline.py --experiment-id ... --stage1-dir ...`

---

## 8. 你应该先看哪些文件

建议按下面顺序阅读：

1. [run_full.py](run_full.py)
2. [analysis/run_shared_prep.py](analysis/run_shared_prep.py)
3. [analysis/verify_shared_prep.py](analysis/verify_shared_prep.py)
4. [run_stage2.py](run_stage2.py)
5. [analysis/run_stage2_pipeline.py](analysis/run_stage2_pipeline.py)
6. [analysis/run_stage2_batch.py](analysis/run_stage2_batch.py)
7. [patent_quality/pipeline.py](patent_quality/pipeline.py)
8. [analysis/common/config.py](analysis/common/config.py)
9. [docs/STAGE2_REFACTOR_PLAN.md](docs/STAGE2_REFACTOR_PLAN.md)
