# 仓库指南

## 项目结构与模块组织
`patent_quality/` 是主流程代码目录：`config.py` 管理配置，`pipeline.py` 负责串联阶段，`nlp.py`、`vocab.py`、`vectorizer.py` 处理分词与向量化，`similarity.py`、`pruning.py`、`quality.py` 负责 BS/FS 计算与结果组装。`run_full.py` 和 `run_full_30years.py` 是全量运行入口。探索性分析放在 `analysis/`，文档放在 `docs/`，小样本回归数据放在 `tests/` 与 `tests/data/`。可重生产物统一放在 `outputs/experiments/<experiment_id>/stage1|stage2|verification/`，测试产物放在 `outputs/tests/`，不应视为源码。

## 构建、测试与开发命令
项目使用 Python 3，依赖见 `docs/README.md`，核心包括 `pandas`、`jieba`、`numpy`、`scipy`、`scikit-learn`、`tqdm`。

- `python tests/test_small.py`：用仓库内小样本跑通整条流水线，并检查 `outputs/tests/test_small_smoke/stage1/patent_quality_output.csv` 是否生成。
- `python run_full.py`：按主配置处理 `data/raw/...` 中的数据，并把结果写入 `outputs/experiments/title_abstract_window3/stage1/`。
- `python run_full_30years.py`：运行更长时间跨度的全量任务。
- `python verify_ir.py`：抽样验证 IR / 相似度计算结果。
- `python profile_matrix.py`：检查稀疏矩阵规模与分布。

## 代码风格与命名约定
遵循现有 Python 风格：4 空格缩进，函数和变量使用 `snake_case`，类与数据类使用 `PascalCase`，例如 `Config`。模块命名尽量短，并按流水线阶段归类。优先通过配置项控制路径、列名和参数，不要把环境相关路径硬编码进逻辑。注释保持简短，只解释不直观的处理流程。仓库中目前没有统一的格式化或 lint 配置，提交前应主动保持与周边代码风格一致。

## 测试规范
新增测试放在 `tests/` 下，命名采用 `test_<feature>.py`。优先使用 `tests/data/` 中的小型 CSV 夹具，保证测试可快速、稳定运行。修改流水线逻辑时，至少验证小样本 smoke test，并确认相关中间产物、断点续跑或输出 CSV 行为没有回归。日常开发不要依赖完整原始数据集做验证。

## 提交与合并请求规范
现有提交历史主要使用简短前缀，如 `feat:`、`fix:`，后接简洁中文说明，例如 `feat: 调整向量剪枝参数`。继续沿用这一模式。提交 PR 时应写清影响了哪个流水线阶段、修改了哪些配置或数据假设、实际执行了哪些验证命令，以及是否改变了输出目录或结果文件格式。只有在 notebook、报告或图表需要人工比对时，再附截图或示例产物路径。

## 数据与配置说明
原始数据建议放在 `data/raw/`，停用词和用户词典分别放在 `stopword/` 与 `user_dict/`。可复现输出统一写入 `outputs/experiments/<experiment_id>/` 或 `outputs/tests/`。除测试夹具外，不要提交大型原始数据或整批重新生成的中间产物。
