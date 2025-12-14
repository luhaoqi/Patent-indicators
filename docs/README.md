# 中国专利质量指标复现（Kelly et al., 2018/2021）

目标：基于中文专利文本复现 `q_j = FS_j / BS_j` 指标，严格遵守回顾性 IDF 与 5 年滑窗。

特性：
- 流式/分块处理，避免越过内存红线
- 中文分词与停用词并集，专利领域停用词内置
- 稀疏矩阵与块乘法计算相似度
- 断点续跑与分年持久化
- 进度可视化与日志

依赖：Python、Pandas、Jieba、NumPy、SciPy、scikit‑learn、tqdm

目录：
- `patent_quality/` 模块代码
- `artifacts/` 中间产物
- `tests/` 小样本数据与脚本
- `docs/` 文档

参考：
- NBER w25266（Kelly 等）与 AEA Insights 论文
- 稀疏余弦相似度文档与实践
