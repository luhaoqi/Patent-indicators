# 使用说明

1. 准备数据与配置
- 将原始 CSV 放入目录或提供单文件路径
- 准备停用词文件与用户词典（可选）

2. 运行
```python
from patent_quality.config import Config
from patent_quality.pipeline import run_all
cfg = Config(
  data_path="path/to/csv_or_dir",
  stopword_paths=["stopwords.txt"],
  user_dict_path="user_dict.txt",
)
run_all(cfg)
```

3. 结果
- `artifacts/patent_quality_output.csv`：包含 `申请号, 申请年份, 专利名称, BS, FS, Quality_q`
- 分年中间产物位于 `artifacts/` 子目录

4. 断点续跑
- 自动记录 `artifacts/checkpoint.json`，重复运行将复用已完成步骤

5. 常见问题
- Windows 并行分词：使用自建进程池方案或单线程分词
- 编码异常：设置 `Config.encoding` 为 `gb18030`
