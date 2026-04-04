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
  text_sep=" ",
)
run_all(cfg)
```

3. 结果
- `outputs/experiments/<experiment_id>/stage1/patent_quality_output.csv`：包含 `申请号, 申请年份, 专利名称, BS, FS, Quality_q`
- 分年中间产物位于 `outputs/experiments/<experiment_id>/stage1/` 子目录

4. 断点续跑
- 自动记录 `outputs/experiments/<experiment_id>/stage1/checkpoint.json`，重复运行将复用已完成步骤

5. 常见问题
- Windows 并行分词：使用自建进程池方案或单线程分词
- 编码异常：设置 `Config.encoding` 为 `gb18030`

6. 个例分析
```bash
python inspect_patent_case.py \
  --experiment-id 标题_摘要_window3 \
  --application-no CN201010211885.1 \
  --application-year 2010
```

- 默认会优先从 `stage1/index`、`stage1/tokens`、`stage1/df`、`stage1/vocab` 读取中间结果，只在拿原文时读取对应年份的原始 CSV
- 现在会按 3 个阶段递进查找，并打印详细日志：
  1. 查 `stage1/index/year=XXXX.csv`
  2. 用户同意后查 `stage1/tokens/year=XXXX.jsonl`
  3. 用户再次同意后扩展到全部年份的 `stage1/index` 与 `stage1/tokens`
- 可用 `--yes-stage2`、`--yes-stage3` 跳过交互确认
- 支持 `--cases-manifest path/to/cases.yaml` 批量运行多个专利个例分析；同一进程内会复用 `vocab / global_df / year_df / stopwords / jieba` 缓存，避免每个 case 重复加载
- manifest 结构示例：
```yaml
shared:
  experiment_id: 标题_摘要_window3
  config_script: run_full.py
  yes_stage2: true
  yes_stage3: true
cases:
  - application_no: CN201010211885.1
    application_year: 2010
  - application_no: CN201010560351.X
    application_year: 2010
```
- 默认输出到 `outputs/experiments/<experiment_id>/verification/case_analysis/*.json`
- 如果分析的实验不是 `run_full.py` 里的主配置，建议显式传入对应的 `--config-script` 或相关参数覆盖
