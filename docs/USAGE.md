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

7. exact 模式单专利相似度拆解

如果你要验证 exact 实验里某一篇专利“最后哪些词参与了计算、每个词贡献了多少、窗口内哪些专利与它最相似”，使用：

```bash
python inspect_patent_similarity_case.py \
  --experiment-id 标题_摘要_ExactTime_window_1 \
  --application-no CN201110047803.9 \
  --year 2020 \
  --date 2020-01-03
```

这个脚本默认面向 `stage1_exact`：

- `--application-no`：申请号，必填
- `--year`：公开公告年份，必填
- `--date`：公开公告日，选填；只有定位不唯一时才需要
- `--stage1-dir`：可直接传 `outputs/experiments/<experiment_id>/stage1_exact`
- `--experiment-id`：不传 `--stage1-dir` 时使用

不指定可选参数时：

- 不传 `--k`：优先从 `pair_contrib/*.npz` 或 `pair_list.json` 推断窗口大小；推断失败时回退到 `Config.window_size=5`
- 不传 `--similarity-threshold`：优先从 `pair_contrib/*.npz` 的 `meta_json` 推断阈值；推断失败时回退到 `Config.similarity_threshold=0.05`
- 不传 `--output-dir`：默认输出到
  `outputs/experiments/<experiment_id>/verification/patent_similarity_case/<case_name>/`
- 不传 `--top-n` / `--bottom-n`：默认保留相似度前 100 和最后 10 条

输出文件：

- `term_contribution.csv`
  目标专利每个词的词频、最终权重、是否参与最终计算、向前/向后贡献
- `backward_similarity.csv`
  往前 `k` 年或 `k` 年窗口内的候选专利相似度，降序保存
- `forward_similarity.csv`
  往后 `k` 年或 `k` 年窗口内的候选专利相似度，降序保存
- `summary.json`
  记录目标专利、窗口参数、候选数量、输出文件路径

`--date YYYY-MM-DD` 的意思：

- exact 模式是按“公开公告日”做窗口，不是只按年份
- 某些数据源里，`申请号 + 公开公告年份` 可能不足以唯一定位一条记录
- 这时加上 `--date`，就是明确告诉脚本你要分析哪一个公开公告日对应的那条专利
- 如果 `申请号 + 年份` 已经唯一，`--date` 可以不传

8. exact 实验批量排名查询

如果你想批量查询一批申请号在两个 exact 实验里的：

- 年内排名
- 年内排名百分比
- `quantity_q`

使用：

```bash
python search_exact_time_patents.py input.csv output.csv
```

默认查询的实验目录：

- `outputs/experiments/标题_摘要_ExactTime_window_1/stage1_exact/`
- `outputs/experiments/标题_摘要_ExactTime_window_3/stage1_exact/`

8.1 输入格式

最少只需要一列：

- `申请号`

也可以额外提供一列公开年份：

- `公开年份`
- 或 `公开公告年份`

如果只有 `申请号`：

- 脚本会先去 `outputs/shared/raw_patent_authorized_parts/*.parquet` 里查这个申请号出现过的所有 `公开公告年份`
- 再按这些年份去两个 exact 实验里查询
- 如果同一个申请号对应多个公开年份，输出会展开成多行

如果同时提供了 `申请号 + 公开年份`：

- 脚本先按输入年份查
- 如果共享授权数据表明实际公开年份不同，也会继续按实际年份补查

8.2 常用命令

只查共享授权 parquet 和实验产物，不查原始 CSV：

```bash
python search_exact_time_patents.py \
  "outputs/第二十四届中国专利金奖.csv" \
  "outputs/第二十四届中国专利金奖_exact_time_lookup.csv" \
  --raw-lookup-mode skip
```

同时尽量精确判断“不是发明授权”等缺失原因：

```bash
python search_exact_time_patents.py \
  "outputs/第二十四届中国专利金奖.csv" \
  "outputs/第二十四届中国专利金奖_exact_time_lookup.csv" \
  --raw-lookup-mode auto
```

如果列名不是标准名称，可以显式指定：

```bash
python search_exact_time_patents.py \
  input.csv \
  output.csv \
  --application-col 申请号 \
  --public-year-col 公开年份
```

8.3 输出列

输出表保留原始输入列，并追加：

- `查询公开年份`
- `标题_摘要_ExactTime_window_1_状态`
- `标题_摘要_ExactTime_window_1_命中公开年份`
- `标题_摘要_ExactTime_window_1_排名`
- `标题_摘要_ExactTime_window_1_年内专利数`
- `标题_摘要_ExactTime_window_1_排名百分比`
- `标题_摘要_ExactTime_window_1_quantity_q`
- `标题_摘要_ExactTime_window_1_原因`
- `标题_摘要_ExactTime_window_3_状态`
- `标题_摘要_ExactTime_window_3_命中公开年份`
- `标题_摘要_ExactTime_window_3_排名`
- `标题_摘要_ExactTime_window_3_年内专利数`
- `标题_摘要_ExactTime_window_3_排名百分比`
- `标题_摘要_ExactTime_window_3_quantity_q`
- `标题_摘要_ExactTime_window_3_原因`

字段解释：

- `查询公开年份`
  本行实际用于查询实验产物的年份
- `命中公开年份`
  该实验最终命中的年份；通常与 `查询公开年份` 相同
- `状态`
  `找到` 或 `未找到`
- `原因`
  没找到时说明原因；找到但发生年份兜底时，也会记录说明

8.4 `--raw-lookup-mode`

- `skip`
  不回查原始 `data/raw/*.csv`，速度更快；但无法精确区分“不是发明授权”与“原始数据不存在”
- `auto`
  优先使用 `rg` 回查原始 CSV；适合日常使用
- `scan`
  逐行扫描原始 CSV；最慢，但最彻底

8.5 日志

脚本默认输出 `INFO` 日志。可以看到：

- 输入 CSV 读取进度
- 共享授权 parquet 查询进度
- 原始 CSV 回查进度
- 每条输入对应的年份展开数量
- 每个实验每个年份的 `index/stats` 加载进度

如果需要更多日志：

```bash
python search_exact_time_patents.py input.csv output.csv --log-level DEBUG
```
