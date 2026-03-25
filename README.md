# 中国专利指标计算与统计分析项目说明

## 1. 项目目标

本项目当前包含两条连续但相对独立的工作流：

1. **第一阶段：复现 Kelly et al. 的专利质量指标**
   基于中文专利文本计算每篇专利的 `BS`、`FS` 和 `Quality_q = FS / (BS + epsilon)`。
2. **第二阶段：围绕第一阶段结果做统计分析、图表展示、公司层聚合与回归**
   重点包括指标诊断、年度趋势、与引证的关系、专精特新企业对比、公司年创新指数、财务回归等。

项目已经有较完整的实现与探索文档，但第二阶段目前主要分散在 `stat/` 下的 notebook 中，适合单次分析，不适合批量比较多组参数结果。

---

## 2. 当前仓库结构

```text
patent_quality/          第一阶段主流程代码
run_full.py              第一阶段全量运行入口
run_full_30years.py      第一阶段长时间窗口运行入口
docs/                    第一阶段说明文档与结构说明
stat/                    第二阶段诊断脚本、图表 notebook、公司财务分析
stat/graph/              主分析数据拼接、图表、专精特新企业分析
stat/公司财务/            公司统一社会信用代码整理、公司创新指数、回归
stopword/                停用词
user_dict/               用户词典
tests/                   小样本与 smoke test
artifacts_full*/         第一阶段已有产物示例
```

---

## 3. 第一阶段：专利指标计算流程

第一阶段核心入口是 `patent_quality.pipeline.run_all(cfg)`，当前代码已经实现为 6 个顺序阶段：

1. **构建词表与分年 DF**
   输入原始专利 CSV，筛选“发明授权”，分词后生成全局词表、分年 DF 和每年文档数。
2. **准备分年 tokens**
   将文本按年切分并持久化，减少后续重复分词。
3. **回顾性 TF-IDF 向量化**
   严格按年份递增，用历史累计 DF 计算当年 IDF，避免未来信息泄露。
4. **向量剪枝**
   去掉手工停用词、高 DF 词、每文档仅保留 TopK 权重项，减少后续计算量。
5. **计算 BS / FS**
   以滑动窗口方式，对年份对构建倒排索引并累计相似度贡献，输出每年每篇专利的 BS / FS。
6. **组装最终结果**
   汇总分年结果，生成 `patent_quality_output.csv`。

### 第一阶段主要输入

- 原始专利数据：`data/raw/...` 或外部按年份 CSV 目录
- 停用词：`stopword/`
- 用户词典：`user_dict/`

### 第一阶段主要输出

以某个 `artifacts_dir` 为根目录，典型产物包括：

- `df/global_df.json`
- `df/term_df_year=YYYY.json`
- `tokens/year=YYYY.jsonl`
- `vectors/year=YYYY.npz`
- `vectors_filtered/year=YYYY.npz`
- `postings/`
- `pair_contrib/`
- `stats/bsfs_year=YYYY.csv`
- `patent_quality_output.csv`

### 常用运行方式

- 小样本 smoke test：`python tests/test_small.py`
- 主配置全量运行：`python run_full.py`
- 长跨度运行：`python run_full_30years.py`

---

## 4. 第二阶段：当前统计分析流程

第二阶段目前可以分成三层。

### 4.1 中间产物诊断脚本

`stat/calc_*.py` 主要直接读取第一阶段的 `artifacts` 目录，用于检查词表、向量稀疏度和年份对 DF 重叠：

- `calc_avg_vocab_usage.py`
- `calc_df_pair_sum.py`
- `calc_topk_df_pair_sum.py`
- `calc_yearly_top_vocab.py`
- `calc_yearly_vocab_size.py`

这些脚本主要服务于“指标计算质量检查”和“剪枝参数理解”，输入是 `vectors/`、`df/`、`vocab/` 等中间结果。

### 4.2 构造主分析表 `main_enriched.parquet`

这一步主要由 [stat/graph/合并数据.ipynb](stat/graph/合并数据.ipynb) 完成，数据流如下：

1. 从第一阶段结果 `patent_quality_output.csv` 里抽取目标申请号；
2. 回到原始按年份专利 CSV 中，按申请号回捞额外字段；
3. 合并、去重、冲突检查，得到补充信息表；
4. 将第一阶段结果与补充字段左连接，输出 `stat/graph/main_enriched.parquet`。

这张表是第二阶段大部分图表分析和公司层分析的主输入。

### 4.3 图表、公司层分析与回归

当前主要 notebook 及作用如下：

- [stat/graph/graph.ipynb](stat/graph/graph.ipynb)
  主要做 `Quality_q` 与被引证次数关系、分布、年度趋势、企业分组描述统计等。
- [stat/graph/graph_with_firmyear_special.ipynb](stat/graph/graph_with_firmyear_special.ipynb)
  在前述图表基础上进一步引入“专精特新 / 特殊企业”的静态与动态（firm-year）口径，对专利层、公司层、公司年层做对比，并导出 `firm_compare.tex`、`firmyear_compare.tex`。
- [stat/公司财务/上市公司子公司.ipynb](stat/公司财务/上市公司子公司.ipynb)
  生成“上市公司（包括所有子公司）各年度的统一社会信用代码列表.csv”，为公司层聚合准备映射关系。
- [stat/公司财务/公司创新指数计算.ipynb](stat/公司财务/公司创新指数计算.ipynb)
  用 `main_enriched.parquet` 与公司年度统一社会信用代码列表，构造 `firm_year_innovation.parquet`。
- [stat/公司财务/reg.ipynb](stat/公司财务/reg.ipynb)
  将公司年创新指标与财务面板数据连接，做固定效应回归。

---

## 5. 当前完整数据流转

```text
原始专利 CSV
  -> 第一阶段 pipeline
  -> artifacts_xxx/
  -> patent_quality_output.csv

patent_quality_output.csv
  + 原始专利额外字段回捞
  -> stat/graph/main_enriched.parquet

main_enriched.parquet
  -> 指标图表 / 专精特新企业对比
  -> 公司层聚合

上市公司母子公司统一社会信用代码面板
  + main_enriched.parquet
  -> stat/公司财务/firm_year_innovation.parquet

firm_year_innovation.parquet
  + 财务面板数据
  -> 回归结果、描述统计表、图表
```

第二阶段除了依赖第一阶段结果外，还依赖几类外部数据：

- 原始专利主表（用于补回额外字段）
- 专精特新 / 科创企业名单 `.dta`
- 上市公司及子公司统一社会信用代码相关表
- 上市公司财务面板 `.dta`

---

## 6. 当前项目的主要问题

第一阶段已经比较接近稳定 pipeline，但第二阶段仍有几个结构性问题：

1. 输出路径基本固定在 `stat/graph/` 或 `stat/公司财务/` 下，容易互相覆盖。
2. notebook 中既有函数，也有一次性实验代码，复用成本高。
3. 图片大多直接 `show()`，未系统化保存，命名也不统一。
4. 第二阶段默认绑定某一组第一阶段结果，不方便批量比较多个参数版本。
5. 不同 notebook 共享相同逻辑（过滤口径、列名、分组统计），但尚未抽成公共模块。

---

## 7. 后续改造方向

后续建议把第二阶段改造成“**给定某一组第一阶段结果目录，就能自动输出该组参数对应的所有第二阶段结果**”的流程。核心原则如下：

1. **阶段一与阶段二解耦**
   第二阶段只接收某个实验目录下的 `patent_quality_output.csv` 或 `artifacts_dir`。
2. **每组参数单独输出**
   不再把结果直接写回 `stat/` 根目录，而是写到独立实验目录。
3. **notebook 函数脚本化**
   优先把已稳定的 notebook 迁移为 `.py` 脚本，notebook 保留作探索入口。
4. **图片、表格、日志统一命名**
   保证不同参数实验之间能直接对比。
5. **增加批处理入口**
   允许给定多组实验配置，顺序跑完整的第二阶段分析。

详细改造方案见 [docs/STAT_改造说明.md](docs/STAT_改造说明.md)。

---

## 8. 建议的阅读顺序

如果是第一次接手这个项目，建议按下面顺序理解：

1. [docs/README.md](docs/README.md)
2. [docs/USAGE.md](docs/USAGE.md)
3. [stat/README.md](stat/README.md)
4. [stat/graph/主要数据表介绍.md](stat/graph/主要数据表介绍.md)
5. [stat/graph/中期图像说明.md](stat/graph/中期图像说明.md)
6. [docs/STAT_改造说明.md](docs/STAT_改造说明.md)
