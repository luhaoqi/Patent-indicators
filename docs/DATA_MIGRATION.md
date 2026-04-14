# 原始数据分布与迁移方案

本文档记录实验复现所需的全部**原始数据**（非中间产物）的当前位置、用途、代码引用点，以及如果要将所有原始数据统一迁移到 `data/raw/` 下需要修改的文件。

> 说明："原始数据"指的是无法通过运行项目代码再生的外部输入。stage1/stage2 的中间产物和输出（`outputs/`、`artifacts_*`、`data/result/`）不在此范围内。

---

## 1. 原始数据清单

### 1.1 专利数据

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `data/raw/中国专利分年份保存数据1985-2025/` | 41 个按年份分割的 CSV（1985–2025），约 92 GB。包含字段：申请号、申请年份、申请人、申请人地址、专利名称、摘要文本、主权项内容、专利类型、申请人城市、申请人类型 | `run_full.py` 中 `Config.data_path`；`analysis/run_shared_prep.py` 中 `DEFAULT_RAW_PATENT_DIR` |

### 1.2 停用词表

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `stopword/专利停用词.txt` | 专利领域自定义停用词 | `Config.manual_stopwords_path`（默认值 `./stopword/专利停用词.txt`） |
| `stopword/中文停用词表.txt` | 通用中文停用词 | `run_full.py` 中 `Config.stopword_paths=["stopword"]`，整个目录被扫描 |
| `stopword/哈工大停用词表.txt` | 哈尔滨工业大学停用词表 | 同上 |
| `stopword/四川大学机器智能实验室停用词库.txt` | 四川大学停用词表 | 同上 |
| `stopword/百度停用词表.txt` | 百度停用词表 | 同上 |

### 1.3 用户词典

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `user_dict/merged_96.txt` | jieba 用户自定义词典（96 条自定义词） | `run_full.py` 中 `Config.user_dict_path` |
| `user_dict/merged.txt` | 更大的合并词典（备用） | 未被当前主流程引用 |

### 1.4 特殊企业名单

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/graph/科创企业名单2024.dta` | 科创/专精特新企业名单（Stata 格式） | `analysis/run_shared_prep.py` 中 `DEFAULT_SPECIAL_LIST_PATH` |

### 1.5 上市公司财务数据

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/公司财务/数据/上市公司财务数据/上市公司财务数据.dta` | 上市公司财务年报数据 1990–2023（Stata 格式） | `analysis/run_shared_prep.py` 中 `DEFAULT_FINANCIAL_DATA_PATH` |

### 1.6 上市公司基本信息

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/公司财务/数据/上市公司基本信息年度表/上市公司统一社会信用代码.csv` | 上市公司母公司统一社会信用代码 | `analysis/run_shared_prep.py` 中 `DEFAULT_LISTEDCO_PARENT_PATH` |
| `analysis/公司财务/数据/上市公司基本信息年度表/STK_LISTEDCOINFOANL.csv` | CSMAR 上市公司基本信息年度表（原始下载） | 上述 CSV 的来源，不直接被代码引用 |

### 1.7 子公司联营合营数据

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint_merged.csv` | 上市公司子公司联营合营明细（合并后） | `analysis/run_shared_prep.py` 中 `DEFAULT_SUBJOINT_CSV_PATH` |
| `analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint.csv` | CSMAR 原始下载 | 合并前的原始数据 |
| `analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint1.csv` | CSMAR 原始下载（补充） | 合并前的原始数据 |

### 1.8 子公司统一社会信用代码映射

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/公司财务/数据/爱企查结果/上市公司子公司对应统一社会信用代码.csv` | 子公司名称 → 统一社会信用代码映射表（从爱企查结果合并而来） | `analysis/run_shared_prep.py` 中 `DEFAULT_SUBSIDIARY_MAPPING_PATH` |

### 1.9 爱企查批量查询原始数据

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/公司财务/数据/aiqicha_query_files_手动/aiqicha_batch_01~34.xlsx` | 34 个手动上传到爱企查的批量查询输入文件 | 不被代码直接引用；是生成爱企查结果的输入 |
| `analysis/公司财务/数据/爱企查结果/批量查询数据导出（企业信息）-【爱企查】-*.xls/xlsx` | 约 44 个爱企查导出文件 | 不被代码直接引用；合并后产出 `爱企查结果_merged.csv` |
| `analysis/公司财务/数据/爱企查结果/爱企查结果_merged.csv` | 爱企查结果合并表 | 是 `上市公司子公司对应统一社会信用代码.csv` 的中间产物 |

### 1.10 子公司退出明细

| 当前位置 | 说明 | 代码引用 |
|---------|------|---------|
| `analysis/公司财务/数据/子公司退出明细表/STK_NotesInvExit.csv` | CSMAR 子公司退出明细表 | 不被当前主流程直接引用，但在 UCC 面板构建的早期版本中曾使用 |

---

## 2. 哪些是复现必需的原始数据

复现实验**最小必需**的原始数据如下（代码会直接读取的文件）：

| 编号 | 文件 | 被谁使用 |
|------|------|---------|
| ① | `data/raw/中国专利分年份保存数据1985-2025/*.csv` | stage1 (`run_full.py`) + shared prep |
| ② | `stopword/` 目录下全部 5 个 txt | stage1 |
| ③ | `user_dict/merged_96.txt` | stage1 |
| ④ | `analysis/graph/科创企业名单2024.dta` | shared prep (`build_special_firm_labels`) |
| ⑤ | `analysis/公司财务/数据/上市公司财务数据/上市公司财务数据.dta` | shared prep (`build_financial_annual_panel`) |
| ⑥ | `analysis/公司财务/数据/上市公司基本信息年度表/上市公司统一社会信用代码.csv` | shared prep (`build_ucc_mapping`) |
| ⑦ | `analysis/公司财务/数据/上市公司子公司联营合营情况表/STK_NotesSubJoint_merged.csv` | shared prep (`build_ucc_mapping`) |
| ⑧ | `analysis/公司财务/数据/爱企查结果/上市公司子公司对应统一社会信用代码.csv` | shared prep (`build_ucc_mapping`) |

爱企查批量查询原始文件（1.9 节）和子公司退出明细（1.10 节）是上游数据准备过程的输入，当前代码不直接读取，但如果需要从头重建 `上市公司子公司对应统一社会信用代码.csv` 则需要它们。

---

## 3. 建议的统一目录结构

如果要将所有原始数据迁移到 `data/raw/` 下统一管理：

```
data/raw/
├── 中国专利分年份保存数据1985-2025/     # ① 已在此处，无需移动
│   ├── 中国专利数据库1985年.csv
│   ├── ...
│   └── 中国专利数据库2025年.csv
├── stopword/                             # ② 从 stopword/ 移入
│   ├── 专利停用词.txt
│   ├── 中文停用词表.txt
│   ├── 哈工大停用词表.txt
│   ├── 四川大学机器智能实验室停用词库.txt
│   └── 百度停用词表.txt
├── user_dict/                            # ③ 从 user_dict/ 移入
│   └── merged_96.txt
├── 科创企业名单2024.dta                   # ④ 从 analysis/graph/ 移入
├── 上市公司财务数据/                       # ⑤ 从 analysis/公司财务/数据/ 移入
│   └── 上市公司财务数据.dta
├── 上市公司基本信息年度表/                  # ⑥ 从 analysis/公司财务/数据/ 移入
│   ├── 上市公司统一社会信用代码.csv
│   └── STK_LISTEDCOINFOANL.csv
├── 上市公司子公司联营合营情况表/             # ⑦ 从 analysis/公司财务/数据/ 移入
│   ├── STK_NotesSubJoint.csv
│   ├── STK_NotesSubJoint1.csv
│   └── STK_NotesSubJoint_merged.csv
├── 爱企查结果/                            # ⑧ 从 analysis/公司财务/数据/ 移入
│   ├── 上市公司子公司对应统一社会信用代码.csv
│   ├── 爱企查结果_merged.csv
│   └── 批量查询数据导出（企业信息）-【爱企查】-*.xls/xlsx
├── aiqicha_query_files/                   # 从 analysis/公司财务/数据/aiqicha_query_files_手动/ 移入
│   └── aiqicha_batch_01~34.xlsx
└── 子公司退出明细表/                       # 从 analysis/公司财务/数据/ 移入
    └── STK_NotesInvExit.csv
```

---

## 4. 迁移后需要修改的文件

### 4.1 代码文件

| 文件 | 需要修改的内容 |
|------|--------------|
| `run_full.py` | `Config.data_path`、`Config.stopword_paths`、`Config.user_dict_path` 的默认值 |
| `patent_quality/config.py` | `manual_stopwords_path` 的默认值 `"./stopword/专利停用词.txt"` → `"data/raw/stopword/专利停用词.txt"` |
| `analysis/run_shared_prep.py` | 6 个 `DEFAULT_*` 常量的路径：`DEFAULT_RAW_PATENT_DIR`、`DEFAULT_SPECIAL_LIST_PATH`、`DEFAULT_FINANCIAL_DATA_PATH`、`DEFAULT_LISTEDCO_PARENT_PATH`、`DEFAULT_SUBSIDIARY_MAPPING_PATH`、`DEFAULT_SUBJOINT_CSV_PATH` |

### 4.2 文档文件

| 文件 | 需要修改的内容 |
|------|--------------|
| `README.md` | §4.2 "第一阶段输入" 中引用的路径，§5.2 "共享预处理运行方式" 中的示例命令 |
| `docs/STRUCTURE_GUIDE.md` | 推荐目录结构说明 |
| `docs/USAGE.md` | 示例命令中的路径 |

### 4.3 配置/忽略文件

| 文件 | 需要修改的内容 |
|------|--------------|
| `.gitignore` | 已包含 `data/`，无需修改。但需确认移除根目录下的 `stopword/` 和 `user_dict/` 相关规则（当前 `user_dict/` 已在 `.gitignore` 中） |

---

## 5. 可以安全删除的旧产物目录

以下目录是中间产物或旧版本输出，不是原始数据，可在确认 `outputs/` 下有最新结果后删除：

| 目录 | 说明 |
|------|------|
| `data/result/` | stage1 旧版本输出（window_size=3 和 window_size=5），已被 `outputs/experiments/` 下的结果取代 |
| `artifacts_full/` | 早期全量运行产物，已被 `outputs/experiments/` 取代 |
| `artifacts_full_30years/` | 早期 30 年运行产物 |
| `artifacts_test/` | 测试运行产物 |
| `analysis/graph/extra_parts/` | 旧版按年 parquet 文件，已被 `outputs/shared/patent_master/` 取代 |
| `analysis/graph/main.parquet`、`main_enriched.parquet`、`企业.parquet` 等 | 旧版 stage2 中间产物，已被 `outputs/experiments/*/stage2/data/` 取代 |
| `analysis/公司财务/firm_year_innovation.parquet` | 旧版公司年创新指数，已被 `outputs/experiments/*/stage2/data/firm_year_innovation.parquet` 取代 |
| `analysis/公司财务/数据/outs/` | 临时导出文件 |
