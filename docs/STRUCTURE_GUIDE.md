# 项目结构与数据放置建议

本文件用于说明推荐的目录组织方式，帮助你把原始数据、处理中间产物与研究输出分离，便于复现与清理。

## 推荐目录结构

```
Patent-indicators/
  data/
    raw/            # 原始数据（大文件，建议 git 忽略）
    processed/      # 清洗后或合并后的数据（可选）
  artifacts/        # 默认中间产物输出目录（可被重建）
  artifacts_full/   # 全量运行产物（可被重建）
  artifacts_full_30years/
  outputs/          # 对外发布结果或汇总表（可选）
  stat/             # 统计分析脚本与探索性笔记本
  docs/             # 文档
```

## 数据放置与命名

- 原始数据建议统一放在 data/raw/，可以是单个 CSV 或按年份分散的 CSV 文件夹。
- 合并/清洗后的数据建议放到 data/processed/，与原始数据分开，方便回滚。
- 运行输出统一放在 artifacts* 目录，作为可重复生成的产物，不建议纳入版本控制。

## Git 忽略建议

- 建议在 .gitignore 中忽略 data/ 与 artifacts*/，避免大文件与中间结果进入仓库。
- 若需要保留少量示例数据，建议放在 tests/data/ 并控制体量。

## 与配置的映射建议

- 运行脚本中 data_path 默认指向 data/raw/。
- artifacts_dir 建议指向 artifacts_full 或 artifacts_full_30years，区分不同实验规模。
- 若要保持完全可复现，请把停用词与用户词典都放到项目内的 stopword/ 与 user_dict/。
