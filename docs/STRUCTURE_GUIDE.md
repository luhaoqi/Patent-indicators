# 项目结构与数据放置建议

本文件用于说明推荐的目录组织方式，帮助你把原始数据、处理中间产物与研究输出分离，便于复现与清理。

## 推荐目录结构

```
Patent-indicators/
  data/
    raw/            # 原始数据（大文件，建议 git 忽略）
    processed/      # 清洗后或合并后的数据（可选）
  outputs/
    shared/
      patent_master/
      special_firm_labels/
      ucc_mapping/
      financial_panel/
      metadata/
      logs/
    experiments/
      <experiment_id>/
        stage1/
        stage2/
        verification/
    tests/
  analysis/             # 统计分析脚本与探索性笔记本
  docs/             # 文档
```

## 数据放置与命名

- 原始数据建议统一放在 data/raw/，可以是单个 CSV 或按年份分散的 CSV 文件夹。
- 合并/清洗后的数据建议放到 data/processed/，与原始数据分开，方便回滚。
- 运行输出统一放在 outputs/ 目录，作为可重复生成的产物，不建议纳入版本控制。
- 与实验无关的共享预处理产物统一放在 outputs/shared/。
- 与某个实验相关的 stage1 / stage2 / verification 结果统一放在 outputs/experiments/<experiment_id>/。

## Git 忽略建议

- 建议在 .gitignore 中忽略 data/ 与 outputs/ 下的大型生成产物，避免中间结果进入仓库。
- 若需要保留少量示例数据，建议放在 tests/data/ 并控制体量。

## 与配置的映射建议

- 运行脚本中 data_path 默认指向 data/raw/。
- stage1 的 artifacts_dir 建议指向 `outputs/experiments/<experiment_id>/stage1/`。
- shared prep 的输出根目录建议使用 `outputs/shared/`。
- stage2 的共享输入根目录建议使用 `outputs/shared/`，实验输出根目录建议使用 `outputs/experiments/<experiment_id>/stage2/`。
- 若要保持完全可复现，请把停用词与用户词典都放到项目内的 stopword/ 与 user_dict/。
