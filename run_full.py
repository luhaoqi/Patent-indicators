from patent_quality.config import Config
from patent_quality.pipeline import run_all
import os
from patent_quality.project_paths import build_experiment_layout, get_project_root


def main():
    experiment_id = "标题_摘要_window3"
    layout = build_experiment_layout(experiment_id)
    # --- 正式运行配置 ---
    cfg = Config(
        # [关键] 数据路径：可指向单个CSV或包含多个CSV的文件夹
        # 默认建议将原始数据放在 data/raw 目录
        data_path="data/raw/中国专利分年份保存数据1985-2025",
        # 停用词表目录
        stopword_paths=["stopword"],
        # 用户自定义词典
        user_dict_path="user_dict/merged_96.txt",
        # [特征选择]
        min_term_count=50,  # 最小词频：小于20次的词被丢弃 (大规模数据建议20-50)
        max_doc_freq_ratio=0.5,  # 最大文档频率：超过50%文档出现的词被丢弃(太通用的词)
        # [算法参数]
        window_size=3,  # 滑动窗口大小：3年 (Kelly标准)
        similarity_threshold=0.05,  # 相似度阈值：0.05
        # [工程参数]
        artifacts_dir=os.fspath(layout.stage1_dir),  # 结果输出目录
        chunksize=100000,  # 批处理大小：10万行/次 (根据内存调整)
        log_level="INFO",
        log_file=os.fspath(layout.stage1_log_path(f"{experiment_id}.log")),  # 日志文件路径
        skip_if_exists=True,  # 断点续跑开关：True=跳过已完成阶段，False=强制重跑
        # [列名映射] (如果您的真实数据列名不同，请在此修改)
        col_id="申请号",
        col_date="申请年份",
        col_type="专利类型",
        col_text_parts=["专利名称", "摘要文本"],
        extra_cols=["申请人", "申请人类型", "申请人地址", "申请人城市"],  # 额外保留的列
    )

    print("=" * 50)
    print("开始执行全量任务")
    print(f"项目根目录: {get_project_root()}")
    print(f"数据路径: {(get_project_root() / cfg.data_path).resolve()}")
    print(f"实验目录: {layout.root}")
    print(f"输出目录: {layout.stage1_dir}")
    print(
        f"参数设置: window={cfg.window_size}, min_term={cfg.min_term_count}, thr={cfg.similarity_threshold}"
    )
    print("=" * 50)

    run_all(cfg)

    print("\n" + "=" * 50)
    print("全量任务执行完毕！")
    print(f"最终结果保存在: {cfg.final_output_path.as_posix()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
