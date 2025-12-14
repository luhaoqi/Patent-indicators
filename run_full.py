from patent_quality.config import Config
from patent_quality.pipeline import run_all
import os

def main():
    # --- 正式运行配置 ---
    cfg = Config(
        # [关键] 数据路径：请修改为您真实的96GB数据路径
        # 可以是单个大CSV文件，也可以是包含多个CSV的文件夹
        # 例如: r"D:\Data\patent_96gb.csv" 或 r"D:\Data\all_csvs"
        # 目前默认指向 tests/data 目录作为演示
        data_path=os.path.join("tests", "data"), 
        
        # 停用词表目录
        stopword_paths=[os.path.join("stopword")],
        
        # 用户自定义词典
        user_dict_path=os.path.join("user_dict", "merged_96.txt"),
        
        # [特征选择]
        min_term_count=20,          # 最小词频：小于20次的词被丢弃 (大规模数据建议20-50)
        max_doc_freq_ratio=0.5,     # 最大文档频率：超过50%文档出现的词被丢弃(太通用的词)
        
        # [算法参数]
        window_size=5,              # 滑动窗口大小：5年 (Kelly标准)
        similarity_threshold=0.05,   # 相似度阈值：0.05
        
        # [工程参数]
        artifacts_dir="artifacts_full_30years", # 结果输出目录 (与测试目录区分开)
        chunksize=100000,           # 批处理大小：10万行/次 (根据内存调整)
        log_level="INFO",
        log_file="artifacts_full_30years/run.log", # 日志文件路径
        skip_if_exists=True,       # 断点续跑开关：True=跳过已完成阶段，False=强制重跑
        
        # [列名映射] (如果您的真实数据列名不同，请在此修改)
        col_id="申请号",
        col_date="申请年份",
        col_type="专利类型",
        col_text_parts=["专利名称", "摘要文本", "主权项内容"],
        extra_cols=["申请人", "申请人类型", "申请人地址", "申请人城市"] # 额外保留的列
    )
    
    print("="*50)
    print(f"开始执行全量任务")
    print(f"数据路径: {os.path.abspath(cfg.data_path)}")
    print(f"输出目录: {os.path.abspath(cfg.artifacts_dir)}")
    print(f"参数设置: window={cfg.window_size}, min_term={cfg.min_term_count}, thr={cfg.similarity_threshold}")
    print("="*50)
    
    run_all(cfg)
    
    print("\n" + "="*50)
    print("全量任务执行完毕！")
    print(f"最终结果保存在: {os.path.join(cfg.artifacts_dir, 'patent_quality_output.csv')}")
    print("="*50)

if __name__ == "__main__":
    main()
