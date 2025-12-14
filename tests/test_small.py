from patent_quality.config import Config
from patent_quality.pipeline import run_all
import os


def main():
    cfg = Config(
        # data_path=os.path.join("tests", "data_small.csv"),
        data_path=os.path.join("tests", "data"),
        stopword_paths=[os.path.join("stopword")],
        # user_dict_path=os.path.join("user_dict", "merged_96.txt"),
        min_term_count=1,
        max_doc_freq_ratio=0.9,
        window_size=2,
        similarity_threshold=0.0,
        artifacts_dir=os.path.join("artifacts_test"),
        chunksize=1000,
        log_file="artifacts_test/run.log",
        skip_if_exists=False,
    )
    run_all(cfg)
    out_csv = os.path.join(cfg.artifacts_dir, "patent_quality_output.csv")
    assert os.path.exists(out_csv)
    with open(out_csv, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        assert len(lines) >= 2
    print("ok")


if __name__ == "__main__":
    main()
