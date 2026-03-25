from patent_quality.config import Config
from patent_quality.pipeline import run_all
import os
from patent_quality.project_paths import build_experiment_layout, get_project_root


def main():
    experiment_id = "test_small_smoke"
    layout = build_experiment_layout(experiment_id, output_root="outputs/tests")
    cfg = Config(
        data_path="tests/data",
        stopword_paths=["stopword"],
        min_term_count=1,
        max_doc_freq_ratio=0.9,
        window_size=2,                     # 特意改小了
        similarity_threshold=0.05,
        artifacts_dir=os.fspath(layout.stage1_dir),
        chunksize=1000,
        log_file=os.fspath(layout.stage1_log_path("test_small.log")),
        skip_if_exists=True,
    )
    run_all(cfg)
    out_csv = cfg.final_output_path
    assert out_csv.exists()
    with open(out_csv, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        assert len(lines) >= 2
    print(f"project_root={get_project_root()}")
    print(f"test_output={out_csv}")
    print("ok")


if __name__ == "__main__":
    main()
