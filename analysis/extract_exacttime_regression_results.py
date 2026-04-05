from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs/experiments/标题_摘要_ExactTime_window_1_vs_3_regression_extract"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    window_label: str
    window_sort: int
    stage_dir: Path

    @property
    def summary_csv(self) -> Path:
        return self.stage_dir / "tables/回归分析/tbl_regression_summary.csv"

    @property
    def sample_csv(self) -> Path:
        return self.stage_dir / "tables/回归分析/tbl_regression_sample_summary.csv"


EXPERIMENTS = (
    ExperimentConfig(
        experiment_id="标题_摘要_ExactTime_window_1",
        window_label="window_1",
        window_sort=1,
        stage_dir=REPO_ROOT / "outputs/experiments/标题_摘要_ExactTime_window_1/stage2_exact",
    ),
    ExperimentConfig(
        experiment_id="标题_摘要_ExactTime_window_3",
        window_label="window_3",
        window_sort=3,
        stage_dir=REPO_ROOT / "outputs/experiments/标题_摘要_ExactTime_window_3/stage2_exact",
    ),
)

QUALITY_VAR_LABELS = {
    "mean_z_q_ft": "标准化专利质量均值",
    "highq_share_ft": "高质量专利占比",
    "log_highq_count_ft": "高质量专利件数对数",
    "mean_raw_q_w_ft": "winsorized 原始质量均值",
    "rd_intensity_asset": "研发强度（研发费用/资产）",
}

DEP_VAR_LABELS = {
    "roa": "ROA",
    "roe": "ROE",
    "ebit_asset": "EBIT / Asset",
    "ebitda_asset": "EBITDA / Asset",
    "profit_asset": "Profit / Asset",
    "profit_margin": "Profit Margin",
    "ebit_margin": "EBIT Margin",
    "ebitda_margin": "EBITDA Margin",
    "log_sales": "Log Sales",
    "log_asset": "Log Asset",
    "sales_growth": "Sales Growth",
    "gassets": "Asset Growth",
    "gfa": "Fixed Asset Growth",
}

OUTCOME_GROUP_LABELS = {
    "profitability": "盈利能力",
    "margin_operations": "利润率与经营表现",
    "growth": "成长性",
    "other": "其他",
}

MODEL_FAMILY_LABELS = {
    "main": "当期主回归",
    "future_main": "未来回归",
    "rd_same_sample": "RD 同样本基线",
    "rd_horse_race": "RD horse-race",
    "rd_only": "RD only",
}

BUCKET_LABELS = {
    "current_main": "当期主回归",
    "current_rd": "RD 对照回归",
    "future": "未来回归",
}

BUCKET_ORDER = {
    "current_main": 0,
    "current_rd": 1,
    "future": 2,
}

OUTCOME_GROUP_ORDER = {
    "profitability": 0,
    "margin_operations": 1,
    "growth": 2,
    "other": 3,
}

DEP_VAR_ORDER = {
    "roa": 0,
    "roe": 1,
    "ebit_asset": 2,
    "ebitda_asset": 3,
    "profit_asset": 4,
    "profit_margin": 5,
    "ebit_margin": 6,
    "ebitda_margin": 7,
    "log_sales": 8,
    "log_asset": 9,
    "sales_growth": 10,
    "gassets": 11,
    "gfa": 12,
}

QUALITY_VAR_ORDER = {
    "mean_z_q_ft": 0,
    "highq_share_ft": 1,
    "log_highq_count_ft": 2,
    "mean_raw_q_w_ft": 3,
    "rd_intensity_asset": 4,
}

PARAMETER_LABELS = {
    **QUALITY_VAR_LABELS,
    "Intercept": "截距",
    "ln_asset": "企业规模（ln_asset）",
    "lev_ratio": "资产负债率（lev_ratio）",
    "gassets": "总资产增长率（gassets）",
    "log_patent_count_ft": "专利数量对数",
}

TOKEN_REPLACEMENTS = {
    "EntityEffects": "firm FE",
    "TimeEffects": "year FE",
}

SPEC_VARIANT_ORDER = {
    "cnt1": 0,
    "cnt0": 1,
    "rdsame": 2,
    "rdhorse": 3,
    "rdonly": 4,
}

PARAMETER_LINE_RE = re.compile(
    r"^\s*(?P<name>\S+)\s+"
    r"(?P<coef>[-+0-9.eE]+)\s+"
    r"(?P<se>[-+0-9.eE]+)\s+"
    r"(?P<t>[-+0-9.eE]+)\s+"
    r"(?P<p>[-+0-9.eE]+)\s+"
    r"(?P<lower>[-+0-9.eE]+)\s+"
    r"(?P<upper>[-+0-9.eE]+)\s*$"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def detect_outcome_group(dep_var: str) -> str:
    if dep_var in {"roa", "roe", "ebit_asset", "ebitda_asset", "profit_asset"}:
        return "profitability"
    if dep_var in {"profit_margin", "ebit_margin", "ebitda_margin", "log_sales", "log_asset"}:
        return "margin_operations"
    if dep_var in {"sales_growth", "gassets", "gfa"}:
        return "growth"
    return "other"


def detect_family_bucket(model_family: str, future_horizon: int) -> str:
    if future_horizon > 0 or model_family == "future_main":
        return "future"
    if model_family == "main":
        return "current_main"
    return "current_rd"


def extract_spec_variant(spec_id: str, sample_threshold: int) -> str:
    match = re.search(rf"_pc{sample_threshold}_(.+)$", spec_id)
    return match.group(1) if match else ""


def build_display_dep(dep_var: str, future_horizon: int) -> str:
    if future_horizon > 0:
        return f"{dep_var}(t+{future_horizon})"
    return dep_var


def parse_formula(formula: str, dep_var: str, future_horizon: int) -> tuple[str, list[str], str]:
    if "~" not in formula:
        return formula, [], formula

    _, rhs_text = formula.split("~", 1)
    rhs_tokens = [token.strip() for token in rhs_text.split("+") if token.strip() and token.strip() != "1"]
    param_order = [token for token in rhs_tokens if token not in {"EntityEffects", "TimeEffects"}]
    readable_rhs = [TOKEN_REPLACEMENTS.get(token, token) for token in rhs_tokens]
    equation_readable = f"{build_display_dep(dep_var, future_horizon)} ~ {' + '.join(readable_rhs)}"
    source_equation = f"{formula.split('~', 1)[0].strip()} ~ {' + '.join(readable_rhs)}"
    return equation_readable, param_order, source_equation


def parameter_role(parameter_name: str, key_regressor: str, rd_var: str | None) -> str:
    if parameter_name == key_regressor:
        return "core_regressor"
    if parameter_name == "Intercept":
        return "intercept"
    if rd_var and parameter_name == rd_var:
        return "rd_regressor"
    if parameter_name in {"ln_asset", "lev_ratio", "gassets", "log_patent_count_ft"}:
        return "control"
    return "other"


def role_order(role: str) -> int:
    return {
        "core_regressor": 0,
        "rd_regressor": 1,
        "control": 2,
        "other": 3,
        "intercept": 4,
    }.get(role, 5)


def parse_parameter_estimates(path: Path) -> list[dict[str, float | str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: list[dict[str, float | str]] = []
    in_section = False
    seen_header = False

    for line in text:
        stripped = line.strip()
        if stripped == "Parameter Estimates":
            in_section = True
            continue
        if not in_section:
            continue
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("Parameter  Std. Err.") or stripped.startswith("Parameter  Std. Err"):
            seen_header = True
            continue
        if stripped.startswith("Parameter Estimates"):
            continue
        if set(stripped) <= {"=", "-"}:
            continue
        if not seen_header:
            continue

        match = PARAMETER_LINE_RE.match(line)
        if not match:
            if rows:
                break
            continue
        rows.append(
            {
                "parameter_name": match.group("name"),
                "coef": float(match.group("coef")),
                "se": float(match.group("se")),
                "t": float(match.group("t")),
                "p": float(match.group("p")),
                "ci_lower": float(match.group("lower")),
                "ci_upper": float(match.group("upper")),
            }
        )
    return rows


def fmt_num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0.0000"
    abs_value = abs(value)
    if abs_value >= 1000 or abs_value < 0.0001:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames}
            writer.writerow(normalized)


def build_group_file_path(output_root: Path, model: dict[str, object]) -> Path:
    return (
        output_root
        / "markdown"
        / str(model["window_label"])
        / str(model["family_bucket"])
        / f"{safe_slug(str(model['dep_var']))}.md"
    )


def render_models_table(models: list[dict[str, object]]) -> list[str]:
    lines = [
        "| spec_id | sample | variant | horizon | equation | nobs | nfirms | R^2(within) | source txt |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for model in models:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model["spec_id"]),
                    str(model["sample_label"]),
                    str(model["spec_variant"]),
                    str(model["future_horizon"]),
                    str(model["equation_readable"]),
                    str(model["nobs"]),
                    str(model["nfirms"]),
                    fmt_num(parse_float(str(model["rsq_within"])) if model["rsq_within"] not in ("", None) else None),
                    str(model["output_txt_rel"]),
                ]
            )
            + " |"
        )
    return lines


def render_parameter_table(parameter_rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| spec_id | parameter | role | coef | t | p |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in parameter_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["spec_id"]),
                    str(row["parameter_name"]),
                    str(row["parameter_role"]),
                    fmt_num(
                        parse_float(str(row["parameter_coef"])) if row["parameter_coef"] not in ("", None) else None
                    ),
                    fmt_num(parse_float(str(row["parameter_t"])) if row["parameter_t"] not in ("", None) else None),
                    fmt_num(parse_float(str(row["parameter_p"])) if row["parameter_p"] not in ("", None) else None),
                ]
            )
            + " |"
        )
    return lines


def sort_models(models: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        models,
        key=lambda item: (
            int(item["window_sort"]),
            BUCKET_ORDER[str(item["family_bucket"])],
            OUTCOME_GROUP_ORDER[str(item["outcome_group"])],
            DEP_VAR_ORDER.get(str(item["dep_var"]), 999),
            QUALITY_VAR_ORDER.get(str(item["key_regressor"]), 999),
            -int(item["sample_threshold"]),
            int(item["future_horizon"]),
            SPEC_VARIANT_ORDER.get(str(item["spec_variant"]), 999),
            str(item["spec_id"]),
        ),
    )


def sort_parameters(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda item: (
            int(item["window_sort"]),
            BUCKET_ORDER[str(item["family_bucket"])],
            OUTCOME_GROUP_ORDER[str(item["outcome_group"])],
            DEP_VAR_ORDER.get(str(item["dep_var"]), 999),
            QUALITY_VAR_ORDER.get(str(item["key_regressor"]), 999),
            -int(item["sample_threshold"]),
            int(item["future_horizon"]),
            SPEC_VARIANT_ORDER.get(str(item["spec_variant"]), 999),
            str(item["spec_id"]),
            int(item["parameter_sort_order"]),
            str(item["parameter_name"]),
        ),
    )


def sort_parameter_names(parameter_names: Iterable[str]) -> list[str]:
    return sorted(
        set(parameter_names),
        key=lambda name: (
            QUALITY_VAR_ORDER.get(name, 999) if name in QUALITY_VAR_ORDER else 900,
            role_order("intercept") if name == "Intercept" else 0,
            0 if name in {"ln_asset", "lev_ratio", "gassets", "log_patent_count_ft"} else 1,
            name != "Intercept",
            name,
        ),
    )


def build_parameter_wide_rows(
    model_rows: list[dict[str, object]],
    parameter_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    parameter_names = sort_parameter_names(row["parameter_name"] for row in parameter_rows)
    stats = ("coef", "t", "p")
    row_map = {str(row["spec_id"]): dict(row) for row in model_rows}

    for row in parameter_rows:
        spec_id = str(row["spec_id"])
        target = row_map[spec_id]
        parameter_name = str(row["parameter_name"])
        target[f"role__{parameter_name}"] = row["parameter_role"]
        target[f"coef__{parameter_name}"] = row["parameter_coef"]
        target[f"t__{parameter_name}"] = row["parameter_t"]
        target[f"p__{parameter_name}"] = row["parameter_p"]

    extra_fields: list[str] = []
    for parameter_name in parameter_names:
        extra_fields.extend(
            [
                f"role__{parameter_name}",
                f"coef__{parameter_name}",
                f"t__{parameter_name}",
                f"p__{parameter_name}",
            ]
        )
    return [row_map[str(row["spec_id"])] for row in model_rows], extra_fields


def write_markdown_files(
    output_root: Path,
    model_rows: list[dict[str, object]],
    parameter_rows: list[dict[str, object]],
) -> None:
    grouped_models: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    grouped_params: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    key_orders: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for model in model_rows:
        group_key = (str(model["window_label"]), str(model["family_bucket"]), str(model["dep_var"]))
        grouped_models[group_key].append(model)
        key_regressor = str(model["key_regressor"])
        if key_regressor not in key_orders[group_key]:
            key_orders[group_key].append(key_regressor)

    for row in parameter_rows:
        grouped_params[
            (
                str(row["window_label"]),
                str(row["family_bucket"]),
                str(row["dep_var"]),
                str(row["key_regressor"]),
            )
        ].append(row)

    for group_key, models in grouped_models.items():
        window_label, family_bucket, dep_var = group_key
        file_path = output_root / "markdown" / window_label / family_bucket / f"{safe_slug(dep_var)}.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        ordered_models = sort_models(models)
        lines = [
            f"# {window_label} / {BUCKET_LABELS[family_bucket]} / {dep_var}",
            "",
            f"- 因变量标签：{DEP_VAR_LABELS.get(dep_var, dep_var)}",
            f"- 因变量分组：{OUTCOME_GROUP_LABELS[detect_outcome_group(dep_var)]}",
            f"- 回归数量：{len(ordered_models)}",
            "",
        ]

        for key_regressor in sorted(key_orders[group_key], key=lambda item: QUALITY_VAR_ORDER.get(item, 999)):
            subset_models = [item for item in ordered_models if str(item["key_regressor"]) == key_regressor]
            subset_params = sort_parameters(
                grouped_params[(window_label, family_bucket, dep_var, key_regressor)]
            )

            lines.extend(
                [
                    f"## {key_regressor}",
                    "",
                    f"- 核心变量标签：{QUALITY_VAR_LABELS.get(key_regressor, key_regressor)}",
                    "",
                    "### 规格表",
                    "",
                    *render_models_table(subset_models),
                    "",
                    "### 参数表",
                    "",
                    *render_parameter_table(subset_params),
                    "",
                ]
            )

        file_path.write_text("\n".join(lines), encoding="utf-8")

    index_lines = [
        "# 回归结果提取索引",
        "",
        "- 根目录：`outputs/experiments/标题_摘要_ExactTime_window_1_vs_3_regression_extract/`",
        "- 这里的 Markdown 只做人工查阅；完整明细请优先看 `csv/` 下两张表。",
        "",
    ]

    for experiment in EXPERIMENTS:
        index_lines.extend([f"## {experiment.window_label}", ""])
        for family_bucket in ("current_main", "current_rd", "future"):
            index_lines.extend([f"### {BUCKET_LABELS[family_bucket]}", ""])
            family_models = [
                model
                for model in model_rows
                if str(model["window_label"]) == experiment.window_label and str(model["family_bucket"]) == family_bucket
            ]
            for dep_var in sorted(
                {str(model["dep_var"]) for model in family_models},
                key=lambda item: DEP_VAR_ORDER.get(item, 999),
            ):
                rel_path = Path(experiment.window_label) / family_bucket / f"{safe_slug(dep_var)}.md"
                index_lines.append(f"- [{dep_var}]({rel_path.as_posix()})")
            index_lines.append("")

    (output_root / "markdown").mkdir(parents=True, exist_ok=True)
    (output_root / "markdown/INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")


def write_readme(output_root: Path, model_rows: list[dict[str, object]], parameter_rows: list[dict[str, object]]) -> None:
    lines = [
        "# ExactTime 双窗口回归结果提取",
        "",
        "本目录把以下两个实验下的回归结果重新整理为便于查阅的格式：",
        "",
        "- `outputs/experiments/标题_摘要_ExactTime_window_1/stage2_exact/`",
        "- `outputs/experiments/标题_摘要_ExactTime_window_3/stage2_exact/`",
        "",
        "生成文件：",
        "",
        "- `csv/regression_models.csv`：每个回归一行，包含回归方程、主系数、样本量、R^2、源 txt 路径。",
        "- `csv/regression_parameters_long.csv`：每个“回归-参数”一行，已把核心自变量排在最前，保留系数、标准误、t、p、置信区间。",
        "- `csv/regression_parameters_wide.csv`：每个回归一行，把所有参数展开成列，便于直接查看控制变量的 `coef / t / p`。",
        "- `markdown/INDEX.md`：按窗口 / 模块 / 因变量分组的阅读索引。",
        "- `markdown/<window>/<family_bucket>/<dep_var>.md`：适合人工顺序查阅的分组 Markdown。",
        "",
        "统计概览：",
        "",
        f"- 回归数量：{len(model_rows)}",
        f"- 参数行数量：{len(parameter_rows)}",
        "",
        "字段设计说明：",
        "",
        "- `equation_readable` 使用 `~` 形式，保留核心自变量在最前，并把固定效应显示为 `firm FE + year FE`。",
        "- `family_bucket` 把结果分成 `current_main`、`current_rd`、`future` 三大块。",
        "- `outcome_group` 按方案文档分成 `盈利能力`、`利润率与经营表现`、`成长性`。",
        "- `spec_variant` 保留原规格后缀，如 `cnt1`、`rdhorse`、`h3_cnt1`。",
        "",
        "建议用法：",
        "",
        "- 先看 `csv/regression_models.csv` 快速定位想要的规格。",
        "- 再用 `spec_id` 到 `csv/regression_parameters_long.csv` 看该回归的完整参数表。",
        "- 如果要人工通读，同一类规格可从 `markdown/INDEX.md` 进入对应分组文件。",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for experiment in EXPERIMENTS:
        sample_rows = {row["spec_id"]: row for row in read_csv_rows(experiment.sample_csv)}
        summary_rows = read_csv_rows(experiment.summary_csv)

        for summary in summary_rows:
            spec_id = summary["spec_id"]
            sample = sample_rows.get(spec_id, {})
            sample_threshold = parse_int(summary.get("sample_threshold")) or 0
            future_horizon = parse_int(summary.get("future_horizon")) or 0
            model_family = summary.get("model_family", "")
            family_bucket = detect_family_bucket(model_family, future_horizon)
            spec_variant = extract_spec_variant(spec_id, sample_threshold)
            outcome_group = detect_outcome_group(summary.get("dep_var", ""))
            equation_readable, parameter_order, equation_source_readable = parse_formula(
                summary.get("formula", ""),
                summary.get("dep_var", ""),
                future_horizon,
            )

            output_txt_rel = summary.get("output_txt", "")
            output_txt_abs = REPO_ROOT / output_txt_rel if output_txt_rel else None
            parameters = parse_parameter_estimates(output_txt_abs) if output_txt_abs and output_txt_abs.exists() else []
            order_lookup = {name: index for index, name in enumerate(parameter_order)}

            model_row = {
                "experiment_id": experiment.experiment_id,
                "window_label": experiment.window_label,
                "window_sort": experiment.window_sort,
                "family_bucket": family_bucket,
                "family_bucket_label": BUCKET_LABELS[family_bucket],
                "model_family": model_family,
                "model_family_label": MODEL_FAMILY_LABELS.get(model_family, model_family),
                "outcome_group": outcome_group,
                "outcome_group_label": OUTCOME_GROUP_LABELS[outcome_group],
                "dep_var": summary.get("dep_var", ""),
                "dep_var_label": DEP_VAR_LABELS.get(summary.get("dep_var", ""), summary.get("dep_var", "")),
                "dep_source": summary.get("dep_source", ""),
                "key_regressor": summary.get("key_regressor", ""),
                "key_regressor_label": QUALITY_VAR_LABELS.get(
                    summary.get("key_regressor", ""), summary.get("key_regressor", "")
                ),
                "rd_var": summary.get("rd_var", ""),
                "rd_var_label": QUALITY_VAR_LABELS.get(summary.get("rd_var", ""), summary.get("rd_var", "")),
                "sample_threshold": sample_threshold,
                "sample_label": f"pc{sample_threshold}",
                "sample_rule": summary.get("sample_rule", ""),
                "year_range": sample.get("year_range") or summary.get("year_range", ""),
                "future_horizon": future_horizon,
                "spec_id": spec_id,
                "spec_variant": spec_variant,
                "status": summary.get("status", ""),
                "add_log_patent_count": parse_bool(summary.get("add_log_patent_count")),
                "controls": summary.get("controls", ""),
                "formula_source": summary.get("formula", ""),
                "equation_readable": equation_readable,
                "equation_source_readable": equation_source_readable,
                "coef": parse_float(summary.get("coef")),
                "se": parse_float(summary.get("se")),
                "t": parse_float(summary.get("t")),
                "p": parse_float(summary.get("p")),
                "rd_coef": parse_float(summary.get("rd_coef")),
                "rd_se": parse_float(summary.get("rd_se")),
                "rd_p": parse_float(summary.get("rd_p")),
                "nobs": parse_int(summary.get("nobs")),
                "nfirms": parse_int(summary.get("nfirms")),
                "rsq_within": parse_float(summary.get("rsq_within")),
                "financial_universe_rows": parse_int(sample.get("financial_universe_rows")),
                "financial_universe_firms": parse_int(sample.get("financial_universe_firms")),
                "patent_matched_rows": parse_int(sample.get("patent_matched_rows")),
                "patent_matched_firms": parse_int(sample.get("patent_matched_firms")),
                "threshold_rows": parse_int(sample.get("threshold_rows")),
                "threshold_firms": parse_int(sample.get("threshold_firms")),
                "rd_available_rows": parse_int(sample.get("rd_available_rows")),
                "rd_available_firms": parse_int(sample.get("rd_available_firms")),
                "final_reg_nobs": parse_int(sample.get("final_reg_nobs")),
                "final_reg_firms": parse_int(sample.get("final_reg_firms")),
                "dropped_by_y_missing": parse_int(sample.get("dropped_by_y_missing")),
                "dropped_by_x_missing": parse_int(sample.get("dropped_by_x_missing")),
                "dropped_by_controls_missing": parse_int(sample.get("dropped_by_controls_missing")),
                "output_txt_rel": output_txt_rel,
                "output_txt_abs": str(output_txt_abs) if output_txt_abs else "",
                "error": summary.get("error", ""),
            }
            model_rows.append(model_row)

            for parameter in parameters:
                parameter_name = str(parameter["parameter_name"])
                param_role = parameter_role(parameter_name, str(model_row["key_regressor"]), str(model_row["rd_var"]) or None)
                parameter_rows.append(
                    {
                        **model_row,
                        "parameter_name": parameter_name,
                        "parameter_label": PARAMETER_LABELS.get(parameter_name, parameter_name),
                        "parameter_role": param_role,
                        "parameter_sort_order": order_lookup.get(parameter_name, 999) * 10 + role_order(param_role),
                        "parameter_coef": parameter["coef"],
                        "parameter_se": parameter["se"],
                        "parameter_t": parameter["t"],
                        "parameter_p": parameter["p"],
                        "parameter_ci_lower": parameter["ci_lower"],
                        "parameter_ci_upper": parameter["ci_upper"],
                    }
                )

    sorted_models = sort_models(model_rows)
    sorted_parameters = sort_parameters(parameter_rows)

    model_fieldnames = [
        "experiment_id",
        "window_label",
        "window_sort",
        "family_bucket",
        "family_bucket_label",
        "model_family",
        "model_family_label",
        "outcome_group",
        "outcome_group_label",
        "dep_var",
        "dep_var_label",
        "dep_source",
        "key_regressor",
        "key_regressor_label",
        "rd_var",
        "rd_var_label",
        "sample_threshold",
        "sample_label",
        "sample_rule",
        "year_range",
        "future_horizon",
        "spec_id",
        "spec_variant",
        "status",
        "add_log_patent_count",
        "controls",
        "formula_source",
        "equation_readable",
        "equation_source_readable",
        "coef",
        "se",
        "t",
        "p",
        "rd_coef",
        "rd_se",
        "rd_p",
        "nobs",
        "nfirms",
        "rsq_within",
        "financial_universe_rows",
        "financial_universe_firms",
        "patent_matched_rows",
        "patent_matched_firms",
        "threshold_rows",
        "threshold_firms",
        "rd_available_rows",
        "rd_available_firms",
        "final_reg_nobs",
        "final_reg_firms",
        "dropped_by_y_missing",
        "dropped_by_x_missing",
        "dropped_by_controls_missing",
        "output_txt_rel",
        "output_txt_abs",
        "error",
    ]

    parameter_fieldnames = [
        *model_fieldnames,
        "parameter_name",
        "parameter_label",
        "parameter_role",
        "parameter_sort_order",
        "parameter_coef",
        "parameter_se",
        "parameter_t",
        "parameter_p",
        "parameter_ci_lower",
        "parameter_ci_upper",
    ]

    parameter_wide_rows, parameter_wide_extra_fields = build_parameter_wide_rows(sorted_models, sorted_parameters)
    parameter_wide_fieldnames = [*model_fieldnames, *parameter_wide_extra_fields]

    write_csv(output_root / "csv/regression_models.csv", sorted_models, model_fieldnames)
    write_csv(output_root / "csv/regression_parameters_long.csv", sorted_parameters, parameter_fieldnames)
    write_csv(output_root / "csv/regression_parameters_wide.csv", parameter_wide_rows, parameter_wide_fieldnames)
    write_markdown_files(output_root, sorted_models, sorted_parameters)
    write_readme(output_root, sorted_models, sorted_parameters)


if __name__ == "__main__":
    main()
