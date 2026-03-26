from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Stage2InputsConfig:
    stage1_dir: str
    shared_root: str


@dataclass(frozen=True)
class DiagnosticsConfig:
    topk_values: tuple[int, ...] = (10, 30, 50)
    yearly_top_vocab_k: int = 50
    max_year_gap: int = 5


@dataclass(frozen=True)
class ExperimentPatentPanelConfig:
    chunksize: int = 100000


@dataclass(frozen=True)
class QualityBasicConfig:
    exclude_years: tuple[int, ...] = (1985, 1986)
    quality_min: float = 1e-5
    bs_min: float = 1e-6
    quality_desc_threshold: float = 5.0


@dataclass(frozen=True)
class SpecialFirmsConfig:
    exclude_years: tuple[int, ...] = (1985, 1986)
    quality_min: float = 1e-5
    bs_min: float = 1e-6
    quality_threshold: float = 1.0
    policy_start_year: int = 2008
    event_window: int = 5


@dataclass(frozen=True)
class InnovationConfig:
    top_k: int = 10
    quality_cap: float = 1000.0


@dataclass(frozen=True)
class RegressionConfig:
    year_min: int = 2000
    year_max: int = 2023


@dataclass(frozen=True)
class Stage2Config:
    experiment_id: str
    output_root: str
    inputs: Stage2InputsConfig
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    build_experiment_patent_panel: ExperimentPatentPanelConfig = field(default_factory=ExperimentPatentPanelConfig)
    analyze_quality_basic: QualityBasicConfig = field(default_factory=QualityBasicConfig)
    analyze_special_firms: SpecialFirmsConfig = field(default_factory=SpecialFirmsConfig)
    build_firm_year_innovation: InnovationConfig = field(default_factory=InnovationConfig)
    run_regressions: RegressionConfig = field(default_factory=RegressionConfig)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_runtime(
        cls,
        *,
        experiment_id: str,
        stage1_dir: Path,
        shared_root: Path,
        output_root: str,
        topk_values: Sequence[int],
        yearly_top_vocab_k: int,
        max_year_gap: int,
        exclude_years: Sequence[int],
        quality_min: float,
        bs_min: float,
        analysis_quality_threshold: float,
        quality_desc_threshold: float,
        policy_start_year: int,
        event_window: int,
        innovation_top_k: int,
        innovation_quality_cap: float,
        regression_year_min: int,
        regression_year_max: int,
        chunksize: int,
        notes: Optional[dict[str, Any]] = None,
    ) -> "Stage2Config":
        grouped_exclude_years = tuple(int(value) for value in exclude_years)
        return cls(
            experiment_id=experiment_id,
            output_root=output_root,
            inputs=Stage2InputsConfig(
                stage1_dir=str(stage1_dir),
                shared_root=str(shared_root),
            ),
            diagnostics=DiagnosticsConfig(
                topk_values=tuple(int(value) for value in topk_values),
                yearly_top_vocab_k=int(yearly_top_vocab_k),
                max_year_gap=int(max_year_gap),
            ),
            build_experiment_patent_panel=ExperimentPatentPanelConfig(chunksize=int(chunksize)),
            analyze_quality_basic=QualityBasicConfig(
                exclude_years=grouped_exclude_years,
                quality_min=float(quality_min),
                bs_min=float(bs_min),
                quality_desc_threshold=float(quality_desc_threshold),
            ),
            analyze_special_firms=SpecialFirmsConfig(
                exclude_years=grouped_exclude_years,
                quality_min=float(quality_min),
                bs_min=float(bs_min),
                quality_threshold=float(analysis_quality_threshold),
                policy_start_year=int(policy_start_year),
                event_window=int(event_window),
            ),
            build_firm_year_innovation=InnovationConfig(
                top_k=int(innovation_top_k),
                quality_cap=float(innovation_quality_cap),
            ),
            run_regressions=RegressionConfig(
                year_min=int(regression_year_min),
                year_max=int(regression_year_max),
            ),
            notes=notes or {},
        )
