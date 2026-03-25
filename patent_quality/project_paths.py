from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = "outputs/experiments"


def get_project_root() -> Path:
    return PROJECT_ROOT


def resolve_project_path(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    anchor = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (anchor / path_obj).resolve()


def to_display_path(path: PathLike) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return path_obj.as_posix()


@dataclass(frozen=True)
class ExperimentLayout:
    experiment_id: str
    root: Path
    stage1_dir: Path
    stage2_dir: Path
    verification_dir: Path

    def ensure_stage1_dirs(self) -> None:
        for directory in (
            self.root,
            self.stage1_dir,
            self.stage1_dir / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_stage2_dirs(self) -> None:
        for directory in (
            self.root,
            self.stage2_dir,
            self.stage2_dir / "metadata",
            self.stage2_dir / "data",
            self.stage2_dir / "diagnostics",
            self.stage2_dir / "figures",
            self.stage2_dir / "tables",
            self.stage2_dir / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_verification_dirs(self) -> None:
        for directory in (
            self.root,
            self.verification_dir,
            self.verification_dir / "ir",
            self.verification_dir / "matrix",
            self.verification_dir / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def stage1_log_path(self, filename: str = "stage1.log") -> Path:
        return self.stage1_dir / "logs" / filename


def build_experiment_layout(
    experiment_id: str,
    output_root: PathLike = DEFAULT_OUTPUT_ROOT,
) -> ExperimentLayout:
    root = resolve_project_path(output_root) / experiment_id
    return ExperimentLayout(
        experiment_id=experiment_id,
        root=root,
        stage1_dir=root / "stage1",
        stage2_dir=root / "stage2",
        verification_dir=root / "verification",
    )


def infer_experiment_id_from_stage1_dir(stage1_dir: PathLike) -> str:
    path_obj = resolve_project_path(stage1_dir)
    if path_obj.name == "stage1" and path_obj.parent.name:
        return path_obj.parent.name
    return path_obj.name
