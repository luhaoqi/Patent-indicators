from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, overload
import json
import sys


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT_CANDIDATE = CURRENT_FILE.parents[2]
if str(REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_CANDIDATE))

from patent_quality.project_paths import (
    DEFAULT_OUTPUT_ROOT,
    build_experiment_layout,
    get_project_root,
    resolve_project_path,
)


PathLike = Union[str, Path]

REPO_ROOT = get_project_root()


def get_repo_root() -> Path:
    return REPO_ROOT


@overload
def resolve_repo_path(path: None, base_dir: Optional[PathLike] = None) -> None:
    ...


@overload
def resolve_repo_path(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    ...


def resolve_repo_path(path: Optional[PathLike], base_dir: Optional[PathLike] = None) -> Optional[Path]:
    if path is None:
        return None
    return resolve_project_path(path, base_dir=base_dir)


def repo_relative(path: PathLike) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path_obj.as_posix()


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_id: str
    root: Path
    stage1_dir: Path
    metadata_dir: Path
    data_dir: Path
    diagnostics_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path

    def ensure_dirs(self) -> None:
        for directory in (
            self.root,
            self.stage1_dir,
            self.metadata_dir,
            self.data_dir,
            self.diagnostics_dir,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def stage2_log_path(self) -> Path:
        return self.logs_dir / "stage2.log"

    def to_metadata(self) -> Dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "root": repo_relative(self.root),
            "stage1_dir": repo_relative(self.stage1_dir),
            "metadata_dir": repo_relative(self.metadata_dir),
            "data_dir": repo_relative(self.data_dir),
            "diagnostics_dir": repo_relative(self.diagnostics_dir),
            "figures_dir": repo_relative(self.figures_dir),
            "tables_dir": repo_relative(self.tables_dir),
            "logs_dir": repo_relative(self.logs_dir),
        }


def build_experiment_paths(experiment_id: str, output_root: PathLike = DEFAULT_OUTPUT_ROOT) -> ExperimentPaths:
    layout = build_experiment_layout(experiment_id, output_root=output_root)
    return ExperimentPaths(
        experiment_id=experiment_id,
        root=layout.root,
        stage1_dir=layout.stage1_dir,
        metadata_dir=layout.stage2_dir / "metadata",
        data_dir=layout.stage2_dir / "data",
        diagnostics_dir=layout.stage2_dir / "diagnostics",
        figures_dir=layout.stage2_dir / "figures",
        tables_dir=layout.stage2_dir / "tables",
        logs_dir=layout.stage2_dir / "logs",
    )


def load_manifest(manifest_path: PathLike) -> Dict[str, Any]:
    resolved = resolve_repo_path(manifest_path)
    assert resolved is not None
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        return json.loads(resolved.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("读取 YAML manifest 需要安装 PyYAML") from exc
        return yaml.safe_load(resolved.read_text(encoding="utf-8"))
    raise ValueError(f"暂不支持的 manifest 格式: {resolved.name}")
