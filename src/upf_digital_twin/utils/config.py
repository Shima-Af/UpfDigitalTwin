"""Config loading utility.

Loads configs/paths.yaml and configs/scenario.yaml relative to the project root.
The project root is detected as the directory containing the configs/ folder,
walking up from this file's location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not find a 'configs/' directory above {start}. "
        "Run from the project root or set the working directory correctly."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_configs(
    project_root: Path | None = None,
    paths_file: str = "configs/paths.yaml",
    scenario_file: str = "configs/scenario.yaml",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (paths_cfg, scenario_cfg) loaded from YAML.

    Args:
        project_root: Explicit root; auto-detected if None.
        paths_file:   Path relative to project root for paths config.
        scenario_file: Path relative to project root for scenario config.
    """
    if project_root is None:
        project_root = _find_project_root(Path(__file__).resolve().parent)

    project_root = Path(project_root)
    paths_cfg = load_yaml(project_root / paths_file)
    scenario_cfg = load_yaml(project_root / scenario_file)
    return paths_cfg, scenario_cfg


def resolve_path(project_root: Path, relative: str) -> Path:
    """Resolve a relative path string from paths.yaml against the project root."""
    return project_root / relative
