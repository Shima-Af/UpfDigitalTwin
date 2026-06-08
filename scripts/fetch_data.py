#!/usr/bin/env python3
"""Fetch / verify the external data this project depends on.

The data (traffic-forecaster artifacts + profiling-twin surrogate models) is
tracked with DVC, not committed to git. On a fresh clone the working tree has
only the small ``*.dvc`` pointer files; this script materialises the actual
files and verifies they are all present, so the pipeline fails loudly *here*
with an actionable message instead of crashing deep inside model loading.

Usage (from the project root):
    python scripts/fetch_data.py            # verify; pull via DVC if anything missing
    python scripts/fetch_data.py --no-pull  # verify only, never touch the network

Exit code is 0 when every required file is present, 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upf_digital_twin.utils.config import load_configs  # noqa: E402

# Which source repo each data subtree comes from (shown when files are missing).
SOURCE_REPOS = {
    "traffic_forecaster": (
        "UpfTrafficForecaster @ feature/cluster-first-stgnn — "
        "https://github.com/Shima-Af/UpfTrafficForecaster/tree/feature/cluster-first-stgnn"
    ),
    "profiling_twin": (
        "UpfProfilingCampaign — https://github.com/Shima-Af/UpfProfilingCampaign"
    ),
}


def required_paths(paths_cfg: dict) -> dict[str, list[Path]]:
    """Build {group: [required files]} from configs/paths.yaml.

    Keys named ``dir`` are skipped; ``models`` is treated as a directory that
    must exist and be non-empty (checked separately below).
    """
    out: dict[str, list[Path]] = {}
    for group in ("traffic_forecaster", "profiling_twin"):
        files: list[Path] = []
        for key, rel in paths_cfg.get(group, {}).items():
            if key in ("dir", "models"):
                continue
            files.append(PROJECT_ROOT / rel)
        out[group] = files
    return out


def models_dir(paths_cfg: dict) -> Path:
    return PROJECT_ROOT / paths_cfg["profiling_twin"]["models"]


def find_missing(paths_cfg: dict) -> dict[str, list[Path]]:
    """Return {group: [missing files]} for any required file not on disk."""
    missing: dict[str, list[Path]] = {}
    for group, files in required_paths(paths_cfg).items():
        absent = [p for p in files if not p.exists()]
        missing[group] = absent

    # models/ must exist and contain at least one .pkl
    mdir = models_dir(paths_cfg)
    if not mdir.exists() or not any(mdir.rglob("*.pkl")):
        missing.setdefault("profiling_twin", []).append(mdir / "<surrogate models>")
    return {g: a for g, a in missing.items() if a}


def dvc_pull() -> bool:
    """Attempt ``dvc pull``. Returns True on success, False if unavailable/failed."""
    if shutil.which("dvc") is None:
        print("  dvc CLI not found — cannot pull automatically.")
        return False
    print("  Running `dvc pull` ...")
    proc = subprocess.run(["dvc", "pull"], cwd=PROJECT_ROOT)
    return proc.returncode == 0


def report_missing(missing: dict[str, list[Path]]) -> None:
    print("\nMISSING DATA — the following required files are not present:\n")
    for group, files in missing.items():
        print(f"  [{group}]  source: {SOURCE_REPOS.get(group, '?')}")
        for p in files:
            try:
                rel = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = p
            print(f"      - {rel}")
        print()
    print("To retrieve them:")
    print("  • If a DVC remote is configured + populated:  dvc pull")
    print("  • Otherwise copy the files from the source repos above into data/external/.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pull", action="store_true",
                        help="verify only; do not attempt `dvc pull`")
    args = parser.parse_args()

    print("\n=== UPF Digital Twin — data fetch / verify ===\n")
    paths_cfg, _ = load_configs(PROJECT_ROOT)

    missing = find_missing(paths_cfg)

    if missing and not args.no_pull:
        print("Some data is missing — attempting DVC pull.")
        if dvc_pull():
            missing = find_missing(paths_cfg)

    if missing:
        report_missing(missing)
        print("=== Data check FAILED. ===\n")
        return 1

    n_files = sum(len(f) for f in required_paths(paths_cfg).values())
    print(f"All required data present ({n_files} files + surrogate models).")
    if shutil.which("dvc") is not None:
        subprocess.run(["dvc", "status"], cwd=PROJECT_ROOT)
    print("\n=== Data check passed. ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
