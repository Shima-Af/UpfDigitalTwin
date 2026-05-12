"""Traffic artifact loader.

Loads the six forecaster output files produced by UpfTrafficForecaster
(https://github.com/Shima-Af/UpfTrafficForecaster/tree/feature/cluster-first-stgnn)
and validates their shapes and contents before returning a clean dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrafficArtifacts:
    """Validated container for all traffic forecaster outputs."""

    predictions_norm: np.ndarray      # (N, H, K)  normalised predicted traffic
    targets_norm: np.ndarray          # (N, H, K)  normalised actual traffic
    cluster_series_norm: np.ndarray   # (K, T)     full normalised time series
    cluster_assignments: pd.DataFrame # BS → cluster mapping
    cluster_bs_map: dict[str, list]   # cluster_id → list of BS ids
    forecast_eval_summary: dict[str, Any]
    service: str
    selected_k: int


def load_traffic_artifacts(paths_cfg: dict, scenario_cfg: dict, project_root: Path) -> TrafficArtifacts:
    """Load and validate all traffic artifacts.

    Args:
        paths_cfg:    Contents of configs/paths.yaml.
        scenario_cfg: Contents of configs/scenario.yaml.
        project_root: Project root Path for resolving relative paths.

    Returns:
        Validated TrafficArtifacts dataclass.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If shapes, service, or K do not match expectations.
    """
    tf = paths_cfg["traffic_forecaster"]
    traffic_cfg = scenario_cfg["traffic"]
    selected_k: int = int(traffic_cfg["selected_k"])
    expected_service: str = traffic_cfg["service"]

    def _resolve(key: str) -> Path:
        p = project_root / tf[key]
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")
        return p

    # --- Load arrays ---
    predictions_norm = np.load(_resolve("predictions"))
    targets_norm = np.load(_resolve("targets"))
    cluster_series_norm = np.load(_resolve("cluster_series"))
    cluster_assignments = pd.read_parquet(_resolve("cluster_assignments"))

    with open(_resolve("cluster_bs_map")) as fh:
        cluster_bs_map: dict = json.load(fh)

    with open(_resolve("forecast_eval_summary")) as fh:
        forecast_eval_summary: dict = json.load(fh)

    # --- Validate predictions / targets ---
    if predictions_norm.shape != targets_norm.shape:
        raise ValueError(
            f"predictions shape {predictions_norm.shape} != "
            f"targets shape {targets_norm.shape}"
        )

    if predictions_norm.ndim != 3:
        raise ValueError(
            f"Expected 3-D arrays (N, H, K), got shape {predictions_norm.shape}"
        )

    n, h, k = predictions_norm.shape
    if k != selected_k:
        raise ValueError(
            f"scenario.yaml selected_k={selected_k} but arrays have K={k}. "
            "Update selected_k or point to the correct K directory."
        )

    # --- Validate cluster_series ---
    if cluster_series_norm.shape[0] != selected_k:
        raise ValueError(
            f"cluster_series has {cluster_series_norm.shape[0]} rows but selected_k={selected_k}"
        )

    # --- Validate service field ---
    summary_service = forecast_eval_summary.get("service", "")
    if summary_service.lower() != expected_service.lower():
        raise ValueError(
            f"scenario service='{expected_service}' but forecast_eval_summary "
            f"service='{summary_service}'"
        )

    # --- Validate no NaN / inf ---
    for name, arr in [("predictions", predictions_norm), ("targets", targets_norm)]:
        if not np.isfinite(arr).all():
            n_bad = (~np.isfinite(arr)).sum()
            raise ValueError(f"{name} contains {n_bad} NaN or inf values")

    return TrafficArtifacts(
        predictions_norm=predictions_norm,
        targets_norm=targets_norm,
        cluster_series_norm=cluster_series_norm,
        cluster_assignments=cluster_assignments,
        cluster_bs_map=cluster_bs_map,
        forecast_eval_summary=forecast_eval_summary,
        service=expected_service,
        selected_k=selected_k,
    )
