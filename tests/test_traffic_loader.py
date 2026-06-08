"""Traffic artifact loading + validation, against a synthetic on-disk dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from upf_digital_twin.data.traffic_loader import load_traffic_artifacts

TF_DIR = "data/external/traffic_forecaster"


def _paths_cfg() -> dict:
    return {
        "traffic_forecaster": {
            "dir": TF_DIR,
            "predictions": f"{TF_DIR}/predictions_test.npy",
            "targets": f"{TF_DIR}/targets_test.npy",
            "cluster_series": f"{TF_DIR}/cluster_series.npy",
            "cluster_assignments": f"{TF_DIR}/cluster_assignments.parquet",
            "cluster_bs_map": f"{TF_DIR}/cluster_bs_map.json",
            "forecast_eval_summary": f"{TF_DIR}/forecast_eval_summary.json",
        }
    }


def _scenario_cfg(selected_k: int = 3, service: str = "Netflix") -> dict:
    return {"traffic": {"selected_k": selected_k, "service": service}}


def _write_dataset(root: Path, *, predictions=None, targets=None, cluster_series=None,
                   service: str = "Netflix") -> None:
    d = root / TF_DIR
    d.mkdir(parents=True, exist_ok=True)
    N, H, K, T = 4, 2, 3, 10
    pred = predictions if predictions is not None else np.random.rand(N, H, K).astype("float32")
    tgt = targets if targets is not None else np.random.rand(N, H, K).astype("float32")
    cs = cluster_series if cluster_series is not None else np.random.rand(K, T).astype("float32")
    np.save(d / "predictions_test.npy", pred)
    np.save(d / "targets_test.npy", tgt)
    np.save(d / "cluster_series.npy", cs)
    pd.DataFrame({"site_id": [1, 2, 3], "cluster_id": [0, 1, 2]}).to_parquet(d / "cluster_assignments.parquet")
    (d / "cluster_bs_map.json").write_text(json.dumps({"0": [1], "1": [2], "2": [3]}))
    (d / "forecast_eval_summary.json").write_text(
        json.dumps({"service": service, "results": [{"K": 3, "test_mae": 0.1}]})
    )


def test_happy_path(tmp_path):
    _write_dataset(tmp_path)
    art = load_traffic_artifacts(_paths_cfg(), _scenario_cfg(), tmp_path)
    assert art.predictions_norm.shape == (4, 2, 3)
    assert art.targets_norm.shape == (4, 2, 3)
    assert art.selected_k == 3
    assert art.service == "Netflix"
    assert art.cluster_series_norm.shape[0] == 3


def test_shape_mismatch_raises(tmp_path):
    _write_dataset(tmp_path, targets=np.random.rand(4, 2, 2).astype("float32"))
    with pytest.raises(ValueError, match="shape"):
        load_traffic_artifacts(_paths_cfg(), _scenario_cfg(), tmp_path)


def test_wrong_ndim_raises(tmp_path):
    flat = np.random.rand(4, 3).astype("float32")
    _write_dataset(tmp_path, predictions=flat, targets=flat, cluster_series=np.random.rand(3, 10).astype("float32"))
    with pytest.raises(ValueError, match="3-D"):
        load_traffic_artifacts(_paths_cfg(), _scenario_cfg(), tmp_path)


def test_selected_k_mismatch_raises(tmp_path):
    _write_dataset(tmp_path)  # arrays have K=3
    with pytest.raises(ValueError, match="selected_k"):
        load_traffic_artifacts(_paths_cfg(), _scenario_cfg(selected_k=5), tmp_path)


def test_service_mismatch_raises(tmp_path):
    _write_dataset(tmp_path, service="YouTube")
    with pytest.raises(ValueError, match="service"):
        load_traffic_artifacts(_paths_cfg(), _scenario_cfg(service="Netflix"), tmp_path)


def test_nan_values_raise(tmp_path):
    bad = np.random.rand(4, 2, 3).astype("float32")
    bad[0, 0, 0] = np.nan
    _write_dataset(tmp_path, predictions=bad)
    with pytest.raises(ValueError, match="NaN"):
        load_traffic_artifacts(_paths_cfg(), _scenario_cfg(), tmp_path)


def test_missing_file_raises(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / TF_DIR / "predictions_test.npy").unlink()
    with pytest.raises(FileNotFoundError):
        load_traffic_artifacts(_paths_cfg(), _scenario_cfg(), tmp_path)
