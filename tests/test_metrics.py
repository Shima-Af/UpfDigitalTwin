"""Rollout metrics (pure pandas, no model/data dependency)."""

from __future__ import annotations

import pandas as pd
import pytest

from upf_digital_twin.evaluation.metrics import compute_metrics


def _df(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with sample_idx, cluster_idx, selected_upf, power, switch, safe, pending."""
    return pd.DataFrame([
        {
            "sample_idx": r["sample_idx"],
            "cluster_idx": r["cluster_idx"],
            "selected_upf": r["selected_upf"],
            "power_watts": r["power"],
            "switching_energy_wh": r["switch"],
            "is_safe": r["safe"],
            "activation_pending": r["pending"],
        }
        for r in rows
    ])


def _sample_df() -> pd.DataFrame:
    return _df([
        # cluster 0: one flip DPDK->USR, one unsafe USR step
        {"sample_idx": 0, "cluster_idx": 0, "selected_upf": "DPDK", "power": 1.0, "switch": 0.0, "safe": True,  "pending": False},
        {"sample_idx": 1, "cluster_idx": 0, "selected_upf": "USR",  "power": 0.5, "switch": 0.1, "safe": True,  "pending": False},
        {"sample_idx": 2, "cluster_idx": 0, "selected_upf": "USR",  "power": 0.5, "switch": 0.0, "safe": False, "pending": False},
        # cluster 1: constant DPDK, one activation-pending step
        {"sample_idx": 0, "cluster_idx": 1, "selected_upf": "DPDK", "power": 1.0, "switch": 0.0, "safe": True,  "pending": True},
        {"sample_idx": 1, "cluster_idx": 1, "selected_upf": "DPDK", "power": 1.0, "switch": 0.0, "safe": True,  "pending": False},
        {"sample_idx": 2, "cluster_idx": 1, "selected_upf": "DPDK", "power": 1.0, "switch": 0.0, "safe": True,  "pending": False},
    ])


def test_energy_decomposition_and_usage():
    m = compute_metrics(_sample_df(), time_step_minutes=15)  # dt_h = 0.25
    assert m["n_steps"] == 6
    assert m["steady_energy_wh"] == pytest.approx(5.0 * 0.25)   # sum(power)=5.0
    assert m["switch_energy_wh"] == pytest.approx(0.1)
    assert m["total_energy_wh"] == pytest.approx(1.35)
    assert m["average_power_w"] == pytest.approx(5.0 / 6)
    assert m["usr_usage_ratio"] == pytest.approx(2 / 6)
    assert m["dpdk_usage_ratio"] == pytest.approx(4 / 6)
    assert m["activation_pending_rate"] == pytest.approx(1 / 6)


def test_unsafe_usr_rate_only_counts_usr_steps():
    m = compute_metrics(_sample_df(), time_step_minutes=15)
    # 2 USR steps, 1 unsafe -> 0.5
    assert m["unsafe_usr_rate"] == pytest.approx(0.5)


def test_flip_rate_is_per_cluster_mean():
    m = compute_metrics(_sample_df(), time_step_minutes=15)
    # cluster 0: 1 flip / 2 transitions = 0.5 ; cluster 1: 0 -> mean 0.25
    assert m["decision_flip_rate"] == pytest.approx(0.25)


def test_energy_saving_and_regret_against_baselines():
    df = _sample_df()
    ref = _df([
        {"sample_idx": i, "cluster_idx": c, "selected_upf": "DPDK", "power": 1.0, "switch": 0.0, "safe": True, "pending": False}
        for c in (0, 1) for i in range(3)
    ])  # ref total = 6*1.0*0.25 = 1.5
    oracle = _df([
        {"sample_idx": i, "cluster_idx": c, "selected_upf": "USR", "power": 0.4, "switch": 0.0, "safe": True, "pending": False}
        for c in (0, 1) for i in range(3)
    ])  # oracle total = 6*0.4*0.25 = 0.6
    m = compute_metrics(df, time_step_minutes=15, reference_dpdk_df=ref, oracle_df=oracle)
    assert m["energy_saving_vs_static_dpdk_pct"] == pytest.approx((1.5 - 1.35) / 1.5 * 100)  # 10%
    assert m["energy_regret_vs_oracle_wh"] == pytest.approx(1.35 - 0.6)                       # 0.75


def test_baselines_optional():
    m = compute_metrics(_sample_df(), time_step_minutes=15)
    assert m["energy_saving_vs_static_dpdk_pct"] is None
    assert m["energy_regret_vs_oracle_wh"] is None


def test_no_usr_steps_gives_zero_unsafe_rate():
    df = _df([
        {"sample_idx": i, "cluster_idx": 0, "selected_upf": "DPDK", "power": 1.0, "switch": 0.0, "safe": True, "pending": False}
        for i in range(3)
    ])
    m = compute_metrics(df, time_step_minutes=15)
    assert m["unsafe_usr_rate"] == 0.0
    assert m["usr_usage_ratio"] == 0.0
