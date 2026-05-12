"""Evaluation metrics for digital-twin rollouts.

Operates on the flat DataFrame produced by rollout.run_rollout().
Energy uses the time step from scenario.yaml.

Total energy includes BOTH steady-state energy (power × dt) AND switching
energy spikes (modeled by the twin) so policies are penalised for flipping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    df: pd.DataFrame,
    time_step_minutes: float,
    reference_dpdk_df: pd.DataFrame | None = None,
    oracle_df:         pd.DataFrame | None = None,
) -> dict:
    """Compute all metrics for one rollout.

    Args:
        df:                 Rollout DataFrame from run_rollout().
        time_step_minutes:  Interval duration in minutes.
        reference_dpdk_df:  Static-DPDK rollout for energy-saving baseline.
        oracle_df:          Oracle rollout for energy-regret baseline.
    """
    dt_h = time_step_minutes / 60.0
    n_steps = len(df)

    steady_energy_wh = float((df["power_watts"] * dt_h).sum())
    switch_energy_wh = float(df["switching_energy_wh"].sum())
    total_energy_wh  = steady_energy_wh + switch_energy_wh
    average_power_w  = float(df["power_watts"].mean())

    # Energy saving vs static DPDK
    energy_saving_pct = None
    if reference_dpdk_df is not None:
        ref_steady = float((reference_dpdk_df["power_watts"] * dt_h).sum())
        ref_switch = float(reference_dpdk_df["switching_energy_wh"].sum())
        ref_total = ref_steady + ref_switch
        if ref_total > 0:
            energy_saving_pct = (ref_total - total_energy_wh) / ref_total * 100.0

    # USR safety: of all USR-selected steps, fraction that violated QoS
    usr_mask = df["selected_upf"] == "USR"
    n_usr = int(usr_mask.sum())
    unsafe_usr_rate = float((~df.loc[usr_mask, "is_safe"]).mean()) if n_usr > 0 else 0.0

    # UPF usage
    usr_usage_ratio  = n_usr / n_steps if n_steps > 0 else 0.0
    dpdk_usage_ratio = 1.0 - usr_usage_ratio

    # Activation pending: fraction of steps where requested != realised
    activation_pending_rate = float(df["activation_pending"].mean())

    # Decision flip rate (per cluster, in time order)
    flip_rates = []
    for _, grp in df.groupby("cluster_idx"):
        actions = grp.sort_values("sample_idx")["selected_upf"].values
        if len(actions) > 1:
            flips = (actions[1:] != actions[:-1]).sum()
            flip_rates.append(flips / (len(actions) - 1))
    decision_flip_rate = float(np.mean(flip_rates)) if flip_rates else 0.0

    # Energy regret vs oracle
    energy_regret_wh = None
    if oracle_df is not None:
        oracle_total = (
            float((oracle_df["power_watts"] * dt_h).sum())
            + float(oracle_df["switching_energy_wh"].sum())
        )
        energy_regret_wh = total_energy_wh - oracle_total

    return {
        "n_steps":                          n_steps,
        "total_energy_wh":                  total_energy_wh,
        "steady_energy_wh":                 steady_energy_wh,
        "switch_energy_wh":                 switch_energy_wh,
        "average_power_w":                  average_power_w,
        "energy_saving_vs_static_dpdk_pct": energy_saving_pct,
        "unsafe_usr_rate":                  unsafe_usr_rate,
        "usr_usage_ratio":                  usr_usage_ratio,
        "dpdk_usage_ratio":                 dpdk_usage_ratio,
        "activation_pending_rate":          activation_pending_rate,
        "decision_flip_rate":               decision_flip_rate,
        "energy_regret_vs_oracle_wh":       energy_regret_wh,
    }
