#!/usr/bin/env python3
"""Threshold demo — compares threshold + hysteresis policies against static and oracle baselines.

Run from the project root:
    python scripts/run_threshold_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from upf_digital_twin.utils.config import load_configs
from upf_digital_twin.data.traffic_loader import load_traffic_artifacts
from upf_digital_twin.twin.digital_twin import DigitalTwin
from upf_digital_twin.twin.threshold_derivation import derive_thresholds, load_forecast_mae_gbps
from upf_digital_twin.policies.static     import StaticDPDKPolicy, StaticUSRPolicy
from upf_digital_twin.policies.threshold  import ThresholdPolicy
from upf_digital_twin.policies.oracle     import OracleThresholdPolicy
from upf_digital_twin.policies.hysteresis import HysteresisPolicy
from upf_digital_twin.evaluation.rollout  import run_rollout, precompute_twin_for_rollouts
from upf_digital_twin.evaluation.metrics  import compute_metrics


def main() -> None:
    print("\n=== UPF Digital Twin — Threshold Demo ===\n")

    paths_cfg, scenario_cfg = load_configs(PROJECT_ROOT)
    traffic_cfg   = scenario_cfg["traffic"]
    threshold_cfg = scenario_cfg["threshold"]

    alpha             = traffic_cfg["calibration"]["alpha_gbps_per_norm"]
    time_step_minutes = traffic_cfg["time_step_minutes"]
    safety_margin_mbps = float(threshold_cfg["safety_margin_mbps"])
    cooldown_steps     = int(threshold_cfg["cooldown_steps"])

    print("Loading traffic artifacts...")
    artifacts = load_traffic_artifacts(paths_cfg, scenario_cfg, PROJECT_ROOT)
    N, H, K = artifacts.predictions_norm.shape
    print(f"  Shape: N={N} windows, H={H} horizon, K={K} clusters")

    twin = DigitalTwin(scenario_cfg, paths_cfg, PROJECT_ROOT)

    # --- Derive thresholds from the surrogate ---
    forecast_mae = load_forecast_mae_gbps(artifacts.forecast_eval_summary, alpha, K)
    band_cfg = threshold_cfg.get("hysteresis_band", "auto")
    if band_cfg == "auto":
        band_mae_used = forecast_mae
    elif isinstance(band_cfg, (int, float)):
        band_mae_used = (band_cfg / 1000.0) / 2.0  # we pass MAE; band = 2*MAE
    else:
        band_mae_used = None
    spec = derive_thresholds(twin, safety_margin_mbps=safety_margin_mbps,
                             forecast_mae_gbps=band_mae_used)

    print(f"\nDerived thresholds (from surrogate):")
    print(f"  Energy break-even:  {spec.energy_breakeven_gbps*1000:6.1f} Mbps")
    print(f"  QoS limit:          {spec.qos_limit_gbps*1000:6.1f} Mbps")
    print(f"  Decision threshold: {spec.decision_gbps*1000:6.1f} Mbps  ({spec.derived_from})")
    print(f"  Hysteresis band:    {spec.hysteresis_band_gbps*1000:6.1f} Mbps  "
          f"(t_up={spec.t_up_gbps*1000:.1f}, t_down={spec.t_down_gbps*1000:.1f})")
    print(f"  Cooldown steps:     {cooldown_steps}\n")

    # --- Build controllers ---
    policies = [
        StaticDPDKPolicy(),
        StaticUSRPolicy(),
        ThresholdPolicy(spec.decision_gbps),
        HysteresisPolicy(spec.t_up_gbps, spec.t_down_gbps, cooldown_steps),
        OracleThresholdPolicy(spec.decision_gbps),
    ]

    # --- Run rollouts (precompute twin once, share across policies) ---
    results_dir = PROJECT_ROOT / paths_cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Precomputing twin lookups (one-shot, shared across all policies)...")
    pre = precompute_twin_for_rollouts(twin, artifacts.targets_norm, alpha, horizon_idx=0)

    rollouts: dict = {}
    for policy in policies:
        print(f"Running rollout: {policy.name} ...")
        df = run_rollout(
            policy=policy,
            digital_twin=twin,
            predictions_norm=artifacts.predictions_norm,
            targets_norm=artifacts.targets_norm,
            alpha=alpha,
            horizon_idx=0,
            precomputed=pre,
        )
        rollouts[policy.name] = df
        path = results_dir / f"rollout_{policy.name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.relative_to(PROJECT_ROOT)}")

    # --- Metrics ---
    print("\nComputing metrics...")
    dpdk_ref   = rollouts["static_dpdk"]
    oracle_ref = rollouts["oracle_threshold"]

    all_metrics: dict = {"_thresholds": {
        "energy_breakeven_gbps": spec.energy_breakeven_gbps,
        "qos_limit_gbps":        spec.qos_limit_gbps,
        "decision_gbps":         spec.decision_gbps,
        "t_up_gbps":             spec.t_up_gbps,
        "t_down_gbps":           spec.t_down_gbps,
        "hysteresis_band_gbps":  spec.hysteresis_band_gbps,
        "cooldown_steps":        cooldown_steps,
        "derived_from":          spec.derived_from,
    }}
    for name, df in rollouts.items():
        m = compute_metrics(df, time_step_minutes, dpdk_ref, oracle_ref)
        m["policy"] = name
        all_metrics[name] = m

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"  Saved → {metrics_path.relative_to(PROJECT_ROOT)}\n")

    # --- Print summary table ---
    print(f"  {'Policy':<22} {'Energy(Wh)':>12} {'AvgPwr(W)':>12} {'Save%':>8} "
          f"{'UnsafeUSR':>10} {'USR%':>8} {'Flips':>8} {'Switch(Wh)':>12}")
    print("  " + "-" * 102)
    for name in ["static_dpdk", "static_usr", "threshold", "hysteresis", "oracle_threshold"]:
        m = all_metrics[name]
        save = m["energy_saving_vs_static_dpdk_pct"]
        save_str = f"{save:>7.1f}" if save is not None else f"{'—':>7}"
        print(
            f"  {name:<22} {m['total_energy_wh']:>12.2f} {m['average_power_w']:>12.4f} "
            f"{save_str} {m['unsafe_usr_rate']:>10.3f} "
            f"{m['usr_usage_ratio']*100:>7.1f}% {m['decision_flip_rate']:>8.3f} "
            f"{m['switch_energy_wh']:>12.4f}"
        )

    print("\n=== Demo complete ===\n")


if __name__ == "__main__":
    main()
