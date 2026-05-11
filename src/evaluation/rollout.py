"""Rollout engine.

Strategy: pre-evaluate the twin once for both UPFs over the entire flat
array of actual loads (vectorised, fast), then do the per-cluster
stateful loop in pure Python — the loop only does dict lookups, calls
DigitalTwin.compute_step (pure function, no model calls), and writes one
row per step.

This makes stateful policies (hysteresis) and switching-cost accounting
cheap even though the rollout is conceptually sequential.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.policies.base import Controller
from src.twin.digital_twin import DigitalTwin
from src.twin.upf_profile import UPFResult


def _is_oracle(policy: Controller) -> bool:
    return "actual_load_gbps" in policy.required_obs_keys()


def _precompute_twin(
    digital_twin: DigitalTwin,
    actual_gbps_2d: np.ndarray,   # (N, K)
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-evaluate the twin for DPDK and USR over a flat (N*K,) load array."""
    flat = actual_gbps_2d.ravel().astype(np.float64)
    d = digital_twin.evaluate_batch("DPDK", flat)
    u = digital_twin.evaluate_batch("USR", flat, alt_power_watts=d["power_watts"])
    d["is_efficient"] = d["power_watts"] < u["power_watts"]
    return {"DPDK": d, "USR": u}


def _result_at(lookup: dict[str, np.ndarray], idx: int, upf_type: str, load_gbps: float) -> UPFResult:
    """Build a UPFResult from a precomputed lookup row."""
    return UPFResult(
        upf_type=upf_type,
        load_gbps=load_gbps,
        power_watts=float(lookup["power_watts"][idx]),
        delay_us=float(lookup["delay_us"][idx]),
        throughput_gbps=float(lookup["throughput_gbps"][idx]),
        predicted_loss=float(lookup["predicted_loss"][idx]),
        is_safe=bool(lookup["is_safe"][idx]),
        is_efficient=bool(lookup["is_efficient"][idx]),
    )


def run_rollout(
    policy: Controller,
    digital_twin: DigitalTwin,
    predictions_norm: np.ndarray,
    targets_norm: np.ndarray,
    alpha: float,
    horizon_idx: int = 0,
    precomputed: dict | None = None,
) -> pd.DataFrame:
    """Run a controller over all clusters and time windows."""
    if predictions_norm.shape != targets_norm.shape:
        raise ValueError(
            f"shapes differ: predictions {predictions_norm.shape} vs "
            f"targets {targets_norm.shape}"
        )
    N, H, K = predictions_norm.shape
    if not (0 <= horizon_idx < H):
        raise ValueError(f"horizon_idx {horizon_idx} out of range [0, {H})")

    pred_gbps   = predictions_norm[:, horizon_idx, :] * alpha    # (N, K)
    actual_gbps = targets_norm[:, horizon_idx, :]   * alpha      # (N, K)
    is_oracle = _is_oracle(policy)

    pre = precomputed if precomputed is not None else _precompute_twin(digital_twin, actual_gbps)

    rows: list[dict] = []
    for k in range(K):
        policy.reset()
        current_action = "DPDK"
        decision_arr = actual_gbps[:, k] if is_oracle else pred_gbps[:, k]

        for n in range(N):
            obs = {"predicted_load_gbps": float(decision_arr[n]),
                   "actual_load_gbps":    float(actual_gbps[n, k])}
            requested = policy.act(obs)
            load = float(actual_gbps[n, k])
            flat_idx = n * K + k

            old_r = _result_at(pre[current_action], flat_idx, current_action, load)
            new_r = (_result_at(pre[requested], flat_idx, requested, load)
                     if requested != current_action else old_r)

            composite, sw_energy, new_current, pending = digital_twin.compute_step(
                current_action, requested, old_r, new_r,
            )
            realised = current_action if pending else requested
            if not pending:
                current_action = new_current

            rows.append({
                "sample_idx":          n,
                "cluster_idx":         k,
                "horizon_idx":         horizon_idx,
                "predicted_load_gbps": float(pred_gbps[n, k]),
                "actual_load_gbps":    load,
                "requested_upf":       requested,
                "selected_upf":        realised,
                "activation_pending":  pending,
                "power_watts":         composite.power_watts,
                "switching_energy_wh": sw_energy,
                "delay_us":            composite.delay_us,
                "predicted_loss":      composite.predicted_loss,
                "is_safe":             composite.is_safe,
                "is_efficient":        composite.is_efficient,
            })

    return pd.DataFrame(rows)


def precompute_twin_for_rollouts(
    digital_twin: DigitalTwin,
    targets_norm: np.ndarray,
    alpha: float,
    horizon_idx: int = 0,
) -> dict:
    actual_gbps = targets_norm[:, horizon_idx, :] * alpha
    return _precompute_twin(digital_twin, actual_gbps)
