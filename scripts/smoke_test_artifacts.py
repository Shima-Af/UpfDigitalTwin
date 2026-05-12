#!/usr/bin/env python3
"""Smoke test — verifies all required artifacts and configs are present and valid.

Run from the project root:
    python scripts/smoke_test_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from upf_digital_twin.utils.config import load_configs, resolve_path

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> int:
    print("\n=== UPF Digital Twin — Smoke Test ===\n")
    ok = True

    # --- 1. Config loading ---
    print("[ Configs ]")
    try:
        paths_cfg, scenario_cfg = load_configs(PROJECT_ROOT)
        ok &= check("paths.yaml loads", True)
        ok &= check("scenario.yaml loads", True)
    except Exception as exc:
        ok &= check("configs load", False, str(exc))
        print("\nCannot continue without configs.")
        return 1

    # --- 2. Scenario fields ---
    print("\n[ Scenario fields ]")
    traffic_cfg   = scenario_cfg.get("traffic", {})
    upf_cfg       = scenario_cfg.get("upf", {})
    threshold_cfg = scenario_cfg.get("threshold", {})
    switching_cfg = scenario_cfg.get("upf_switching", {})
    qos_cfg       = upf_cfg.get("qos_budget", {})

    alpha = traffic_cfg.get("calibration", {}).get("alpha_gbps_per_norm")
    ok &= check("alpha_gbps_per_norm present", alpha is not None,
                "missing in scenario.yaml traffic.calibration")
    ok &= check("selected_k present", "selected_k" in traffic_cfg)
    ok &= check("upf.qos_budget.delay_budget_us present", "delay_budget_us" in qos_cfg)
    ok &= check("upf.qos_budget.max_loss_pkts_per_interval present", "max_loss_pkts_per_interval" in qos_cfg)
    ok &= check("threshold.safety_margin_mbps present", "safety_margin_mbps" in threshold_cfg)
    ok &= check("threshold.cooldown_steps present", "cooldown_steps" in threshold_cfg)
    ok &= check("upf_switching.accounting present",
                switching_cfg.get("accounting") in ("sub_step", "round_down", "round_up"),
                f"got {switching_cfg.get('accounting')!r}, expected one of sub_step|round_down|round_up")
    sc_path = paths_cfg.get("profiling_twin", {}).get("switching_costs")
    sc_full = (PROJECT_ROOT / sc_path) if sc_path else None
    ok &= check("switching_costs.yaml exists",
                sc_full is not None and sc_full.exists(),
                str(sc_full) if sc_full else "path missing in paths.yaml")

    # --- 3. Traffic artifact files ---
    print("\n[ Traffic forecaster artifacts ]")
    import numpy as np
    import pandas as pd

    tf = paths_cfg["traffic_forecaster"]
    selected_k = int(traffic_cfg.get("selected_k", -1))

    required_files = [
        "predictions", "targets", "cluster_series",
        "cluster_assignments", "cluster_bs_map", "forecast_eval_summary",
    ]
    for key in required_files:
        p = resolve_path(PROJECT_ROOT, tf[key])
        ok &= check(f"{p.name} exists", p.exists(), str(p))

    # Shape validation
    pred_path = resolve_path(PROJECT_ROOT, tf["predictions"])
    tgt_path = resolve_path(PROJECT_ROOT, tf["targets"])
    if pred_path.exists() and tgt_path.exists():
        pred = np.load(pred_path)
        tgt = np.load(tgt_path)
        ok &= check(f"predictions/targets same shape ({pred.shape})", pred.shape == tgt.shape)
        ok &= check("arrays are 3-D (N, H, K)", pred.ndim == 3, f"got ndim={pred.ndim}")
        if pred.ndim == 3:
            _, _, k = pred.shape
            ok &= check(
                f"selected_k={selected_k} matches array K={k}",
                k == selected_k,
                f"scenario says {selected_k}, arrays have {k}",
            )
        ok &= check("no NaN/inf in predictions", bool(np.isfinite(pred).all()))
        ok &= check("no NaN/inf in targets", bool(np.isfinite(tgt).all()))

    # --- 4. Profiling twin directory ---
    print("\n[ Profiling twin artifacts ]")
    pt = paths_cfg["profiling_twin"]
    pt_dir = resolve_path(PROJECT_ROOT, pt["dir"])
    ok &= check("profiling_twin dir exists", pt_dir.exists(), str(pt_dir))
    for key in ("params", "manifest"):
        p = resolve_path(PROJECT_ROOT, pt[key])
        ok &= check(f"{p.name} exists", p.exists(), str(p))

    # --- Summary ---
    print()
    if ok:
        print("=== All checks passed. Ready to run threshold demo. ===\n")
        return 0
    else:
        print("=== Some checks FAILED. Fix the issues above before running the demo. ===\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
