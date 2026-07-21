#!/usr/bin/env python3
"""Generate the data-driven figures for the Digital Twin thesis chapter.

Run from the project root with the project venv:
    .venv/bin/python reports/make_figures.py

All figures are written to reports/figures/ as 300-dpi PNGs.  Diagrams that
must be drawn by hand (architecture, switching-accounting timeline) are NOT
produced here — see the placeholder descriptions in the chapter.

The only controllers exercised here are the non-learning ones (static DPDK,
static USR, hysteresis); learned controllers are deferred to a later chapter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upf_digital_twin.utils.config import load_configs
from upf_digital_twin.data.traffic_loader import load_traffic_artifacts
from upf_digital_twin.twin.digital_twin import DigitalTwin
from upf_digital_twin.twin.threshold_derivation import derive_thresholds, load_forecast_mae_gbps

# ---- Consistent style (match profiling chapter: DPDK blue, USR red) --------
DPDK_C = "#1f4e79"
USR_C = "#c0392b"
ACC_C = "#117733"
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    paths_cfg, scenario_cfg = load_configs(PROJECT_ROOT)
    alpha = scenario_cfg["traffic"]["calibration"]["alpha_gbps_per_norm"]
    twin = DigitalTwin(scenario_cfg, paths_cfg, PROJECT_ROOT)
    profile = twin._profile
    artifacts = load_traffic_artifacts(paths_cfg, scenario_cfg, PROJECT_ROOT)

    # Derived operating points (twin physics, not a controller)
    mae = load_forecast_mae_gbps(artifacts.forecast_eval_summary, alpha, artifacts.selected_k)
    spec = derive_thresholds(twin, safety_margin_mbps=10.0, forecast_mae_gbps=mae)
    print("Derived: breakeven=%.1f qos=%.1f decision=%.1f t_up=%.1f t_down=%.1f Mbps"
          % (spec.energy_breakeven_gbps * 1e3, spec.qos_limit_gbps * 1e3,
             spec.decision_gbps * 1e3, spec.t_up_gbps * 1e3, spec.t_down_gbps * 1e3))

    # =====================================================================
    # Figure 1 — Surrogate response curves (the twin's predicted physics)
    # =====================================================================
    loads = np.linspace(1e-3, 0.5, 400)
    d = profile.evaluate_batch("DPDK", loads)
    u = profile.evaluate_batch("USR", loads, alt_power_watts=d["power_watts"])

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    (ax_p, ax_c), (ax_d, ax_l) = axes

    ax_p.plot(loads, d["power_watts"], color=DPDK_C, label="DPDK")
    ax_p.plot(loads, u["power_watts"], color=USR_C, label="USR (usr\\_full)")
    ax_p.set_ylabel("Power (W)")
    ax_p.set_title("(a) Power")
    ax_p.legend(loc="upper left")

    ax_c.plot(loads, d["cpu_pct"], color=DPDK_C)
    ax_c.plot(loads, u["cpu_pct"], color=USR_C)
    ax_c.set_ylabel("CPU (%)")
    ax_c.set_title("(b) CPU utilisation")

    ax_d.plot(loads, d["delay_us"], color=DPDK_C)
    ax_d.plot(loads, u["delay_us"], color=USR_C)
    ax_d.axhline(profile._delay_budget_us, ls="--", color="k", lw=1,
                 label="QoS budget (200 µs)")
    ax_d.set_ylabel("Downlink delay (µs)")
    ax_d.set_xlabel("Offered load (Gbps)")
    ax_d.set_title("(c) Downlink delay")
    ax_d.set_yscale("log")
    ax_d.legend(loc="lower right")

    ax_l.plot(loads, d["predicted_loss"], color=DPDK_C)
    ax_l.plot(loads, u["predicted_loss"], color=USR_C)
    ax_l.set_ylabel("Predicted packet loss (pkts/interval)")
    ax_l.set_xlabel("Offered load (Gbps)")
    ax_l.set_title("(d) Packet loss")
    ax_l.set_yscale("symlog")
    for ax in (ax_p, ax_c, ax_d, ax_l):
        ax.set_xlim(0, 0.5)
    _save(fig, "fig_dt_surrogate_curves.png")

    # =====================================================================
    # Figure 2 — Threshold derivation (break-even + QoS limit + band)
    # =====================================================================
    z = np.linspace(1e-3, 0.15, 400)
    dz = profile.evaluate_batch("DPDK", z)
    uz = profile.evaluate_batch("USR", z, alt_power_watts=dz["power_watts"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(z, dz["power_watts"], color=DPDK_C, label="DPDK power")
    ax.plot(z, uz["power_watts"], color=USR_C, label="USR power")

    # Safe region (USR is_safe AND cheaper than DPDK)
    safe = uz["is_safe"]
    ax.fill_between(z, 0, ax.get_ylim()[1], where=safe & (uz["power_watts"] < dz["power_watts"]),
                    color=ACC_C, alpha=0.10, label="USR safe & efficient")

    for val, lab, c, ls in [
        (spec.energy_breakeven_gbps, "break-even", "#7030a0", "-."),
        (spec.qos_limit_gbps, "QoS limit", "#d35400", ":"),
        (spec.decision_gbps, "decision ($t_{up}$)", "k", "--"),
        (spec.t_down_gbps, "$t_{down}$", "gray", "--"),
    ]:
        ax.axvline(val, color=c, ls=ls, lw=1.3,
                   label=f"{lab} = {val*1e3:.0f} Mbps")
    ax.axvspan(spec.t_down_gbps, spec.t_up_gbps, color="gray", alpha=0.12)
    ax.set_xlabel("Offered load (Gbps)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Operating thresholds derived from the surrogate")
    ax.set_xlim(0, 0.15)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    _save(fig, "fig_dt_threshold_derivation.png")

    # =====================================================================
    # Figure 3 — Offered-load distribution vs the derived threshold
    # =====================================================================
    actual = artifacts.targets_norm[:, 0, :].ravel() * alpha  # Gbps
    below = float((actual < spec.decision_gbps).mean()) * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(actual, bins=80, range=(0, 0.5), color="#888888", alpha=0.8)
    ax.axvline(spec.decision_gbps, color="k", ls="--", lw=1.4,
               label=f"decision = {spec.decision_gbps*1e3:.0f} Mbps "
                     f"({below:.0f}% of steps below)")
    ax.set_xlabel("Offered load (Gbps)")
    ax.set_ylabel("Count (cluster-steps)")
    ax.set_title("Distribution of offered load across clusters and time")
    ax.set_xlim(0, 0.5)
    ax.legend(loc="upper right")
    _save(fig, "fig_dt_load_distribution.png")

    # =====================================================================
    # Figure 4 — Hysteresis rollout time series (representative cluster)
    # =====================================================================
    hyst = pd.read_parquet(PROJECT_ROOT / "results/threshold_demo/rollout_hysteresis.parquet")
    # Pick the cluster with the most switches to show interesting behaviour.
    flips = {}
    for k, g in hyst.groupby("cluster_idx"):
        a = g.sort_values("sample_idx")["selected_upf"].values
        flips[k] = int((a[1:] != a[:-1]).sum())
    k_show = max(flips, key=flips.get)
    g = hyst[hyst.cluster_idx == k_show].sort_values("sample_idx").reset_index(drop=True)
    # A readable contiguous window around switching activity
    sw_idx = np.where(g["selected_upf"].values[1:] != g["selected_upf"].values[:-1])[0]
    centre = int(sw_idx[len(sw_idx) // 2]) if len(sw_idx) else 0
    lo = max(0, centre - 140)
    hi = min(len(g), lo + 300)
    w = g.iloc[lo:hi]
    x = w["sample_idx"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(x, w["actual_load_gbps"], color="#333333", lw=1.0, label="actual load")
    ax1.plot(x, w["predicted_load_gbps"], color="#999999", lw=0.8, ls=":", label="predicted load")
    ax1.axhline(spec.t_up_gbps, color="k", ls="--", lw=1, label="$t_{up}$")
    ax1.axhline(spec.t_down_gbps, color="gray", ls="--", lw=1, label="$t_{down}$")
    # Shade USR-selected steps
    usr = (w["selected_upf"] == "USR").values
    ax1.fill_between(x, 0, ax1.get_ylim()[1], where=usr, color=USR_C, alpha=0.12,
                     step="mid", label="USR selected")
    ax1.set_ylabel("Load (Gbps)")
    ax1.set_title(f"Hysteresis controller driving the twin — cluster {k_show}")
    ax1.legend(loc="upper right", fontsize=8, ncol=3)

    ax2.plot(x, w["power_watts"], color=ACC_C, lw=1.0)
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Time step (15 min each)")
    _save(fig, "fig_dt_hysteresis_rollout.png")

    # =====================================================================
    # Figure 5 — Aggregate comparison (non-learning controllers only)
    # =====================================================================
    import json
    metrics = json.loads((PROJECT_ROOT / "results/threshold_demo/metrics.json").read_text())
    order = ["static_dpdk", "static_usr", "hysteresis"]
    labels = ["Static DPDK", "Static USR", "Hysteresis"]
    energy = [metrics[p]["total_energy_wh"] for p in order]
    saving = [metrics[p]["energy_saving_vs_static_dpdk_pct"] for p in order]
    unsafe = [metrics[p]["unsafe_usr_rate"] * 100 for p in order]
    usr_use = [metrics[p]["usr_usage_ratio"] * 100 for p in order]
    colors = [DPDK_C, USR_C, ACC_C]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    a0, a1, a2 = axes
    a0.bar(labels, energy, color=colors)
    a0.set_ylabel("Total energy (Wh)")
    a0.set_title("(a) Energy")
    for i, v in enumerate(saving):
        a0.text(i, energy[i], f"{v:.0f}% saved" if v else "ref", ha="center", va="bottom", fontsize=8)
    a1.bar(labels, unsafe, color=colors)
    a1.set_ylabel("Unsafe USR steps (%)")
    a1.set_title("(b) QoS safety")
    a2.bar(labels, usr_use, color=colors)
    a2.set_ylabel("USR usage (%)")
    a2.set_title("(c) USR usage")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    _save(fig, "fig_dt_policy_comparison.png")

    print("All figures written to", FIG_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
