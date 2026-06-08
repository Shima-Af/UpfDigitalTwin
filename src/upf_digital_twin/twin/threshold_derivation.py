"""Derive operating thresholds from the surrogate models.

Replaces hardcoded thresholds with values computed from the two-layer
surrogate at twin-load time.  The derivation answers two questions:

  T_qos_max:     highest USR load at which QoS is still preserved
                 (no predicted packet loss, delay within budget).
  T_breakeven:   load at which USR power equals DPDK power.
                 Below this, USR is more efficient. Above, DPDK is.

The decision threshold a controller should use is the *minimum* of these
two, optionally minus a safety margin.

Anti-flapping (hysteresis band) is derived from forecast accuracy:
  hysteresis_band = 2 * forecast_MAE
This guarantees a single noisy forecast cannot trigger a flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdSpec:
    """All operating points derived from the surrogate models."""

    # Physics, derived once
    energy_breakeven_gbps: float    # USR power == DPDK power
    qos_limit_gbps:        float    # last load where USR is still QoS-OK
    delay_limit_gbps:      float    # last load where USR delay <= budget

    # Decision points used by controllers
    decision_gbps:         float    # min(breakeven, qos) - safety_margin
    hysteresis_band_gbps:  float    # 2 * forecast_MAE
    t_up_gbps:             float    # USR -> DPDK switch
    t_down_gbps:           float    # DPDK -> USR switch (= t_up - band)

    # Provenance
    safety_margin_mbps:    float
    forecast_mae_gbps:     float | None
    derived_from:          str     # short human-readable explanation


def derive_thresholds(
    twin,                            # DigitalTwin instance
    safety_margin_mbps: float = 10.0,
    forecast_mae_gbps:  float | None = None,
    sweep_max_gbps:     float = 0.5,
    sweep_resolution_mbps: float = 1.0,
) -> ThresholdSpec:
    """Compute decision thresholds from the surrogate models.

    Args:
        twin:                  DigitalTwin (must expose ._profile).
        safety_margin_mbps:    Subtract this from the limiting threshold to
                               keep a buffer from the QoS / breakeven cliff.
        forecast_mae_gbps:     Forecast error magnitude. If provided, the
                               hysteresis band = 2 * MAE. If None, band = 0
                               (pure threshold, no anti-flapping).
        sweep_max_gbps:        Upper bound of load sweep used for derivation.
        sweep_resolution_mbps: Granularity of the sweep.

    Returns:
        ThresholdSpec with all derived values + provenance.
    """
    profile = twin._profile  # noqa: SLF001

    n_pts = int(sweep_max_gbps * 1000 / sweep_resolution_mbps)
    loads = np.linspace(sweep_resolution_mbps / 1000, sweep_max_gbps, n_pts)

    dpdk = profile.evaluate_batch("DPDK", loads)
    usr  = profile.evaluate_batch("USR",  loads, alt_power_watts=dpdk["power_watts"])

    # --- Energy break-even: first load where USR power >= DPDK power ---
    usr_above_dpdk = usr["power_watts"] >= dpdk["power_watts"]
    if usr_above_dpdk.any():
        breakeven_idx = int(np.argmax(usr_above_dpdk))
        breakeven = float(loads[breakeven_idx])
    else:
        breakeven = float(sweep_max_gbps)

    # --- QoS limit: last load where USR is_safe ---
    usr_safe = usr["is_safe"]
    if usr_safe.all():
        qos_limit = float(sweep_max_gbps)
    elif not usr_safe.any():
        qos_limit = 0.0
    else:
        # last True before the first False
        first_unsafe = int(np.argmax(~usr_safe))
        qos_limit = float(loads[first_unsafe - 1]) if first_unsafe > 0 else 0.0

    # --- Delay-only limit (informational) ---
    delay_us = usr["delay_us"]
    delay_budget = profile._delay_budget_us  # noqa: SLF001
    delay_ok = delay_us <= delay_budget
    if delay_ok.all():
        delay_limit = float(sweep_max_gbps)
    elif not delay_ok.any():
        delay_limit = 0.0
    else:
        first_bad = int(np.argmax(~delay_ok))
        delay_limit = float(loads[first_bad - 1]) if first_bad > 0 else 0.0

    # --- Decision threshold ---
    safety_margin_gbps = safety_margin_mbps / 1000.0
    limiting = min(breakeven, qos_limit)
    decision = max(0.0, limiting - safety_margin_gbps)

    # --- Hysteresis band ---
    band = 2 * forecast_mae_gbps if forecast_mae_gbps is not None else 0.0
    t_up   = decision
    t_down = max(0.0, decision - band)

    derived_from = (
        f"min(breakeven={breakeven*1000:.1f} Mbps, qos={qos_limit*1000:.1f} Mbps) "
        f"- {safety_margin_mbps:.0f} Mbps margin"
    )

    return ThresholdSpec(
        energy_breakeven_gbps=breakeven,
        qos_limit_gbps=qos_limit,
        delay_limit_gbps=delay_limit,
        decision_gbps=decision,
        hysteresis_band_gbps=band,
        t_up_gbps=t_up,
        t_down_gbps=t_down,
        safety_margin_mbps=safety_margin_mbps,
        forecast_mae_gbps=forecast_mae_gbps,
        derived_from=derived_from,
    )


def load_forecast_mae_gbps(forecast_eval_summary: dict, alpha: float, K: int) -> float | None:
    """Extract the MAE for the configured K from the forecaster's summary.

    The summary is a list of {'K': k, 'test_mae': mae} entries.  MAE is in
    normalised units, so we multiply by alpha to get Gbps.
    """
    for row in forecast_eval_summary.get("results", []):
        if int(row.get("K", -1)) == int(K):
            mae_norm = float(row.get("test_mae", 0.0))
            return mae_norm * alpha
    return None
