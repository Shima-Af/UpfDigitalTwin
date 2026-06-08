"""Threshold derivation from a stubbed surrogate (no real models loaded)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from upf_digital_twin.twin.threshold_derivation import (
    derive_thresholds,
    load_forecast_mae_gbps,
)


class _StubProfile:
    """Synthetic UPF curves with known break-even / QoS / delay crossings.

    DPDK: flat 1.0 W, always safe.
    USR:  power = 10*load  (crosses DPDK at load=0.1 Gbps),
          QoS-safe for load <= 0.08, delay within budget for load <= 0.12.
    """

    _delay_budget_us = 200.0

    def evaluate_batch(self, upf, loads, alt_power_watts=None):
        loads = np.asarray(loads, dtype=float)
        if upf == "DPDK":
            return {
                "power_watts": np.full_like(loads, 1.0),
                "is_safe": np.ones_like(loads, dtype=bool),
                "delay_us": np.full_like(loads, 50.0),
            }
        return {
            "power_watts": loads * 10.0,
            "is_safe": loads <= 0.08,
            "delay_us": np.where(loads <= 0.12, 50.0, 300.0),
        }


def _stub_twin():
    return types.SimpleNamespace(_profile=_StubProfile())


def test_derives_breakeven_qos_and_delay_limits():
    spec = derive_thresholds(_stub_twin(), safety_margin_mbps=10.0, forecast_mae_gbps=None)
    assert spec.energy_breakeven_gbps == pytest.approx(0.10, abs=0.0015)
    assert spec.qos_limit_gbps == pytest.approx(0.08, abs=0.0015)
    assert spec.delay_limit_gbps == pytest.approx(0.12, abs=0.0015)


def test_decision_is_limiting_minus_margin():
    spec = derive_thresholds(_stub_twin(), safety_margin_mbps=10.0, forecast_mae_gbps=None)
    # min(breakeven=0.10, qos=0.08) - 0.010 margin = 0.070
    assert spec.decision_gbps == pytest.approx(0.07, abs=0.0015)


def test_hysteresis_band_is_twice_forecast_mae():
    spec = derive_thresholds(_stub_twin(), safety_margin_mbps=10.0, forecast_mae_gbps=0.005)
    assert spec.hysteresis_band_gbps == pytest.approx(0.010)
    assert spec.t_up_gbps == pytest.approx(spec.decision_gbps)
    assert spec.t_down_gbps == pytest.approx(spec.decision_gbps - 0.010, abs=1e-9)


def test_no_mae_means_no_band():
    spec = derive_thresholds(_stub_twin(), safety_margin_mbps=10.0, forecast_mae_gbps=None)
    assert spec.hysteresis_band_gbps == 0.0
    assert spec.t_down_gbps == pytest.approx(spec.t_up_gbps)


def test_load_forecast_mae_scales_by_alpha():
    summary = {"results": [{"K": 5, "test_mae": 0.16}, {"K": 10, "test_mae": 0.10}]}
    assert load_forecast_mae_gbps(summary, alpha=0.12, K=10) == pytest.approx(0.012)


def test_load_forecast_mae_missing_k_returns_none():
    summary = {"results": [{"K": 5, "test_mae": 0.16}]}
    assert load_forecast_mae_gbps(summary, alpha=0.12, K=10) is None
