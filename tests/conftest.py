"""Shared fixtures and factory helpers for the test suite.

All helpers here are hermetic — they build objects in-memory and never read the
48 MB DVC-tracked dataset, so the bulk of the suite runs in CI without data.
"""

from __future__ import annotations

import pytest

from upf_digital_twin.twin.digital_twin import DigitalTwin
from upf_digital_twin.twin.upf_profile import UPFResult


@pytest.fixture
def make_result():
    """Factory for UPFResult objects with sensible defaults."""

    def _make(
        upf: str = "USR",
        power: float = 0.5,
        delay: float = 50.0,
        throughput: float = 0.1,
        loss: float = 0.0,
        is_safe: bool = True,
        is_efficient: bool = False,
        load: float = 0.1,
    ) -> UPFResult:
        return UPFResult(
            upf_type=upf,
            load_gbps=load,
            power_watts=power,
            delay_us=delay,
            throughput_gbps=throughput,
            predicted_loss=loss,
            is_safe=is_safe,
            is_efficient=is_efficient,
        )

    return _make


@pytest.fixture
def bare_twin():
    """Factory for a DigitalTwin with switching params set, but NO models loaded.

    Bypasses __init__ (which would read manifest.json + switching_costs.yaml) so
    DigitalTwin.compute_step — pure switching/accounting logic — can be tested in
    isolation.
    """

    def _make(
        accounting: str = "sub_step",
        prewarm_enabled: bool = False,
        prewarm_standby_w: float = 0.05,
        act_dpdk_s: float = 2.0,
        act_usr_s: float = 1.0,
        step_s: float = 900.0,
        spike_dpdk_wh: float = 0.02,
        spike_usr_wh: float = 0.01,
    ) -> DigitalTwin:
        twin = object.__new__(DigitalTwin)
        twin.accounting = accounting
        twin.prewarm_enabled = prewarm_enabled
        twin.prewarm_standby_w = prewarm_standby_w
        twin.activation_duration_s = {"DPDK": act_dpdk_s, "USR": act_usr_s}
        # Per-variant switching energy is now a load-independent constant.
        twin.switch_spike_wh = {"DPDK": spike_dpdk_wh, "USR": spike_usr_wh}
        twin.step_s = step_s
        twin.step_h = step_s / 3600.0
        return twin

    return _make
