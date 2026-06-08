"""Controller (policy) behaviour."""

from __future__ import annotations

import numpy as np

from upf_digital_twin.policies.static import StaticDPDKPolicy, StaticUSRPolicy
from upf_digital_twin.policies.threshold import ThresholdPolicy
from upf_digital_twin.policies.oracle import OracleThresholdPolicy
from upf_digital_twin.policies.hysteresis import HysteresisPolicy


def test_static_policies_are_constant():
    assert StaticDPDKPolicy().act({}) == "DPDK"
    assert StaticUSRPolicy().act({}) == "USR"


def test_threshold_switches_on_predicted_load():
    p = ThresholdPolicy(decision_gbps=0.07)
    assert p.act({"predicted_load_gbps": 0.05}) == "USR"   # below -> USR
    assert p.act({"predicted_load_gbps": 0.07}) == "DPDK"  # equal -> DPDK (strict <)
    assert p.act({"predicted_load_gbps": 0.10}) == "DPDK"  # above -> DPDK
    assert p.required_obs_keys() == ("predicted_load_gbps",)


def test_threshold_vectorised_matches_scalar():
    p = ThresholdPolicy(decision_gbps=0.07)
    loads = np.array([0.01, 0.07, 0.2])
    out = p.act_array_stateless(loads)
    assert list(out) == ["USR", "DPDK", "DPDK"]


def test_oracle_reads_actual_load_not_predicted():
    p = OracleThresholdPolicy(decision_gbps=0.07)
    # predicted says DPDK, actual says USR — oracle must follow actual
    obs = {"predicted_load_gbps": 0.20, "actual_load_gbps": 0.01}
    assert p.act(obs) == "USR"
    assert "actual_load_gbps" in p.required_obs_keys()


def test_hysteresis_rejects_inverted_band():
    import pytest

    with pytest.raises(ValueError):
        HysteresisPolicy(t_up_gbps=0.05, t_down_gbps=0.07)


def test_hysteresis_holds_through_band_no_flapping():
    # band is [t_down=0.04, t_up=0.07]; cooldown 0 to isolate the band logic
    p = HysteresisPolicy(t_up_gbps=0.07, t_down_gbps=0.04, cooldown_steps=0)
    p.reset()
    # Start low -> USR (first call uses band midpoint 0.055)
    assert p.act({"predicted_load_gbps": 0.01}) == "USR"
    # Inside the band: must hold USR, not flip
    assert p.act({"predicted_load_gbps": 0.06}) == "USR"
    # Cross t_up: flip to DPDK
    assert p.act({"predicted_load_gbps": 0.08}) == "DPDK"
    # Back inside band: hold DPDK
    assert p.act({"predicted_load_gbps": 0.05}) == "DPDK"
    # Below t_down: flip back to USR
    assert p.act({"predicted_load_gbps": 0.03}) == "USR"


def test_hysteresis_cooldown_blocks_immediate_reflip():
    p = HysteresisPolicy(t_up_gbps=0.07, t_down_gbps=0.04, cooldown_steps=2)
    p.reset()
    p.act({"predicted_load_gbps": 0.01})        # -> USR
    assert p.act({"predicted_load_gbps": 0.08}) == "DPDK"  # flip, starts cooldown=2
    # Even though load drops below t_down, cooldown forces holding DPDK
    assert p.act({"predicted_load_gbps": 0.01}) == "DPDK"
    assert p.act({"predicted_load_gbps": 0.01}) == "DPDK"
    # Cooldown elapsed -> may switch again
    assert p.act({"predicted_load_gbps": 0.01}) == "USR"


def test_hysteresis_reset_clears_state():
    p = HysteresisPolicy(t_up_gbps=0.07, t_down_gbps=0.04, cooldown_steps=5)
    p.act({"predicted_load_gbps": 0.08})
    p.reset()
    assert p._last_action is None
    assert p._cooldown_left == 0
