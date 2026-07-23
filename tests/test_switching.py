"""DigitalTwin.compute_step — switching-cost accounting (pure logic)."""

from __future__ import annotations

import pytest


def test_no_switch_returns_old_result_unchanged(make_result, bare_twin):
    twin = bare_twin(prewarm_enabled=False)
    old = make_result(upf="DPDK", power=1.0)
    composite, sw_wh, new_current, pending = twin.compute_step("DPDK", "DPDK", old, old)
    assert composite is old
    assert sw_wh == 0.0
    assert new_current == "DPDK"
    assert pending is False


def test_no_switch_charges_prewarm_standby_when_usr_serving(make_result, bare_twin):
    # prewarm keeps DPDK on standby while USR serves -> standby paid every step
    twin = bare_twin(prewarm_enabled=True, prewarm_standby_w=0.05, step_s=900.0)
    old = make_result(upf="USR", power=0.5)
    _, sw_wh, _, _ = twin.compute_step("USR", "USR", old, old)
    assert sw_wh == pytest.approx(0.05 * 0.25)  # standby_w * step_h (900s = 0.25h)


def test_sub_step_weighted_average_power_and_spike(make_result, bare_twin):
    # step_s=10, act_usr=4 -> frac_old = 0.4 (old serves 40% of the step)
    twin = bare_twin(accounting="sub_step", step_s=10.0, act_usr_s=4.0)
    old = make_result(upf="DPDK", power=1.0)
    new = make_result(upf="USR", power=0.5)
    composite, sw_wh, new_current, pending = twin.compute_step("DPDK", "USR", old, new)
    assert composite.power_watts == pytest.approx(0.4 * 1.0 + 0.6 * 0.5)  # 0.7
    assert sw_wh == pytest.approx(0.01)   # constant USR spike (load-independent)
    assert new_current == "USR"
    assert pending is False


def test_round_up_leaves_activation_pending(make_result, bare_twin):
    twin = bare_twin(accounting="round_up")
    old = make_result(upf="DPDK", power=1.0)
    new = make_result(upf="USR", power=0.5)
    composite, sw_wh, new_current, pending = twin.compute_step("DPDK", "USR", old, new)
    assert pending is True
    assert composite is old            # old UPF served the whole step
    assert new_current == "DPDK"        # switch did not complete
    assert sw_wh == 0.0                 # no spike paid yet


def test_round_down_is_instant_switch(make_result, bare_twin):
    twin = bare_twin(accounting="round_down")
    old = make_result(upf="DPDK", power=1.0)
    new = make_result(upf="USR", power=0.5)
    composite, sw_wh, new_current, pending = twin.compute_step("DPDK", "USR", old, new)
    assert composite.power_watts == pytest.approx(0.5)   # new UPF for the full step
    assert new_current == "USR"
    assert pending is False
    assert sw_wh == pytest.approx(0.01)                  # constant USR spike


def test_switch_safety_is_conservative(make_result, bare_twin):
    # If either UPF is unsafe at this load during the switch, composite is unsafe.
    twin = bare_twin(accounting="round_down")
    old = make_result(upf="DPDK", power=1.0, is_safe=True, loss=0.0)
    new = make_result(upf="USR", power=0.5, is_safe=False, loss=3.0)
    composite, *_ = twin.compute_step("DPDK", "USR", old, new)
    assert composite.is_safe is False
    assert composite.predicted_loss == pytest.approx(3.0)  # max(old, new)
    assert composite.is_efficient is False                 # not meaningful mid-switch
