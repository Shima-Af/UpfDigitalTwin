"""End-to-end twin checks that require the real surrogate models.

Skipped automatically when the DVC-tracked data is not present (e.g. CI without
credentials). Run `python scripts/fetch_data.py` first to enable them locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = PROJECT_ROOT / "data/external/profiling_twin/models/manifest.json"

pytestmark = pytest.mark.skipif(
    not _MANIFEST.exists(),
    reason="surrogate models absent — run `python scripts/fetch_data.py`",
)


def _twin():
    from upf_digital_twin.twin.digital_twin import DigitalTwin
    from upf_digital_twin.utils.config import load_configs

    paths_cfg, scenario_cfg = load_configs(PROJECT_ROOT)
    return DigitalTwin(scenario_cfg, paths_cfg, PROJECT_ROOT)


def test_evaluate_action_returns_physical_values():
    twin = _twin()
    r = twin.evaluate_action("DPDK", 0.05)
    assert r.upf_type == "DPDK"
    assert r.power_watts > 0.0
    assert r.delay_us >= 0.0


def test_dpdk_and_usr_efficiency_flags_are_complementary():
    twin = _twin()
    load = 0.05
    d = twin.evaluate_action("DPDK", load)
    u = twin.evaluate_action("USR", load)
    # is_efficient means "cheaper than the alternative" — they can't both be True
    assert not (d.is_efficient and u.is_efficient)
