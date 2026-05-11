"""Oracle threshold policy — uses actual load instead of predicted.

Same threshold rule as ThresholdPolicy but cheats by reading the realised
load from the observation. Not deployable — exists only as an upper bound
on what a perfect forecast would achieve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Controller


class OracleThresholdPolicy(Controller):
    name = "oracle_threshold"

    def __init__(self, decision_gbps: float) -> None:
        self.decision_gbps = decision_gbps

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> str:
        return "USR" if observation["actual_load_gbps"] < self.decision_gbps else "DPDK"

    def required_obs_keys(self) -> tuple[str, ...]:
        return ("actual_load_gbps",)

    def act_array_stateless(self, actual_load_gbps: np.ndarray) -> np.ndarray:
        return np.where(actual_load_gbps < self.decision_gbps, "USR", "DPDK").astype(object)
