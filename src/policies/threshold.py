"""Single-threshold policy — stateless.

Switches based on predicted load: USR if predicted < threshold else DPDK.
No anti-flapping. For hysteresis + cooldown use HysteresisPolicy instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Controller


class ThresholdPolicy(Controller):
    name = "threshold"

    def __init__(self, decision_gbps: float) -> None:
        self.decision_gbps = decision_gbps

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> str:
        return "USR" if observation["predicted_load_gbps"] < self.decision_gbps else "DPDK"

    def required_obs_keys(self) -> tuple[str, ...]:
        return ("predicted_load_gbps",)

    def act_array_stateless(self, predicted_load_gbps: np.ndarray) -> np.ndarray:
        return np.where(predicted_load_gbps < self.decision_gbps, "USR", "DPDK").astype(object)
