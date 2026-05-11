"""Hysteresis threshold policy — STATEFUL.

Two thresholds + cooldown to prevent rapid flipping:
  USR -> DPDK  when predicted load >= t_up_gbps
  DPDK -> USR  when predicted load <= t_down_gbps
  After any switch: must wait `cooldown_steps` before another switch.

Initial action: chosen by single-threshold logic at the first call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Controller


class HysteresisPolicy(Controller):
    name = "hysteresis"

    def __init__(
        self,
        t_up_gbps:    float,
        t_down_gbps:  float,
        cooldown_steps: int = 1,
    ) -> None:
        if t_down_gbps > t_up_gbps:
            raise ValueError("t_down_gbps must be <= t_up_gbps")
        self.t_up_gbps    = t_up_gbps
        self.t_down_gbps  = t_down_gbps
        self.cooldown_steps = int(cooldown_steps)
        self._last_action: str | None = None
        self._cooldown_left = 0

    def reset(self) -> None:
        self._last_action = None
        self._cooldown_left = 0

    def required_obs_keys(self) -> tuple[str, ...]:
        return ("predicted_load_gbps",)

    def act(self, observation: dict[str, Any]) -> str:
        load = float(observation["predicted_load_gbps"])

        # Initial step: pick by midpoint of band
        if self._last_action is None:
            decision_pt = 0.5 * (self.t_up_gbps + self.t_down_gbps)
            self._last_action = "USR" if load < decision_pt else "DPDK"
            return self._last_action

        # In cooldown: must keep last action regardless of thresholds
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return self._last_action

        # Try to switch
        new_action = self._last_action
        if self._last_action == "USR" and load >= self.t_up_gbps:
            new_action = "DPDK"
        elif self._last_action == "DPDK" and load <= self.t_down_gbps:
            new_action = "USR"

        if new_action != self._last_action:
            self._cooldown_left = self.cooldown_steps
        self._last_action = new_action
        return new_action

    # Hysteresis is stateful; sequential per-cluster execution required.
    def act_sequence(self, predicted_load_gbps: np.ndarray) -> np.ndarray:
        """Run hysteresis over a 1-D time series. Returns matching string array."""
        out = np.empty(len(predicted_load_gbps), dtype=object)
        self.reset()
        for i, load in enumerate(predicted_load_gbps):
            out[i] = self.act({"predicted_load_gbps": float(load)})
        return out
