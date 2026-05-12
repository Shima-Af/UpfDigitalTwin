"""Controller (policy) base class.

Every controller — static, threshold, hysteresis, oracle, future PPO — exposes
the same surface so the rollout engine and the dashboard can call them
interchangeably.

Why not Gym?
  Gym requires obs/action spaces, step/reset, info dicts. That overhead is
  only useful when an external RL library calls the env from outside. For
  in-process evaluation of any policy (rule-based or learned) we just need
  reset() + act(obs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Controller(ABC):
    """Minimal controller interface — works for rule-based and learned policies."""

    name: str = "base"

    @abstractmethod
    def reset(self) -> None:
        """Clear any internal state (last action, cooldown counter, etc.)."""

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> str:
        """Return one of {'DPDK', 'USR'} given the observation dict.

        The observation must contain keys the controller declares it needs
        (see Controller.required_obs_keys()).
        """

    def required_obs_keys(self) -> tuple[str, ...]:
        """Tuple of obs keys this controller reads. Used for validation."""
        return ()

    # ------------------------------------------------------------------
    # Convenience: vectorised single-step decisions for stateless policies
    # ------------------------------------------------------------------

    def act_array_stateless(self, predicted_load_gbps: np.ndarray) -> np.ndarray:
        """Default vectorised path for stateless policies.

        Stateful policies (hysteresis, RL with memory) should override this
        OR be wrapped by a per-cluster sequential rollout.
        """
        out = np.empty(predicted_load_gbps.shape, dtype=object)
        flat = predicted_load_gbps.ravel()
        out_flat = out.ravel()
        for i, load in enumerate(flat):
            out_flat[i] = self.act({"predicted_load_gbps": float(load)})
        return out
