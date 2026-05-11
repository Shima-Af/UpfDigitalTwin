"""Static baseline policies — always pick the same UPF."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Controller


class StaticDPDKPolicy(Controller):
    name = "static_dpdk"

    def reset(self) -> None: ...
    def act(self, observation: dict[str, Any]) -> str: return "DPDK"

    def act_array_stateless(self, predicted_load_gbps: np.ndarray) -> np.ndarray:
        return np.full(predicted_load_gbps.shape, "DPDK", dtype=object)


class StaticUSRPolicy(Controller):
    name = "static_usr"

    def reset(self) -> None: ...
    def act(self, observation: dict[str, Any]) -> str: return "USR"

    def act_array_stateless(self, predicted_load_gbps: np.ndarray) -> np.ndarray:
        return np.full(predicted_load_gbps.shape, "USR", dtype=object)
