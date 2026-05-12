"""UPF behavior profile — measurement-grounded surrogate models.

Loads the two-layer sklearn surrogates from UpfProfilingCampaign
(data/external/profiling_twin/models/) and uses the lite variant,
which only requires traffic throughput in kbits/s as input — derived
from load_gbps via unit conversion.  This makes it compatible with
NetMob-level traffic data that has no per-packet observability.

Layer 1: offered load → {throughput, cpu, packet_loss_delta, delay}
Layer 2: offered load + L1 predictions → power_watts

Safety semantics (one flag per concern, no redundancy):
  is_safe       — QoS is preserved (no predicted packet loss AND delay
                  within configured budget). DPDK is always safe within
                  measured load range.
  is_efficient  — Power for this UPF is lower than power of the
                  alternative UPF at the same load.

A USR result with is_safe=True, is_efficient=False means USR is operating
within QoS but already consuming more than DPDK — the threshold controller
should switch to DPDK from this point onward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

UPFType = Literal["DPDK", "USR"]

_GBPS_TO_KBITS_S = 1_000_000.0   # 1 Gbps = 1e6 kbits/s

# Maps the twin's UPF type strings to profiling-campaign variant names
_VARIANT_MAP: dict[str, str] = {
    "DPDK": "dpdk",
    "USR":  "usr_full",
}

L1_TARGETS = [
    "throughput_gbps",
    "cpu_pct",
    "gtpu_packets_dn__packets_lost_delta",
    "downlink_one_way_delay_distribution__weighted_mean_delay_us",
]


@dataclass
class UPFResult:
    """Structured output of one UPF evaluation at one load point."""

    upf_type: str
    load_gbps: float
    power_watts: float
    delay_us: float
    throughput_gbps: float
    predicted_loss: float
    is_safe: bool        # QoS preserved
    is_efficient: bool   # power < alternative UPF's power at same load


class UPFProfile:
    """Measurement-grounded UPF surrogate using UpfProfilingCampaign lite models.

    Models are loaded lazily on first use and cached for subsequent calls.
    """

    def __init__(self, upf_cfg: dict, models_dir: Path, manifest_path: Path) -> None:
        # QoS budgets (used to compute is_safe)
        qos = upf_cfg.get("qos_budget", {})
        self._delay_budget_us       = float(qos.get("delay_budget_us", 200.0))
        self._max_loss_pkts         = float(qos.get("max_loss_pkts_per_interval", 0.0))

        self._models_dir   = models_dir
        self._manifest     = self._load_manifest(manifest_path)
        self._model_cache: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_dpdk(self, load_gbps: float) -> UPFResult:
        return self._evaluate("DPDK", load_gbps)

    def evaluate_usr(self, load_gbps: float) -> UPFResult:
        return self._evaluate("USR", load_gbps)

    def evaluate(self, upf_type: UPFType, load_gbps: float) -> UPFResult:
        if upf_type not in ("DPDK", "USR"):
            raise ValueError(f"Unknown upf_type '{upf_type}'. Expected 'DPDK' or 'USR'.")
        return self._evaluate(upf_type, load_gbps)

    def evaluate_batch(
        self,
        upf_type: UPFType,
        load_gbps_arr: np.ndarray,
        alt_power_watts: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Vectorised evaluation over a 1-D array of load values.

        Args:
            upf_type:        "DPDK" or "USR".
            load_gbps_arr:   (N,) array of offered loads.
            alt_power_watts: Optional (N,) array of the *other* UPF's power at
                             the same loads. If provided, is_efficient is
                             computed against it; otherwise is_efficient is
                             defined as False (no comparison available).

        Returns dict with: power_watts, throughput_gbps, cpu_pct,
        delay_us, predicted_loss, is_safe, is_efficient.
        """
        variant = _VARIANT_MAP[upf_type]
        X       = self._lite_X(load_gbps_arr)          # (N, 2)

        # --- Layer 1 ---
        l1: dict[str, np.ndarray] = {}
        for target in L1_TARGETS:
            model = self._get_model(variant, "layer1", target)
            l1[target] = model.predict(X)               # type: ignore[union-attr]

        # --- Layer 2 ---
        X_l2 = np.column_stack([
            X,
            l1["throughput_gbps"],
            l1["cpu_pct"],
            l1["gtpu_packets_dn__packets_lost_delta"],
            l1["downlink_one_way_delay_distribution__weighted_mean_delay_us"],
        ])
        power_model = self._get_model(variant, "layer2", "power_watts")
        power_watts = np.maximum(0.0, power_model.predict(X_l2))  # type: ignore[union-attr]

        # --- Safety: QoS preserved ---
        delay_us       = l1["downlink_one_way_delay_distribution__weighted_mean_delay_us"]
        predicted_loss = l1["gtpu_packets_dn__packets_lost_delta"]
        is_safe = (predicted_loss <= self._max_loss_pkts) & (delay_us <= self._delay_budget_us)

        # --- Efficiency: this UPF cheaper than alternative at this load ---
        if alt_power_watts is not None:
            is_efficient = power_watts < alt_power_watts
        else:
            is_efficient = np.zeros(len(load_gbps_arr), dtype=bool)

        return {
            "power_watts":      power_watts,
            "throughput_gbps":  l1["throughput_gbps"],
            "cpu_pct":          l1["cpu_pct"],
            "delay_us":         delay_us,
            "predicted_loss":   predicted_loss,
            "is_safe":          is_safe,
            "is_efficient":     is_efficient,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_manifest(self, path: Path) -> dict:
        with open(path) as fh:
            return json.load(fh)

    def _get_model(self, variant: str, layer: str, target: str) -> object:
        key = f"{variant}__{layer}__{target}__lite"
        if key not in self._model_cache:
            entry = self._manifest.get(key)
            if entry is None:
                raise KeyError(f"No manifest entry for '{key}'")
            rel = Path(entry["path"].replace("\\", "/"))
            pkl_path = self._models_dir / rel.name if rel.parts[0] == "models" else self._models_dir / rel
            if not pkl_path.exists():
                fname = f"{variant}__{target}__lite.pkl"
                layer_dir = "layer1" if layer == "layer1" else "layer2"
                pkl_path = self._models_dir / layer_dir / fname
            self._model_cache[key] = joblib.load(pkl_path)
        return self._model_cache[key]

    def _lite_X(self, load_gbps_arr: np.ndarray) -> np.ndarray:
        """Build (N, 2) lite input matrix from a 1-D load array."""
        kbits = load_gbps_arr * _GBPS_TO_KBITS_S
        return np.column_stack([kbits, kbits])

    def _evaluate(self, upf_type: UPFType, load_gbps: float) -> UPFResult:
        """Evaluate a single load point with full alternative-aware efficiency flag."""
        loads = np.array([load_gbps])
        alt = "USR" if upf_type == "DPDK" else "DPDK"
        # Predict alt power once for the efficiency flag
        alt_power = self.evaluate_batch(alt, loads)["power_watts"]
        out = self.evaluate_batch(upf_type, loads, alt_power_watts=alt_power)
        return UPFResult(
            upf_type=upf_type,
            load_gbps=load_gbps,
            power_watts=float(out["power_watts"][0]),
            delay_us=float(out["delay_us"][0]),
            throughput_gbps=float(out["throughput_gbps"][0]),
            predicted_loss=float(out["predicted_loss"][0]),
            is_safe=bool(out["is_safe"][0]),
            is_efficient=bool(out["is_efficient"][0]),
        )
