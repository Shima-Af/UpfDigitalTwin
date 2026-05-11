"""Digital twin core.

Two evaluation modes:

  evaluate_action(action, load)            — stateless lookup. Pure physics
                                              of one (action, load) point.

  TwinSession.step(action, load)           — stateful rollout.  Tracks the
                                              previous action and applies
                                              the configured activation
                                              accounting policy.

Activation accounting (configured in scenario.yaml `upf_switching.accounting`):
  sub_step   — weighted-average power within a switching step.  Old UPF
               serves for `activation_duration_s` of the step, new UPF for
               the rest.  Energy spike paid once.
  round_down — switch is instant within the step (only spike paid).
  round_up   — new UPF unavailable for the full step (worst-case).

The activation duration is hardware-independent for the same software
stack (loaded from data/external/profiling_twin/switching_costs.yaml).
The energy spike is computed at run-time as
  spike_wh(load) = surrogate_steady_power(load) × activation_duration_s / 3600
which is supported by the source measurement: their reported net energy
equals net_power × activation_duration to within rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import yaml

from .upf_profile import UPFProfile, UPFResult, UPFType


AccountingMode = Literal["sub_step", "round_down", "round_up"]


@dataclass
class StepResult:
    """Outcome of one stateful twin step."""

    realised_action:     str       # what physically served traffic this step
    requested_action:    str       # what the controller asked for
    upf_result:          UPFResult # composite (may be a weighted average across UPFs)
    switching_energy_wh: float     # spike + standby paid this step
    activation_pending:  bool      # True if requested UPF was not yet ready (round_up only)


class DigitalTwin:
    """Stateless twin (pure physics) plus a session factory for rollouts."""

    def __init__(self, scenario_cfg: dict, paths_cfg: dict, project_root: Path) -> None:
        upf_cfg   = scenario_cfg.get("upf", {})
        pt        = paths_cfg["profiling_twin"]
        models_dir    = project_root / pt["models"]
        manifest_path = project_root / pt["manifest"]
        self._profile = UPFProfile(upf_cfg, models_dir, manifest_path)

        # --- Switching cost configuration ---
        sw  = scenario_cfg.get("upf_switching", {})
        self.accounting: AccountingMode = sw.get("accounting", "sub_step")
        prewarm = sw.get("prewarm", {}) or {}
        self.prewarm_enabled    = bool(prewarm.get("enabled", False))
        self.prewarm_standby_w  = float(prewarm.get("standby_power_watts", 0.0))

        # Load activation durations from switching_costs.yaml (with provenance)
        sc_path = project_root / pt["switching_costs"]
        with open(sc_path) as fh:
            sc = yaml.safe_load(fh)
        self.activation_duration_s: dict[str, float] = {
            "DPDK": float(sc["activation_duration_s"]["dpdk"]),
            "USR":  float(sc["activation_duration_s"]["usr"]),
        }

        # Time-step in seconds + hours
        self.step_s = float(scenario_cfg["traffic"]["time_step_minutes"]) * 60.0
        self.step_h = self.step_s / 3600.0

    # ------------------------------------------------------------------
    # Stateless lookup (used by dashboard, sweeps, etc.)
    # ------------------------------------------------------------------

    def evaluate_action(self, action: UPFType, actual_load_gbps: float) -> UPFResult:
        return self._profile.evaluate(action, actual_load_gbps)

    def evaluate_batch(self, action: UPFType, load_gbps_arr, alt_power_watts=None):
        return self._profile.evaluate_batch(action, load_gbps_arr, alt_power_watts)

    # ------------------------------------------------------------------
    # Stateful session for sequential rollouts
    # ------------------------------------------------------------------

    def session(self, initial_action: str = "DPDK") -> "TwinSession":
        return TwinSession(self, initial_action)

    # ------------------------------------------------------------------
    # Pure switching computation — used by both TwinSession.step and the
    # batched rollout loop.  No side effects.
    # ------------------------------------------------------------------

    def compute_step(
        self,
        current_action: str,
        requested_action: str,
        old_result: UPFResult,
        new_result: UPFResult,
    ) -> tuple[UPFResult, float, str, bool]:
        """Apply the activation-accounting rule for one step.

        Returns:
            (composite_result, switching_energy_wh, new_current_action, activation_pending)

            composite_result   — weighted-average UPFResult for THIS step
            switching_energy_wh — spike + prewarm standby paid THIS step
            new_current_action  — what `current_action` should become for next step
            activation_pending  — True iff new UPF is not yet ready (round_up only)
        """
        # Standby power for prewarmed-but-not-serving DPDK
        standby_wh = (
            self.prewarm_standby_w * self.step_h
            if self.prewarm_enabled and current_action != "DPDK"
            else 0.0
        )

        # Case A: no switch
        if requested_action == current_action:
            return old_result, standby_wh, current_action, False

        # Case B: switch requested
        activation_s = self.activation_duration_s.get(requested_action, 0.0)
        spike_wh = new_result.power_watts * activation_s / 3600.0

        if self.accounting == "round_down":
            frac_old = 0.0
        elif self.accounting == "round_up":
            frac_old = 1.0
        else:  # sub_step
            frac_old = min(activation_s / self.step_s, 1.0) if self.step_s > 0 else 0.0

        # If frac_old == 1.0 the switch never completes this step → no spike yet
        if frac_old >= 1.0:
            return old_result, standby_wh, current_action, True

        # Time-weighted average power
        avg_power = frac_old * old_result.power_watts + (1.0 - frac_old) * new_result.power_watts

        # Composite KPI: KPIs reflect the UPF that served the *majority* of the
        # step.  Safety is conservative — any exposure to either UPF being
        # unsafe at this load counts.
        bulk = new_result if frac_old < 0.5 else old_result
        composite = replace(
            new_result,
            power_watts=avg_power,
            delay_us=bulk.delay_us,
            throughput_gbps=bulk.throughput_gbps,
            predicted_loss=max(old_result.predicted_loss, new_result.predicted_loss),
            is_safe=old_result.is_safe and new_result.is_safe,
            is_efficient=False,  # not meaningful during a switch
        )

        return composite, spike_wh + standby_wh, requested_action, False


class TwinSession:
    """Stateful per-episode wrapper.  One session per cluster / time series."""

    def __init__(self, twin: DigitalTwin, initial_action: str) -> None:
        self._twin = twin
        self._current_action: str = initial_action

    def step(self, requested_action: str, actual_load_gbps: float) -> StepResult:
        old_result = self._twin.evaluate_action(self._current_action, actual_load_gbps)
        new_result = (
            self._twin.evaluate_action(requested_action, actual_load_gbps)
            if requested_action != self._current_action
            else old_result
        )
        composite, sw_energy, new_current, pending = self._twin.compute_step(
            self._current_action, requested_action, old_result, new_result,
        )
        realised = self._current_action if pending else requested_action
        if not pending:
            self._current_action = new_current
        return StepResult(
            realised_action=realised,
            requested_action=requested_action,
            upf_result=composite,
            switching_energy_wh=sw_energy,
            activation_pending=pending,
        )
