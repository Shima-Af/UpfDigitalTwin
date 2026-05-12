"""UPF Digital Twin — public API.

Typical external use (e.g. from an RL Gymnasium env):

    from upf_digital_twin import DigitalTwin

    twin = DigitalTwin.from_data_dir(
        data_dir="/path/to/data/external",
        scenario_cfg=my_scenario_dict,
    )
    session = twin.session()
    result = session.step(action="USR", load_gbps=0.42)
"""

from upf_digital_twin.twin.digital_twin import DigitalTwin, StepResult
from upf_digital_twin.twin.upf_profile import UPFProfile, UPFResult, UPFType
from upf_digital_twin.twin.threshold_derivation import (
    derive_thresholds,
    load_forecast_mae_gbps,
)
from upf_digital_twin.policies.base import Controller
from upf_digital_twin.policies.static import StaticDPDKPolicy, StaticUSRPolicy
from upf_digital_twin.policies.threshold import ThresholdPolicy
from upf_digital_twin.policies.hysteresis import HysteresisPolicy
from upf_digital_twin.policies.oracle import OracleThresholdPolicy
from upf_digital_twin.evaluation.rollout import (
    run_rollout,
    precompute_twin_for_rollouts,
)
from upf_digital_twin.evaluation.metrics import compute_metrics
from upf_digital_twin.data.traffic_loader import load_traffic_artifacts
from upf_digital_twin.utils.config import load_configs, load_yaml, resolve_path

__all__ = [
    "DigitalTwin",
    "StepResult",
    "UPFProfile",
    "UPFResult",
    "UPFType",
    "derive_thresholds",
    "load_forecast_mae_gbps",
    "Controller",
    "StaticDPDKPolicy",
    "StaticUSRPolicy",
    "ThresholdPolicy",
    "HysteresisPolicy",
    "OracleThresholdPolicy",
    "run_rollout",
    "precompute_twin_for_rollouts",
    "compute_metrics",
    "load_traffic_artifacts",
    "load_configs",
    "load_yaml",
    "resolve_path",
]
