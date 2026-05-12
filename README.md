# UPF Digital Twin Environment

This repository implements a **measurement-grounded UPF digital twin environment**.

The *environment* aspect refers to the controller-facing API that evaluates candidate UPF actions.  
The *digital twin* aspect refers to the measurement-grounded predictive representation of physical UPF realizations — it answers *what would happen* given a traffic load and a UPF configuration.

---

## What this repo is

A controller-agnostic, measurement-grounded digital twin that answers:

> Given traffic demand and a candidate UPF configuration/action, what would happen in terms of **power, QoS, packet loss, delay, and safety**?

The controller answers a separate question: *which action should be chosen?*  
Policies live in `src/upf_digital_twin/policies/` and are kept strictly separate from the twin in `src/upf_digital_twin/twin/`.

## What this repo is NOT

- Not a traffic forecaster
- Not a PPO/RL training repo
- Not an orchestrator of other systems

---

## Source repositories

| Artifact | Source |
|---|---|
| Traffic forecasting artifacts | [UpfTrafficForecaster @ feature/cluster-first-stgnn](https://github.com/Shima-Af/UpfTrafficForecaster/tree/feature/cluster-first-stgnn) |
| UPF profiling / surrogate models | [UpfProfilingCampaign](https://github.com/Shima-Af/UpfProfilingCampaign) |

---

## Inputs

| File | Location | Description |
|---|---|---|
| `predictions_test.npy` | `data/external/traffic_forecaster/` | Normalised forecast array (N, H, K) |
| `targets_test.npy` | same | Normalised actual traffic (N, H, K) |
| `cluster_series.npy` | same | Full normalised cluster time series (K, T) |
| `cluster_assignments.parquet` | same | Base-station → cluster mapping |
| `cluster_bs_map.json` | same | Cluster → BS list |
| `forecast_eval_summary.json` | same | WAPE/MAE by K value |
| `params.yaml` | `data/external/profiling_twin/` | UPF capacity / threshold parameters |
| `models/manifest.json` | `data/external/profiling_twin/` | Surrogate model registry |

---

## Outputs (threshold demo)

| File | Description |
|---|---|
| `results/threshold_demo/metrics.json` | Per-policy metric summary |
| `results/threshold_demo/rollout_threshold.parquet` | Per-step records, threshold policy |
| `results/threshold_demo/rollout_static_dpdk.parquet` | Per-step records, always DPDK |
| `results/threshold_demo/rollout_static_usr.parquet` | Per-step records, always USR |
| `results/threshold_demo/rollout_oracle.parquet` | Per-step records, oracle baseline |

---

## First demo use case

Compare a simple **threshold policy** (route to USR if forecast < 0.12 Gbps, else DPDK)
against two static baselines (always-DPDK, always-USR) and an oracle upper bound.

Metrics reported: total energy (Wh), average power (W), energy saving vs static DPDK,
unsafe USR rate, USR/DPDK usage ratio, decision flip rate, energy regret vs oracle.

---

## How to run

**Smoke test** — verify all artifacts and configs are present:
```bash
python scripts/smoke_test_artifacts.py
```

**Threshold demo** — run all four policies and print comparison table:
```bash
python scripts/run_threshold_demo.py
```

Results are written to `results/threshold_demo/`.

---

## Project structure

```
configs/
  paths.yaml          File-system paths (no hardcoded paths in code)
  scenario.yaml       Scenario parameters (alpha, thresholds, UPF config)

data/external/
  traffic_forecaster/ Artifacts from UpfTrafficForecaster
  profiling_twin/     Artifacts from UpfProfilingCampaign

src/upf_digital_twin/
  data/
    traffic_loader.py Load and validate traffic forecaster artifacts
  twin/
    upf_profile.py    UPF behavior model (placeholder → real surrogate later)
    digital_twin.py   Twin core: evaluate_action / evaluate_many
  policies/
    static.py         StaticDPDKPolicy, StaticUSRPolicy
    threshold.py      ThresholdPolicy (uses predicted load)
    oracle.py         OracleThresholdPolicy (uses actual load — upper bound only)
  evaluation/
    rollout.py        Run a policy over (N, H, K) test arrays
    metrics.py        Energy, safety, flip-rate, regret metrics
  utils/
    config.py         Load configs/paths.yaml + configs/scenario.yaml

scripts/
  smoke_test_artifacts.py  Verify all files and shapes are correct
  run_threshold_demo.py    Run threshold vs baselines, save results
```

---

## Using as a library (e.g. from an RL Gymnasium env)

This repo is `pip install`-able. From another project:

```bash
pip install -e /path/to/UPF_NDT
```

Then, in the consumer (the data files are *not* shipped with the package — see the
data contract below for what to provide):

```python
from upf_digital_twin import DigitalTwin

scenario_cfg = {
    "upf": {
        "qos_budget": {"delay_budget_us": 200.0, "max_loss_pkts_per_interval": 0.0},
    },
    "upf_switching": {
        "accounting": "sub_step",
        "prewarm": {"enabled": True, "standby_power_watts": 0.05},
    },
    "traffic": {"time_step_minutes": 15},
}

twin = DigitalTwin.from_data_dir(
    data_dir="/path/to/your/data/external",  # see data contract below
    scenario_cfg=scenario_cfg,
)

# Stateless: pure physics of one (action, load) point
result = twin.evaluate_action("USR", actual_load_gbps=0.42)

# Stateful: drive a sequential rollout (used inside a gym env's step())
session = twin.session(initial_action="DPDK")
step    = session.step(requested_action="USR", actual_load_gbps=0.42)
```

### Data contract for external consumers

`DigitalTwin.from_data_dir(data_dir, ...)` expects this layout under `data_dir`:

```
<data_dir>/
  profiling_twin/
    models/
      manifest.json          # surrogate model registry
      layer1/                # layer-1 surrogate models
      layer2/                # layer-2 surrogate models
    switching_costs.yaml     # activation durations (DPDK, USR)
```

Source these from [UpfProfilingCampaign](https://github.com/Shima-Af/UpfProfilingCampaign).
Traffic-forecaster artifacts (`predictions_test.npy`, etc.) are only needed if the
consumer also calls `load_traffic_artifacts` — for a gym env, traffic typically comes
from the RL repo's own data pipeline, so the `profiling_twin/` subtree is the only
hard requirement.
