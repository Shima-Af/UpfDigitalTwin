# Scenario Reconciliation — Digital Twin vs. RL Controller

**Status: RESOLVED (2026-07-21) — author chose to unify on the controller's values.**
`configs/scenario.yaml` now ships alpha=1.0, max_loss=5.0, prewarm=off, plus a fixed
20 Mbps hysteresis band (fourth divergence, found during reconciliation — see §4).

> ## CHAPTER RE-RUN AND REWRITTEN (2026-07-21)
>
> `reports/chapter_digital_twin.tex`, its five figures, and the built PDF have all been
> regenerated under the reconciled config. The headline result changed by roughly an
> order of magnitude:
>
> | | old (chapter) | reconciled |
> |---|---|---|
> | Hysteresis energy saving | **49.2 %** | **4.8 %** |
> | Static USR energy saving | **37.3 %** | **−154.5 %** (costs more than DPDK) |
> | Oracle ceiling | 51.0 % | **6.7 %** |
> | Steps below decision threshold | 77.8 % | 14 % |
>
> The chapter's argument was rewritten, not merely renumbered — see "Chapter rewrite"
> in the addendum for which passages changed and which claims were reversed. **The
> rewritten narrative needs author review**: it now presents the twin's contribution as
> bounding the achievable headroom rather than demonstrating a large saving.

The sections below were written before the decision and describe the pre-reconciliation
state.

Date: 2026-07-21
Twin config: `configs/scenario.yaml`
Downstream consumer: the RL controller repository's `configs/scenario_rl.yaml`, which
declares itself canonical for the controller chapter and pins this package at `v0.1.0`.

This note maps each divergent field to a canonical value, a rationale, and — critically —
whether the twin chapter's already-reported numbers were computed under a value that
differs from the controller chapter's.

## Provenance of the twin chapter's numbers

Re-running the rollout at the twin's current config reproduces
`reports/chapter_digital_twin.tex` exactly:

| Chapter claim | Value in .tex | Reproduced |
|---|---|---|
| Static DPDK energy | 2067.1 Wh | 2067.05 Wh |
| Static USR energy | 1296.1 Wh | 1296.06 Wh |
| Static USR saving / unsafe | 37.3 % / 18.8 % | 37.30 % / 18.80 % |
| Hysteresis saving / unsafe | 49 % / 1.7 % | 49.21 % / 1.67 % |
| Decision threshold | 71 Mbps (QoS-limited) | 71.0 Mbps (QoS-limited) |
| Steps below threshold | ~78 % | 77.8 % |

So the chapter as written was computed under **alpha = 0.12, max_loss = 0.0,
prewarm = enabled @ 0.05 W**. Every divergence below therefore has the potential to
invalidate a printed number, and none of them were silently applied.

---

## 1. `traffic.calibration.alpha_gbps_per_norm` — 0.12 (twin) vs `traffic.alpha` = 1.0 (controller)

**Verdict: GENUINE DIVERGENCE. Must be unified. Highest severity of the three.**

This is not a tuning difference; it is a different physical regime, and the two chapters
currently describe different papers. Measured on the shared traffic artifacts
(10,090 cluster-steps, 1-step-ahead actuals):

| | alpha = 0.12 | alpha = 1.0 |
|---|---|---|
| Mean offered load | 0.056 Gbps | 0.468 Gbps |
| Peak load | 0.70 Gbps | 5.80 Gbps |
| Per-cluster mean range | 0.013–0.199 Gbps | 0.108–1.659 Gbps |
| **Steps below the 71 Mbps decision threshold** | **77.8 %** | **11.5 %** |

The alpha = 1.0 column independently corroborates the controller repo's declared
0.06–1.66 Gbps / 5.8 Gbps peak figures, so the two repos agree on the *arithmetic*;
they disagree only on which alpha is canonical.

**Why this is severe.** The twin chapter's motivating argument is a claim about the load
distribution: "roughly 78 % of all cluster-steps fall below the decision threshold, i.e.
in the regime where USR is both safe and cheaper... a controller that exploits this stands
to recover most of DPDK's fixed power cost." At alpha = 1.0 that figure is **11.5 %**. The
argument does not weaken — it **inverts**. USR-by-default becomes the wrong default, and
the chapter's central "why energy-aware switching pays" subsection would need rewriting,
not renumbering. The energy-saving and unsafe-rate rows of Table `tab:dt_metrics` would
also all change, since USR is safe on far fewer steps.

**Recommendation.** Both repos should carry alpha = 1.0, and the twin chapter should be
re-run and re-written under it. Reasons:

- The controller repo's rationale is sound and explicitly documented: NetMob is
  dimensionless by construction (privacy-preserving), the forecaster sums per-BS
  `dl_norm` across each cluster, and **no physical scaler exists to recover**. Neither
  0.12 nor 1.0 is measured; both are declarations. Given that, the tie should be broken
  by which declaration produces the more defensible operating regime.
- alpha = 1.0 puts the fleet at 0.06–1.66 Gbps mean with 5.8 Gbps peaks — a realistic
  per-site MEC-UPF range. alpha = 0.12 puts the *entire fleet* below 0.70 Gbps peak,
  i.e. almost always under the USR knee. A twin whose safety mechanism almost never
  binds is a weak demonstration of a safety mechanism.
- An examiner comparing the two chapters will otherwise find the same twin, the same
  data, and two incompatible load scales with no stated reason.

**Fallback if a re-run is not feasible before submission.** Keep 0.12 but state it
explicitly in the chapter as an illustrative low-load scenario — name it as a declared
assumption, not a calibration; give the reason (no recoverable scaler); state the
resulting regime (peak 0.70 Gbps); and cross-reference that the controller chapter
adopts 1.0 and why. Silence here is the one option that is not defensible.

**AUTHOR SIGN-OFF REQUIRED — this changes every aggregate in Table `tab:dt_metrics`
and inverts the 78 % claim.**

## 2. `upf.qos_budget.max_loss_pkts_per_interval` — 0.0 (twin) vs 5.0 (controller)

**Verdict: PARTIALLY defensible, but NOT harmless as assumed. Must be unified for the
threshold; may stay split for the reward.**

The framing that "the twin's binary `is_safe` legitimately uses 0.0 while the
controller's graded score needs a nonzero normalisation base" is correct as far as it
goes — 0.0 would indeed divide-by-zero in the graded score. But this field does not only
feed `is_safe` at evaluation time. It feeds **threshold derivation**, and the twin
publishes those thresholds for the controller to consume. Measured:

| max_loss | Energy break-even | QoS limit | Decision threshold | Binding constraint |
|---|---|---|---|---|
| **0.0** (twin) | 91.0 Mbps | **81.0 Mbps** | **71.0 Mbps** | QoS-limited |
| **5.0** (controller) | 91.0 Mbps | **149.0 Mbps** | **81.0 Mbps** | energy-limited |

Raising the loss tolerance to 5 packets/interval moves the QoS limit by **68 Mbps** and
**flips which constraint binds**. The chapter's sentence "giving lambda_dec = 71 Mbps
here (QoS-limited)" becomes false under 5.0 — it would be 81 Mbps and energy-limited.
That parenthetical is load-bearing: the surrounding text argues the QoS limit is
discoverable *only because* the twin uses `usr_full`, and that argument rests on QoS
being the binding constraint.

**Recommendation.** Split the field by role rather than letting one value serve two
masters:

- Keep **0.0** as the twin's `is_safe` / threshold-derivation budget. Zero tolerance is
  the honest reading of "QoS preserved" and it is what the chapter's numbers assume.
- Let the controller keep **5.0** *only* as the denominator of its graded loss score,
  and rename it there (e.g. `loss_score_normalisation_pkts`) so it is visibly not a
  safety budget.
- If instead 5.0 is meant as a genuine relaxation of the QoS budget, then both repos
  must adopt it and the twin chapter's threshold section needs re-running.

Left unsplit, the two repos will disagree about which loads are safe while appearing to
share one config field.

**AUTHOR SIGN-OFF REQUIRED — under 5.0 the decision threshold moves 71 → 81 Mbps and
"QoS-limited" becomes "energy-limited".**

## 3. `upf_switching.prewarm` — enabled/0.05 W (twin) vs disabled/0.0 W (controller)

**Verdict: GENUINE DIVERGENCE. Must be unified. Magnitude-affecting.**

Prewarm is a modelling choice about the deployment, not about a consumer's scoring
scheme, so the two repos cannot defensibly hold different values — they would be
modelling different hardware setups. Measured effect on the chapter's headline table:

| | prewarm on (chapter) | prewarm off (controller) |
|---|---|---|
| Static USR saving | **37.3 %** | **43.4 %** |
| Hysteresis saving | **49.2 %** | **53.8 %** |
| Hysteresis "switch energy" | 95.63 Wh | 0.40 Wh |
| Unsafe rates | unchanged (18.80 % / 1.67 %) | unchanged |

Adopting the controller's `enabled: false` would move both headline savings up by
~5–6 percentage points. Safety metrics are unaffected.

**Additional finding — a reporting-semantics bug, independent of which value wins.**
Under prewarm, the standby draw is being accumulated into `switch_energy_wh`. Static USR
*never switches*, yet reports 126.00 Wh of "switch energy" — that figure is 100 % standby
(0.05 W x 10,090 steps x 0.25 h = 126.1 Wh). Any statement of the form "switching costs X Wh"
derived from this column is currently reporting a continuous standby tax instead. Recommend
separating `standby_energy_wh` from `switch_energy_wh` in `evaluation/metrics.py` regardless
of the prewarm decision; the chapter's Table `tab:dt_metrics` does not print this column, so
this is a correctness fix rather than an erratum.

**Recommendation.** Prefer the controller's `enabled: false` / 0.0 W as canonical, unless
the profiling campaign specifically justifies an always-on standby DPDK. It is the more
conservative claim (it does not spend energy the deployment may not actually spend), and
it makes the switching-cost column mean what its name says. Requires re-running the twin
chapter's table.

**AUTHOR SIGN-OFF REQUIRED — changes both reported saving percentages.**

---

## Summary

| Field | Twin | Controller | Verdict | Chapter numbers affected? |
|---|---|---|---|---|
| `alpha` | 0.12 | 1.0 | Unify — recommend **1.0** | **Yes — inverts the 78 % claim, changes all aggregates** |
| `max_loss_pkts_per_interval` | 0.0 | 5.0 | Split by role — twin keeps **0.0**, controller renames its 5.0 | **Yes if unified at 5.0 — threshold 71 → 81 Mbps** |
| `prewarm` | on / 0.05 W | off / 0.0 W | Unify — recommend **off** | **Yes — savings 37.3 → 43.4 %, 49.2 → 53.8 %** |

All three are magnitude-affecting. **No config value in this repo has been changed**;
`configs/scenario.yaml` still ships the values under which the chapter was computed, so
the chapter and the package remain self-consistent as released. Once the author decides,
the config change plus a re-run of `reports/make_figures.py` and
`scripts/run_threshold_demo.py` should land together in one commit and a new tag, so the
controller repo can bump its pin to a version whose numbers match its own chapter.

### Reproducing the tables above

```bash
python scripts/run_threshold_demo.py     # thresholds + rollout aggregates
python reports/make_figures.py           # chapter figures
```

Vary the three fields in `configs/scenario.yaml` to reproduce each comparison column.

---

# Addendum (2026-07-21) — decision, baseline audit, and reconciled numbers

## Decision

Author elected to unify on the controller's values: **alpha = 1.0, max_loss = 5.0,
prewarm = off**. Applied to `configs/scenario.yaml`.

## 4. `threshold.hysteresis_band` — auto (twin) vs fixed 20 Mbps (controller)

**A fourth divergence, not in the original brief. Found by auditing the controller's
baselines. This one silently breaks the hysteresis controller.**

The twin config used `hysteresis_band: auto` (= 2 x forecast MAE). The MAE is converted
to Gbps through alpha, so **the band scales with alpha**:

| alpha | forecast MAE | auto band | decision threshold | t_down |
|---|---|---|---|---|
| 0.12 | 13.3 Mbps | 26.6 Mbps | 71 Mbps | 44.4 Mbps (fine) |
| 1.0 | 110.7 Mbps | **221.3 Mbps** | 81 Mbps | **0.0 Mbps (broken)** |

At alpha=1.0 the band is wider than the decision threshold, so `t_down` clamps to 0, the
controller can never switch back to USR, and hysteresis **degenerates into always-DPDK** —
while still running and still reporting metrics. It is not obviously broken from the output.

The controller repo had already hit this independently. Its dashboard documents
`hysteresis-auto` as "paper-faithful auto band (2 x forecast MAE); **tends to degenerate
to always-DPDK when the forecaster is noisy**", and its headline classical baseline is
`hysteresis-tuned` with a manually-set **20 Mbps** band, labelled "best classical
controller".

**Resolution.** Twin config set to `hysteresis_band: 20.0` to match. Additionally,
`derive_thresholds` now emits a `RuntimeWarning` when `band >= decision`, so this failure
mode cannot recur silently in either repo.

## Baseline audit — do the two repos' baselines match?

**Algorithmically yes; the divergence was entirely in configuration.**
`UpfRLControllers/src/baselines/{threshold_derivation,hysteresis}.py` are acknowledged
verbatim ports of this repo's implementations ("Ports `derive_thresholds` from
UpfDigitalTwin"). Diffing them shows only cosmetic drift:

| | Twin | Controller | Behavioural? |
|---|---|---|---|
| Switching rule, cooldown, initial-action logic | identical | identical | no |
| `min(breakeven, qos) - margin` | identical | identical | no |
| band = 2 x MAE | identical | identical | no |
| Action encoding | `"DPDK"` / `"USR"` | `0` / `1` | no |
| Base class | `Controller` ABC | plain class | no |
| `load_forecast_mae_gbps` input | parsed dict | `Path` | no |
| `MultiAgentHysteresis` (K instances) | absent | present | controller-only |
| **Hysteresis band in use** | **auto (2 x MAE)** | **fixed 20 Mbps** | **YES** |

So the earlier degenerate result was not caused by mismatched baseline code. It was the
band configuration, and it would have hit the controller repo identically had it used
`auto`.

## Reconciled numbers (10,090 cluster-steps, alpha=1.0, max_loss=5.0, prewarm=off)

Derived operating points: break-even 91.0 Mbps, QoS limit 149.0 Mbps,
**decision threshold 81.0 Mbps (energy-limited)**.

| Policy | Energy (Wh) | Avg power (W) | Saving (%) | Unsafe USR (%) | USR use (%) | Flip rate |
|---|---|---|---|---|---|---|
| Static DPDK | 2070.72 | 0.8209 | 0.0 | 0.00 | 0.0 | 0.000 |
| Static USR | 5269.81 | 2.0891 | **−154.5** | 70.26 | 100.0 | 0.000 |
| Threshold | 1969.57 | 0.7803 | 4.9 | 4.16 | 14.0 | 0.042 |
| Hysteresis (auto band) | 2070.72 | 0.8209 | **0.0 — degenerate** | 0.00 | 0.0 | 0.000 |
| **Hysteresis (band=20 Mbps)** | 1970.99 | 0.7811 | **4.8** | 3.02 | 11.5 | 0.021 |
| Oracle | 1931.03 | 0.7647 | 6.7 | 0.00 | 13.7 | 0.072 |

## Open concern for the author and supervisor

At the reconciled settings the **oracle ceiling is 6.7 %** — that is the best any
controller with perfect foresight can achieve on this hardware and traffic. The
best realisable classical controller gets 4.8 %, i.e. it already captures ~72 % of the
available headroom.

This is worth raising deliberately, because it bears on both chapters:

- The twin chapter's energy-saving contribution shrinks from ~49 % to ~4.8 %.
- Static USR is no longer an energy-greedy alternative; it *costs 2.5x DPDK* and is
  unsafe 70 % of the time. The framing "USR saves energy but risks QoS" no longer holds.
- For the controller chapter: if the classical baseline is within 1.9 pp of oracle,
  the room for a learned policy to demonstrate value is thin. It is worth checking what
  the trained MAPPO/PPO agents actually achieve against these baselines before
  committing to alpha=1.0 in print.

The underlying cause is physical: the profiled UPF's USR path saturates around 81-149
Mbps, while alpha=1.0 places traffic at 0.11-1.66 Gbps mean with 5.8 Gbps peaks — one to
two orders of magnitude above the knee. In that regime DPDK is almost always correct and
there is little switching decision left to make. alpha=0.12 was the regime in which the
measured hardware had a genuine decision; alpha=1.0 is the regime the controller repo
declared. Both are declarations, and the choice determines whether the thesis has an
interesting control problem.

## Reducing the duplication

The controller repo currently maintains ~200 lines of ported baseline code that must stay
in lockstep with this repo by hand. This reconciliation is exactly the failure that
duplication produces. Recommended direction (one-way dependency, twin stays consumer-
agnostic):

1. This repo exports the canonical baselines as public API — `derive_thresholds` plus the
   `Static*`, `Threshold`, `Hysteresis`, `Oracle` policies.
2. The controller deletes `src/baselines/` and imports from `upf_digital_twin`, keeping
   only what is genuinely its own: `MultiAgentHysteresis` and the int-action adapter.
3. Two small accommodations here to make that painless, neither controller-specific:
   accept either a parsed dict or a path in `load_forecast_mae_gbps`, and expose a
   neutral action-encoding helper so int-valued consumers need not re-implement policies.
4. Scenario fields that both repos read (alpha, QoS budget, switching, band) should live
   in one file shipped by this package, with the controller overlaying only RL-specific
   blocks — so a divergence like this one becomes impossible rather than merely detectable.

---

# Chapter rewrite (2026-07-21)

Figures and rollouts regenerated (`reports/make_figures.py`,
`scripts/run_threshold_demo.py`), `.tex` updated, PDF rebuilt (15 pages).

**Bug found while regenerating:** `make_figures.py` hardcoded
`safety_margin_mbps=10.0` and always used the auto band, ignoring the scenario's
`threshold:` block entirely. It would have rendered the degenerate always-DPDK
hysteresis figure while the demo script rendered the correct one. Now reads the config,
matching `run_threshold_demo.py`. The load-distribution histogram range was also
hardcoded to 0–0.5 Gbps, which clipped the entire distribution at alpha=1.0; it now
follows the data.

## Numbers updated

| Location | Was | Now |
|---|---|---|
| Eq. `eq:dt_alpha` | alpha = 0.12 | alpha = 1.0 |
| Eq. `eq:dt_is_safe` | L_max = 0 (zero tolerance) | L_max = 5 pkts/interval |
| QoS limit | 81 Mbps | 149 Mbps |
| Decision threshold | 71 Mbps (QoS-limited) | 81 Mbps (energy-limited) |
| Hysteresis band | 26.6 Mbps (auto, from 13.3 Mbps MAE) | 20 Mbps fixed |
| Switch points | 71 / 44 Mbps | 81 / 61 Mbps |
| Steps below threshold | 78 % | 14 % |
| Table `tab:dt_metrics` | 3 rows | 5 rows (adds Threshold + Oracle bound) |

## Claims reversed — these need author review

1. **"Why energy-aware switching pays"** (retitled *"How much room is there for
   energy-aware switching?"*). The old section argued 78 % of steps sit below the
   threshold so a controller recovers most of DPDK's cost. That is now 14 %. The section
   was rewritten to establish the 6.7 % oracle bound *before* presenting controllers,
   and to attribute the narrow headroom to a physical capacity mismatch (USR saturates
   at 91–149 Mbps; alpha=1 puts loads at 0.11–1.66 Gbps mean, 5.8 Gbps peak).

2. **Static USR framing.** Previously "cheap but unsafe" (saves 37 %, unsafe 18.8 %).
   Now it is the *most expensive* option: 2.5x DPDK's energy, unsafe on 70.3 % of steps.
   The intuition that the software UPF is the cheap option now holds only below the knee,
   and the chapter says so explicitly.

3. **The demonstration claim** (summary item 5). Was "recovers 49 % of DPDK's energy."
   Now "recovers 4.8 % — roughly 72 % of the 6.7 % available to an informed oracle."
   A sixth summary item was added on bounded headroom, arguing that establishing the
   bound is itself the result and that UPF *capacity*, not control policy, is the
   dominant lever in this deployment.

4. **Anti-flapping section.** Now reports the auto-band failure mode explicitly: the band
   inherits alpha through the forecast MAE, reaching 221 Mbps at alpha=1.0 — wider than
   the 81 Mbps threshold — which clamps `t_down` to 0 and silently degenerates hysteresis
   into static DPDK. Eq. `eq:dt_band` is retained as the principled default; the fixed
   20 Mbps band actually used is now Eq. `eq:dt_band_used`.

5. **Prewarm.** Now stated as disabled, with the cold-start spikes visible in the rollout
   figure and a note that this is the more conservative accounting.

The alpha passage additionally states plainly that alpha is a *declared modelling
assumption, not a recovered calibration*, gives the reason (NetMob is dimensionless;
no scaler exists to fit), and cross-references the controller chapter.

## Still open

The concern recorded above stands and is now visible in the chapter itself: the oracle
bound is 6.7 % and the best classical controller reaches 4.8 %. Whether a learned policy
can justify its complexity against a 1.9 pp gap is a question for the controller chapter,
and worth settling before either chapter goes to print.
