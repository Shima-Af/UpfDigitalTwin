# Scenario Reconciliation — Digital Twin vs. RL Controller

**Status: OPEN — requires author decision. No config values were changed by this note.**

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
