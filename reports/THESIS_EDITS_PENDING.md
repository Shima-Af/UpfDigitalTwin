# Pending thesis edits — Digital Twin chapter

Manual LaTeX edits still to apply in `chapter_digital_twin.tex`. Code, data, and
results are already fixed and committed; these are prose/equation changes only.
Delete each entry once applied.

---

## 1. Energy-spike rule — ✅ DONE 2026-07-23 (applied in chapter; kept for record)

**Where:** subsection "Activation duration and the energy-spike rule"
(~line 377–390, the paragraph after the activation-duration paragraph).

**Why:** the printed Eq. (dt_spike) is wrong two ways — it writes the spike as a
function of load-dependent steady power $P_{\mathrm{steady}}(\lambda)$, but
activation happens at zero traffic so the cost is a per-variant constant; and it
"validates" itself with $34.81$ W, which is the net **burst** power, not steady
power. Code + `switching_costs.yaml` now implement the corrected constant model
(committed: profiling `thesis-v1.2`, twin `58a01b2`).

**Replace the old block (keep the activation-duration paragraph above it) with:**

```latex
Absolute switching \emph{energy} measured on one machine is not portable to
another, and activation occurs at \emph{zero traffic}—the new \upf{} is not yet
forwarding—so the spike cannot depend on the serving load $\lambda$. It is a
per-variant constant. The portable quantity is the dimensionless
\emph{burst ratio}
\begin{equation}
  k_v \;=\; \frac{P^{\mathrm{act}}_{v}}{P^{\mathrm{idle}}_{v}}\Bigg|_{\mathrm{source}}
  \label{eq:dt_burst}
\end{equation}
the factor by which activation raises power above the machine's own idle draw
(source measurements: $k_{\dpdk}=1.73$, $k_{\usr}=0.37$). Rebasing onto this
deployment's per-process attributed idle power $P^{\mathrm{idle}}_{v}$ and the
software-stack activation duration $t_{\mathrm{act},v}$ gives
\begin{equation}
  E_{\mathrm{spike},v} \;=\;
  \frac{k_v \, P^{\mathrm{idle}}_{v} \, t_{\mathrm{act},v}}{3600}\quad[\text{Wh}]
  \label{eq:dt_spike}
\end{equation}
which is independent of $\lambda$. With $P^{\mathrm{idle}}_{\dpdk}=0.82$\,W,
$P^{\mathrm{idle}}_{\usr}=0.015$\,W, this yields
$E_{\mathrm{spike},\dpdk}\approx 9.5\times10^{-3}$\,Wh and
$E_{\mathrm{spike},\usr}\approx 5.2\times10^{-6}$\,Wh. The two-order-of-magnitude
asymmetry—\dpdk{} activation is a long, CPU-heavy driver and hugepage
initialisation, \usr{} a short container start—is preserved, whereas anchoring
to loaded power would have flattened it.
```

**Check after pasting:** any later reference to `\eqref{eq:dt_spike}` or the
words "steady-state power" in the switching context still reads correctly; the
`round_down`/`sub_step` accounting paragraphs below are unaffected (they use
$t_{\mathrm{act}}$, which is unchanged).

---

## 2. (already handled by you) Threshold caption

The figure caption for the threshold-derivation plot should read QoS limit
$149$, decision $\lambda_{\uparrow}=81$, $\lambda_{\downarrow}=61$ Mbps — you
were editing this when the switching fix landed. Listed here only so it is not
forgotten; remove if done.
