"""UPF Digital Twin — Interactive Streamlit Dashboard.

Run from the project root:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_configs
from src.twin.digital_twin import DigitalTwin
from src.twin.threshold_derivation import derive_thresholds, load_forecast_mae_gbps
from src.twin.upf_profile import UPFResult
from src.policies.hysteresis import HysteresisPolicy

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="UPF Digital Twin", page_icon="📡", layout="wide")

# ── Color palette (matches mockup style) ─────────────────────────────────────
COL_DPDK     = "#378ADD"
COL_USR      = "#1D9E75"
COL_TRAFFIC  = "#BA7517"
COL_POWER    = "#D85A30"
COL_QOS      = "#7F77DD"
COL_THRESH   = "#E24B4A"

# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_twin():
    paths_cfg, scenario_cfg = load_configs(PROJECT_ROOT)
    twin = DigitalTwin(scenario_cfg, paths_cfg, PROJECT_ROOT)
    return twin, paths_cfg, scenario_cfg

@st.cache_data
def load_traffic_data(_paths_cfg):
    tf = _paths_cfg["traffic_forecaster"]
    pred = np.load(PROJECT_ROOT / tf["predictions"])
    tgt  = np.load(PROJECT_ROOT / tf["targets"])
    cs   = np.load(PROJECT_ROOT / tf["cluster_series"])
    ca   = pd.read_parquet(PROJECT_ROOT / tf["cluster_assignments"])
    bs   = pd.read_parquet(PROJECT_ROOT / tf["bs_locations"])
    with open(PROJECT_ROOT / tf["cluster_bs_map"]) as fh:
        bsmap = json.load(fh)
    with open(PROJECT_ROOT / tf["forecast_eval_summary"]) as fh:
        fes = json.load(fh)
    return pred, tgt, cs, ca, bs, bsmap, fes

@st.cache_data
def load_manifest(_paths_cfg):
    with open(PROJECT_ROOT / _paths_cfg["profiling_twin"]["manifest"]) as fh:
        return json.load(fh)

@st.cache_data
def precompute_curves(_twin_id, n_points: int = 500):
    twin, _, _ = load_twin()
    loads = np.linspace(0.001, 0.50, n_points)
    d = twin._profile.evaluate_batch("DPDK", loads)
    u = twin._profile.evaluate_batch("USR",  loads, alt_power_watts=d["power_watts"])
    d["is_efficient"] = d["power_watts"] < u["power_watts"]
    return loads, d, u

twin, paths_cfg, scenario_cfg = load_twin()
upf_cfg          = scenario_cfg["upf"]
threshold_cfg    = scenario_cfg["threshold"]
qos_cfg          = upf_cfg["qos_budget"]
delay_budget_us  = qos_cfg["delay_budget_us"]
max_loss_pkts    = qos_cfg["max_loss_pkts_per_interval"]
safety_margin_mbps = float(threshold_cfg["safety_margin_mbps"])
cooldown_steps     = int(threshold_cfg["cooldown_steps"])

alpha            = scenario_cfg["traffic"]["calibration"]["alpha_gbps_per_norm"]
time_step_min    = scenario_cfg["traffic"]["time_step_minutes"]

pred, tgt, cs, ca, bs_loc, bsmap, fes = load_traffic_data(paths_cfg)
manifest = load_manifest(paths_cfg)
N, H, K = pred.shape

forecast_mae_gbps = load_forecast_mae_gbps(fes, alpha, K)
spec = derive_thresholds(twin, safety_margin_mbps=safety_margin_mbps,
                         forecast_mae_gbps=forecast_mae_gbps)

pred_gbps = pred * alpha
tgt_gbps  = tgt  * alpha
cs_gbps   = cs   * alpha


# ── Helpers ──────────────────────────────────────────────────────────────────
def qos_score(delay_us: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Continuous QoS score in [0, 1]: 1 = perfect, 0 = budget exceeded."""
    delay_term = np.clip(1.0 - (delay_us / delay_budget_us), 0.0, 1.0)
    loss_term  = np.where(loss <= max_loss_pkts, 1.0, 0.5)
    return delay_term * loss_term


@st.cache_data(show_spinner=False)
def _twin_batch(cluster: int, horizon: int) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """Cache the heavy sklearn batch evaluation per (cluster, horizon).

    Shared across all 5 policies for the same episode.
    """
    actual_g = tgt_gbps[:, horizon, cluster].astype(np.float64)
    d = twin._profile.evaluate_batch("DPDK", actual_g)
    u = twin._profile.evaluate_batch("USR",  actual_g, alt_power_watts=d["power_watts"])
    return (d["power_watts"],     u["power_watts"],
            d["delay_us"],        u["delay_us"],
            d["predicted_loss"],  u["predicted_loss"])


@st.cache_data(show_spinner=False)
def run_episode(policy_name: str, cluster: int, horizon: int) -> dict:
    """Run one policy on one cluster's time series.

    Fast path: re-use the cached batch evaluation, then a tight Python
    loop on raw numpy lookups applies the switching-cost accounting.
    Cached across reruns by (policy, cluster, horizon).
    """
    actual_g   = tgt_gbps[:, horizon, cluster]
    forecast_g = pred_gbps[:, horizon, cluster]

    # 1) Decide actions (no twin calls)
    if policy_name == "static_dpdk":
        decisions = np.full(N, "DPDK")
    elif policy_name == "static_usr":
        decisions = np.full(N, "USR")
    elif policy_name == "threshold":
        decisions = np.where(forecast_g < spec.decision_gbps, "USR", "DPDK").astype(object)
    elif policy_name == "oracle":
        decisions = np.where(actual_g < spec.decision_gbps, "USR", "DPDK").astype(object)
    elif policy_name == "hysteresis":
        h = HysteresisPolicy(spec.t_up_gbps, spec.t_down_gbps, cooldown_steps)
        decisions = h.act_sequence(forecast_g)
    else:
        raise ValueError(policy_name)

    # 2) Pull cached batch arrays (one batch shared by all 5 policies)
    p_d, p_u, dl_d, dl_u, ls_d, ls_u = _twin_batch(cluster, horizon)

    # 3) Read twin parameters once (no attr lookups in loop)
    accounting   = twin.accounting
    step_s       = twin.step_s
    step_h       = twin.step_h
    prewarm_on   = twin.prewarm_enabled
    standby_w    = twin.prewarm_standby_w
    act_s_dpdk   = twin.activation_duration_s.get("DPDK", 0.0)
    act_s_usr    = twin.activation_duration_s.get("USR",  0.0)

    if accounting == "round_down":
        frac_old_dpdk = frac_old_usr = 0.0
    elif accounting == "round_up":
        frac_old_dpdk = frac_old_usr = 1.0
    else:  # sub_step
        frac_old_dpdk = min(act_s_dpdk / step_s, 1.0) if step_s > 0 else 0.0
        frac_old_usr  = min(act_s_usr  / step_s, 1.0) if step_s > 0 else 0.0

    # 5) Tight Python loop on primitives only
    power = np.zeros(N); switch = np.zeros(N)
    delay = np.zeros(N); loss = np.zeros(N)
    realised = np.empty(N, dtype=object); pending = np.zeros(N, dtype=bool)
    current = "DPDK"   # 0 = DPDK index, 1 = USR index pattern below

    for i in range(N):
        req = decisions[i]
        # Standby: USR running, prewarm enabled → small standby cost every step
        sw_energy = standby_w * step_h if (prewarm_on and current == "USR") else 0.0

        if req == current:
            # No switch — use current UPF's lookup
            if current == "DPDK":
                power[i]=p_d[i]; delay[i]=dl_d[i]; loss[i]=ls_d[i]
            else:
                power[i]=p_u[i]; delay[i]=dl_u[i]; loss[i]=ls_u[i]
            switch[i] = sw_energy
            realised[i] = current
            continue

        # Switch requested
        act_s    = act_s_dpdk if req == "DPDK" else act_s_usr
        frac_old = frac_old_dpdk if req == "DPDK" else frac_old_usr
        p_new    = p_d[i] if req == "DPDK" else p_u[i]
        p_old    = p_d[i] if current == "DPDK" else p_u[i]
        dl_new   = dl_d[i] if req == "DPDK" else dl_u[i]
        dl_old   = dl_d[i] if current == "DPDK" else dl_u[i]
        ls_new   = ls_d[i] if req == "DPDK" else ls_u[i]
        ls_old   = ls_d[i] if current == "DPDK" else ls_u[i]

        if frac_old >= 1.0:
            # Activation pending: old UPF serves full step, no spike yet
            power[i]    = p_old
            switch[i]   = sw_energy
            delay[i]    = dl_old
            loss[i]     = ls_old
            realised[i] = current
            pending[i]  = True
        else:
            # Sub-step / instant: weighted avg + one-time spike
            avg = frac_old * p_old + (1.0 - frac_old) * p_new
            bulk_delay = dl_new if frac_old < 0.5 else dl_old
            bulk_loss  = ls_new if ls_new > ls_old else ls_old
            spike = p_new * act_s / 3600.0
            power[i]    = avg
            switch[i]   = sw_energy + spike
            delay[i]    = bulk_delay
            loss[i]     = bulk_loss
            realised[i] = req
            current     = req

    return {
        "actual_mbps":   actual_g * 1000,
        "forecast_mbps": forecast_g * 1000,
        "decisions":     decisions,
        "realised":      realised,
        "power":         power,
        "switch":        switch,
        "delay":         delay,
        "loss":          loss,
        "qos":           qos_score(delay, loss),
        "pending":       pending,
    }


def cumulative_metrics(ep: dict, t: int) -> dict:
    """Compute up-to-slot-t metrics for the orchestration view."""
    sl = slice(0, t + 1)
    dt_h = time_step_min / 60
    energy_wh = float((ep["power"][sl] * dt_h).sum() + ep["switch"][sl].sum())

    # SEC: power per Mbps, averaged
    mbps = np.maximum(ep["actual_mbps"][sl], 1.0)  # avoid div-by-zero
    sec  = float((ep["power"][sl] / mbps).mean()) * 1000  # convert to mW/Mbps for readability

    # QoS violation rate (slots where qos < 0.9)
    viol = float((ep["qos"][sl] < 0.9).mean()) * 100

    # Switch count
    decisions = ep["realised"][sl]
    switches = int((decisions[1:] != decisions[:-1]).sum()) if len(decisions) > 1 else 0

    return {
        "energy_wh": energy_wh,
        "sec":       sec,
        "viol_pct":  viol,
        "switches":  switches,
    }


def make_strip_html(decisions: np.ndarray, t: int, n_total: int) -> str:
    """Build a horizontal colored strip showing UPF action over time as an SVG.

    Same-color consecutive slots are merged into a single rect so the SVG
    has at most a few dozen elements even with N=1000+ slots.
    """
    def _color(i: int) -> str:
        if i > t:
            return "rgba(136,135,128,.22)"
        return COL_DPDK if decisions[i] == "DPDK" else COL_USR

    # Run-length-encode the colored cells
    rects = []
    start = 0
    cur_color = _color(0)
    for i in range(1, n_total):
        c = _color(i)
        if c != cur_color:
            rects.append((start, i - start, cur_color))
            start = i
            cur_color = c
    rects.append((start, n_total - start, cur_color))

    svg_rects = "".join(
        f'<rect x="{x}" y="0" width="{w}" height="18" fill="{c}"/>'
        for x, w, c in rects
    )
    return (
        f'<svg viewBox="0 0 {n_total} 18" preserveAspectRatio="none" '
        f'style="width:100%;height:18px;border-radius:3px;background:rgba(136,135,128,.05);display:block">'
        f'{svg_rects}</svg>'
    )


def metric_card(label: str, value: str, sub: str = "", color: str = None) -> str:
    color_style = f"color:{color}" if color else ""
    return f"""
    <div style="background:rgba(128,128,128,0.08);border-radius:6px;padding:11px 13px;flex:1;min-width:0">
      <p style="font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 3px">{label}</p>
      <p style="font-size:20px;font-weight:500;margin:0;line-height:1.2;{color_style}">{value}</p>
      <p style="font-size:10px;color:#888780;margin:2px 0 0">{sub}</p>
    </div>
    """


# ── Sidebar (always visible) ─────────────────────────────────────────────────
st.sidebar.title("📡 UPF Digital Twin")
st.sidebar.markdown(f"**Service:** {fes.get('service','?')} · **K=**{K} · **N=**{N} · **H=**{H}")
st.sidebar.markdown(f"**α** = {alpha} Gbps/norm · **dt** = {time_step_min} min")

st.sidebar.divider()
st.sidebar.subheader("Derived thresholds")
st.sidebar.markdown(f"""
- Energy break-even: **{spec.energy_breakeven_gbps*1000:.1f}** Mbps
- QoS limit: **{spec.qos_limit_gbps*1000:.1f}** Mbps
- Decision T: **{spec.decision_gbps*1000:.1f}** Mbps
- T_up / T_down: {spec.t_up_gbps*1000:.1f} / {spec.t_down_gbps*1000:.1f} Mbps
- Hysteresis band: {spec.hysteresis_band_gbps*1000:.1f} Mbps
""")
st.sidebar.caption(spec.derived_from)

with st.sidebar.expander("ℹ️ Window / horizon"):
    st.markdown(f"""
- **K = {K}** clusters of base stations grouped by traffic similarity.
- **N = {N}** test windows. Each window starts at one time-step and is 1 sample of the test set.
- **H = {H}** forecast steps per window. h=0 = next step, h=3 = 1 hour ahead.
""")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_orch, tab_live, tab_curves, tab_clusters, tab_pareto, tab_cf, tab_model = st.tabs([
    "🎛️  Orchestration",
    "▶️  Live",
    "📈  Curves",
    "🌍  Clusters",
    "📐  Pareto",
    "🔁  Counterfactual",
    "🧪  Model quality",
])

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 — Orchestration (the showpiece view, inspired by the HTML mockup)
# ═════════════════════════════════════════════════════════════════════════════
with tab_orch:
    # Header strip: title left, controller toggle right
    head_left, head_right = st.columns([3, 2])
    with head_left:
        st.markdown(
            f"""<div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                        color:#888780;font-family:monospace;margin-bottom:3px">
                UpfDigitalTwin · NetMob Lyon
              </div>
              <div style="font-size:15px;font-weight:500">Energy-Aware Orchestration Simulator</div>""",
            unsafe_allow_html=True,
        )
    with head_right:
        controller = st.radio(
            "Controller",
            options=["threshold", "hysteresis", "oracle", "static_dpdk", "static_usr"],
            format_func=lambda x: {
                "threshold":   "Threshold",
                "hysteresis":  "Hysteresis",
                "oracle":      "Oracle",
                "static_dpdk": "Static DPDK",
                "static_usr":  "Static USR",
            }[x],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

    # Cluster + horizon selectors
    sel1, sel2, _ = st.columns([1, 1, 4])
    with sel1:
        cluster = st.selectbox("Cluster", list(range(K)), index=9)
    with sel2:
        horizon = st.selectbox("Horizon", list(range(H)), index=0,
                                help="0 = immediate next step")

    # Episode results are cached via @st.cache_data → instant on revisit
    eps = {
        name: run_episode(name, cluster, horizon)
        for name in ["static_dpdk", "static_usr", "threshold", "hysteresis", "oracle"]
    }
    ep      = eps[controller]
    ep_dpdk = eps["static_dpdk"]   # baseline for energy-saving badge

    # Time slider
    sli, info = st.columns([5, 1])
    with sli:
        t = st.slider("slot", 0, N - 1, N - 1, 1, label_visibility="collapsed")
    with info:
        st.markdown(
            f"<div style='font-size:11px;color:#888780;font-family:monospace;text-align:right;padding-top:6px'>"
            f"slot <b>{t}</b> / {N-1} · {(t * time_step_min) // 60:.0f}h{(t*time_step_min)%60:02.0f}m"
            f"</div>",
            unsafe_allow_html=True,
        )

    # KPI metric cards row
    m = cumulative_metrics(ep, t)
    m_dpdk = cumulative_metrics(ep_dpdk, t)
    cards_html = "<div style='display:flex;gap:8px;margin-bottom:.9rem'>"
    cards_html += metric_card("Total energy", f"{m['energy_wh']:.1f}", "Wh attributed")
    cards_html += metric_card("Mean SEC",      f"{m['sec']:.3f}",       "mW / Mbps")
    viol_color = "#E24B4A" if m["viol_pct"] > 1 else None
    cards_html += metric_card("QoS violations", f"{m['viol_pct']:.1f}%", "% slots (qos < 0.9)", color=viol_color)
    cards_html += metric_card("Mode switches",  f"{m['switches']}",       "reconfigurations")
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Energy-saving badge vs static DPDK
    if controller != "static_dpdk" and m_dpdk["energy_wh"] > 0:
        save_pct = (m_dpdk["energy_wh"] - m["energy_wh"]) / m_dpdk["energy_wh"] * 100
        if save_pct > 0.1:
            st.markdown(
                f"<div style='display:inline-block;font-size:11px;padding:3px 10px;"
                f"border-radius:6px;background:rgba(29,158,117,.15);color:#1D9E75;"
                f"margin-bottom:.7rem'>↓ {save_pct:.1f}% energy vs Static DPDK</div>",
                unsafe_allow_html=True,
            )

    # Configuration timeline strip
    st.markdown(
        "<div style='display:flex;gap:14px;margin-bottom:5px;align-items:center;flex-wrap:wrap'>"
        "<span style='font-size:10px;color:#888780;text-transform:uppercase;letter-spacing:.08em'>Configuration:</span>"
        f"<span style='font-size:11px;color:#888780'><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:{COL_DPDK};vertical-align:middle;margin-right:4px'></span>DPDK</span>"
        f"<span style='font-size:11px;color:#888780'><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:{COL_USR};vertical-align:middle;margin-right:4px'></span>USR</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(make_strip_html(ep["realised"], t, N), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Traffic chart (full width)
    def _line_chart(y, color, title, ymin=None, ymax=None,
                    threshold_y=None, threshold_label=None):
        fig = go.Figure()
        # Past portion (solid)
        x = np.arange(N)
        y_past = np.where(x <= t, y, np.nan)
        fig.add_trace(go.Scatter(
            x=x, y=y_past, mode="lines",
            line=dict(color=color, width=1.6), name=title, hoverinfo="x+y",
        ))
        if threshold_y is not None:
            fig.add_hline(y=threshold_y, line_dash="dash", line_color=COL_THRESH, line_width=1,
                           annotation_text=threshold_label, annotation_position="top right",
                           annotation=dict(font_size=9, font_color=COL_THRESH))
        fig.update_layout(
            margin=dict(l=10, r=10, t=4, b=4), height=130,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", zeroline=False,
                       tickfont=dict(size=9, color="#888780")),
            yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", zeroline=False,
                       tickfont=dict(size=9, color="#888780"),
                       range=[ymin, ymax] if (ymin is not None and ymax is not None) else None),
        )
        return fig

    st.markdown(
        f"<p style='font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 5px'>Traffic load (Mbps)</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _line_chart(ep["actual_mbps"], COL_TRAFFIC, "Traffic", 0, ep["actual_mbps"].max() * 1.1),
        use_container_width=True, config={"displayModeBar": False},
    )

    # Power + QoS side by side
    cL, cR = st.columns(2)
    with cL:
        st.markdown(
            f"<p style='font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 5px'>Attributed power (W)</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _line_chart(ep["power"], COL_POWER, "Power", 0, max(ep["power"].max() * 1.1, 1.5)),
            use_container_width=True, config={"displayModeBar": False},
        )
    with cR:
        st.markdown(
            f"<p style='font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 5px'>QoS compliance score</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _line_chart(ep["qos"], COL_QOS, "QoS", 0.5, 1.02,
                         threshold_y=0.9, threshold_label="τ = 0.90"),
            use_container_width=True, config={"displayModeBar": False},
        )

    # Per-policy comparison strip below
    st.divider()
    st.markdown("##### Per-policy comparison (this episode, this slot)")
    rows = []
    for name in ["static_dpdk", "static_usr", "threshold", "hysteresis", "oracle"]:
        mm = cumulative_metrics(eps[name], t)
        save = (m_dpdk["energy_wh"] - mm["energy_wh"]) / m_dpdk["energy_wh"] * 100 if m_dpdk["energy_wh"] > 0 else 0
        rows.append({
            "Policy":            name,
            "Energy (Wh)":       round(mm["energy_wh"], 2),
            "SEC (mW/Mbps)":     round(mm["sec"], 3),
            "QoS viol %":        round(mm["viol_pct"], 2),
            "Switches":          mm["switches"],
            "Save vs DPDK %":    round(save, 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — Live evaluation
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    load_gbps = st.slider("Offered load (Gbps)", 0.0, 0.50, 0.05, 0.001, format="%.3f")

    r_dpdk = twin.evaluate_action("DPDK", load_gbps)
    r_usr  = twin.evaluate_action("USR",  load_gbps)
    policy_action = "USR" if load_gbps < spec.decision_gbps else "DPDK"
    r_policy = r_dpdk if policy_action == "DPDK" else r_usr

    def _safe(b): return "✅ QoS preserved" if b else "⚠️ QoS violation"
    def _eff(b):  return "💚 Energy-efficient" if b else "💸 Wasteful"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### DPDK")
        st.metric("Power",      f"{r_dpdk.power_watts:.4f} W")
        st.metric("Throughput", f"{r_dpdk.throughput_gbps:.4f} Gbps")
        st.metric("Delay",      f"{r_dpdk.delay_us:.1f} µs")
        st.markdown(_safe(r_dpdk.is_safe));  st.markdown(_eff(r_dpdk.is_efficient))
    with c2:
        st.markdown("### USR")
        st.metric("Power",      f"{r_usr.power_watts:.4f} W",
                  delta=f"{r_usr.power_watts - r_dpdk.power_watts:+.4f} vs DPDK", delta_color="inverse")
        st.metric("Throughput", f"{r_usr.throughput_gbps:.4f} Gbps")
        st.metric("Delay",      f"{r_usr.delay_us:.1f} µs")
        st.markdown(_safe(r_usr.is_safe));   st.markdown(_eff(r_usr.is_efficient))
    with c3:
        st.markdown(f"### Threshold → **{policy_action}**")
        st.metric("Power",      f"{r_policy.power_watts:.4f} W")
        save_pct = (r_dpdk.power_watts - r_policy.power_watts) / r_dpdk.power_watts * 100
        st.metric("Saving vs DPDK", f"{save_pct:.1f}%")
        st.metric("Delay",      f"{r_policy.delay_us:.1f} µs")
        st.markdown(_safe(r_policy.is_safe)); st.markdown(_eff(r_policy.is_efficient))

# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — Curves
# ═════════════════════════════════════════════════════════════════════════════
with tab_curves:
    loads, b_dpdk, b_usr = precompute_curves(id(twin))

    policy_power = np.where(loads < spec.decision_gbps, b_usr["power_watts"], b_dpdk["power_watts"])
    df_p = pd.DataFrame({
        "Load (Gbps)": loads, "DPDK": b_dpdk["power_watts"],
        "USR": b_usr["power_watts"], "Threshold": policy_power,
    }).set_index("Load (Gbps)")
    st.markdown("**Power vs load**")
    st.line_chart(df_p)
    st.caption(
        f"Break-even at {spec.energy_breakeven_gbps*1000:.0f} Mbps · "
        f"QoS limit at {spec.qos_limit_gbps*1000:.0f} Mbps · "
        f"Decision T at {spec.decision_gbps*1000:.0f} Mbps"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Downlink delay (µs)**")
        df_d = pd.DataFrame({"Load (Gbps)": loads, "DPDK": b_dpdk["delay_us"], "USR": b_usr["delay_us"]}).set_index("Load (Gbps)")
        st.line_chart(df_d)
        st.caption(f"Delay budget: {delay_budget_us} µs")
    with c2:
        st.markdown("**Predicted USR packet loss delta**")
        df_l = pd.DataFrame({"Load (Gbps)": loads, "USR loss": b_usr["predicted_loss"]}).set_index("Load (Gbps)")
        st.line_chart(df_l)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 — Clusters
# ═════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.markdown(f"**{len(bs_loc)} base stations** in Lyon, grouped into **K={K}** traffic clusters.")
    bs_with_cluster = bs_loc.merge(ca, on="site_id", how="left").dropna()
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
               "#1abc9c", "#e67e22", "#34495e", "#f1c40f", "#7f8c8d"]
    bs_with_cluster["color"] = bs_with_cluster["cluster_id"].astype(int).map(lambda i: palette[i % 10])

    c_map, c_stats = st.columns([3, 2])
    with c_map:
        st.markdown("##### Geographic distribution")
        st.map(bs_with_cluster.rename(columns={"lat":"latitude","lon":"longitude"}),
               size=20, color="color", zoom=10)
    with c_stats:
        st.markdown("##### Cluster statistics")
        sizes = pd.Series({int(k): len(v) for k, v in bsmap.items()}, name="# BS").sort_index()
        peak  = pd.Series({i: cs_gbps[i].max() for i in range(K)}, name="Peak (Gbps)").sort_index()
        mean  = pd.Series({i: cs_gbps[i].mean() for i in range(K)}, name="Mean (Gbps)").sort_index()
        cstats = pd.concat([sizes, peak.round(3), mean.round(3)], axis=1)
        cstats.index.name = "Cluster"
        st.dataframe(cstats, use_container_width=True)

    st.divider()
    st.markdown("##### Traffic time series (full historical data)")
    sel_cluster = st.selectbox("Cluster", list(range(K)), index=0, key="ck")
    series = cs_gbps[sel_cluster]
    df_ts = pd.DataFrame({"Time step (15 min)": np.arange(len(series)), "Load (Gbps)": series}).set_index("Time step (15 min)")
    st.line_chart(df_ts)
    st.caption(f"{len(series)} time steps of 15 min = {len(series) * 15 / 60 / 24:.1f} days")

# ═════════════════════════════════════════════════════════════════════════════
# Tab 5 — Pareto
# ═════════════════════════════════════════════════════════════════════════════
with tab_pareto:
    st.markdown("Sweep the decision threshold across all test windows.")

    @st.cache_data
    def sweep_thresholds(_twin_id):
        thresholds = np.linspace(0.030, 0.20, 36)
        actual_flat   = tgt_gbps[:, 0, :].ravel()
        forecast_flat = pred_gbps[:, 0, :].ravel()
        d = twin._profile.evaluate_batch("DPDK", actual_flat)
        u = twin._profile.evaluate_batch("USR",  actual_flat, alt_power_watts=d["power_watts"])
        dt_h = time_step_min / 60
        baseline_energy = (d["power_watts"] * dt_h).sum()

        rows = []
        for thr in thresholds:
            mask_usr = forecast_flat < thr
            power = np.where(mask_usr, u["power_watts"], d["power_watts"])
            unsafe = np.where(mask_usr, ~u["is_safe"], False)
            energy = (power * dt_h).sum()
            rows.append({
                "threshold_mbps":   thr * 1000,
                "saving_pct":       (baseline_energy - energy) / baseline_energy * 100,
                "unsafe_usr_rate":  float(unsafe[mask_usr].mean()) if mask_usr.any() else 0.0,
                "usr_ratio":        float(mask_usr.mean()),
            })
        return pd.DataFrame(rows)

    sweep = sweep_thresholds(id(twin))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Energy saving (%) vs threshold**")
        st.line_chart(sweep.set_index("threshold_mbps")["saving_pct"])
        st.markdown("**Unsafe USR rate vs threshold**")
        st.line_chart(sweep.set_index("threshold_mbps")["unsafe_usr_rate"])
    with c2:
        st.markdown("**Pareto: saving vs unsafe rate**")
        st.scatter_chart(sweep, x="unsafe_usr_rate", y="saving_pct", size="threshold_mbps")
        st.caption("Top-left is best. Each dot's size = threshold value (Mbps).")

    st.markdown("**Sweep table**")
    st.dataframe(sweep.round(4), use_container_width=True, hide_index=True)
    st.caption(f"Current decision threshold (derived): **{spec.decision_gbps*1000:.1f} Mbps**")

# ═════════════════════════════════════════════════════════════════════════════
# Tab 6 — Counterfactual
# ═════════════════════════════════════════════════════════════════════════════
with tab_cf:
    st.markdown("Pick one window. See what each policy would choose.")
    c1, c2, c3 = st.columns(3)
    with c1: cf_n = st.number_input("Sample (window)", 0, N - 1, 0)
    with c2: cf_h = st.number_input("Horizon step",    0, H - 1, 0)
    with c3: cf_k = st.number_input("Cluster",         0, K - 1, 0)

    p = float(pred_gbps[cf_n, cf_h, cf_k])
    a = float(tgt_gbps[cf_n, cf_h, cf_k])

    st.metric("Predicted load (Gbps)", f"{p:.4f}")
    st.metric("Actual load (Gbps)",    f"{a:.4f}", delta=f"{a - p:+.4f} vs forecast")

    cf_thr = st.slider("Counterfactual threshold (Mbps)", 30.0, 200.0,
                        float(spec.decision_gbps * 1000), 1.0, format="%.1f", key="cf_thr")
    cf_thr_g = cf_thr / 1000.0

    rows = []
    for label, decided in [("Threshold (forecast)", p), ("Oracle (actual)", a)]:
        chosen = "USR" if decided < cf_thr_g else "DPDK"
        r = twin.evaluate_action(chosen, a)
        rows.append({"Policy": label, "Chooses": chosen, "Power (W)": round(r.power_watts, 4),
                     "Delay (µs)": round(r.delay_us, 1), "Safe?": "✅" if r.is_safe else "⚠️"})
    for label, fixed in [("Static DPDK", "DPDK"), ("Static USR", "USR")]:
        r = twin.evaluate_action(fixed, a)
        rows.append({"Policy": label, "Chooses": fixed, "Power (W)": round(r.power_watts, 4),
                     "Delay (µs)": round(r.delay_us, 1), "Safe?": "✅" if r.is_safe else "⚠️"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 7 — Model quality
# ═════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("Surrogate-model accuracy from `manifest.json`.")
    rows = []
    for key, entry in manifest.items():
        rows.append({
            "Variant": entry["variant"], "Layer": entry["layer"],
            "Target": entry["target"], "Model variant": entry["model_variant"],
            "Algorithm": entry["best_params"].get("model_type", "?"),
            "MAE": entry["mae"], "RMSE": entry["rmse"], "R²": entry["r2"],
        })
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1: only_lite = st.checkbox("Show only lite models (used by the twin)", value=True)
    with c2: layer_filter = st.selectbox("Layer", ["all", "layer1", "layer2"])
    view = df.copy()
    if only_lite: view = view[view["Model variant"] == "lite"]
    if layer_filter != "all": view = view[view["Layer"] == layer_filter]
    view = view.sort_values(["Variant", "Layer", "Target"])
    st.dataframe(view.round(4), use_container_width=True, hide_index=True)

    st.markdown("**Mean R² per layer × variant (lite models)**")
    bar_df = view.groupby(["Layer", "Variant"])["R²"].mean().reset_index()
    bar_df["Layer × Variant"] = bar_df["Layer"] + " · " + bar_df["Variant"]
    st.bar_chart(bar_df.set_index("Layer × Variant")["R²"])

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Twin source: [UpfProfilingCampaign](https://github.com/Shima-Af/UpfProfilingCampaign)  ·  "
    "Traffic: [UpfTrafficForecaster](https://github.com/Shima-Af/UpfTrafficForecaster/tree/feature/cluster-first-stgnn)"
)
