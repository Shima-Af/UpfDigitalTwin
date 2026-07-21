"""UPF Digital Twin — Interactive Streamlit Dashboard.

Run from the project root:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from upf_digital_twin.utils.config import load_configs
from upf_digital_twin.twin.digital_twin import DigitalTwin
from upf_digital_twin.twin.threshold_derivation import derive_thresholds, load_forecast_mae_gbps
from upf_digital_twin.twin.upf_profile import UPFResult
from upf_digital_twin.policies.hysteresis import HysteresisPolicy

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPF Digital Twin · Energy-Aware Orchestration",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme / CSS (Inter font, dark product look, hide Streamlit chrome) ──
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      .stApp {
          background: radial-gradient(1200px 600px at 15% -10%, #15233b 0%, #0b1220 55%) fixed;
          color: #e6edf6;
      }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
      header[data-testid="stHeader"] { background: transparent; height: 0; }
      .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px; }
      h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif; letter-spacing: -0.015em; font-weight: 600; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,.06); }
      .stTabs [data-baseweb="tab"] {
          height: 42px; padding: 0 18px; background: transparent; color: #93a6bd;
          font-weight: 500; font-size: 13.5px; border-radius: 9px 9px 0 0;
      }
      .stTabs [aria-selected="true"] {
          background: rgba(34,211,238,.10); color: #e6edf6; border-bottom: 2px solid #22d3ee;
      }
      [data-testid="stSidebar"] { background: #0c1424; border-right: 1px solid rgba(255,255,255,.06); }
      [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
      [data-testid="stMetric"] {
          background: #131c2e; border: 1px solid rgba(255,255,255,.07);
          border-radius: 12px; padding: 12px 16px;
      }
      [data-testid="stMetricLabel"] p { color: #93a6bd; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
      [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,.07); border-radius: 10px; background: rgba(255,255,255,.02); }
      .stButton > button {
          border-radius: 9px; border: 1px solid rgba(34,211,238,.35);
          background: rgba(34,211,238,.10); color: #cfeffb; font-weight: 600;
      }
      .stButton > button:hover { border-color: #22d3ee; background: rgba(34,211,238,.18); }
      [data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,.06); border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Color palette (chart traces) ─────────────────────────────────────────────
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

# ── Friendly naming: traffic regions + forecast lead times ───────────────────
# Demo-friendly names for the K traffic-similarity zones (Lyon areas).
REGION_NAMES = [
    "Presqu'île", "Part-Dieu", "Confluence", "Croix-Rousse", "Villeurbanne",
    "Gerland", "Vaise", "Bron", "Vénissieux", "Guillotière",
]

def region_label(k: int) -> str:
    return REGION_NAMES[k] if 0 <= k < len(REGION_NAMES) else f"Zone {k}"

def lead_label(h: int) -> str:
    mins = (h + 1) * int(time_step_min)
    if mins < 60:
        return f"{mins} min ahead"
    h_, m_ = divmod(mins, 60)
    return f"{h_} h ahead" if m_ == 0 else f"{h_} h {m_} min ahead"


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
    # No leading whitespace — Streamlit's markdown parser treats 4-space
    # indentation as a code block, which would print the raw HTML.
    return (
        f'<div style="background:rgba(128,128,128,0.08);border-radius:6px;padding:11px 13px;flex:1;min-width:0">'
        f'<p style="font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 3px">{label}</p>'
        f'<p style="font-size:20px;font-weight:500;margin:0;line-height:1.2;{color_style}">{value}</p>'
        f'<p style="font-size:10px;color:#888780;margin:2px 0 0">{sub}</p>'
        f'</div>'
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
def _srow(label: str, value: str, accent: str = "#e6edf6") -> str:
    return ("<div style='display:flex;justify-content:space-between;font-size:12px;"
            "padding:3px 0;color:#9fb0c3'>"
            f"<span>{label}</span><b style='color:{accent}'>{value}</b></div>")

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:.1rem 0 .2rem'>"
        "<div style='width:32px;height:32px;border-radius:9px;background:linear-gradient(135deg,#22d3ee,#2dd4bf);"
        "display:flex;align-items:center;justify-content:center;font-size:17px'>📡</div>"
        "<div><div style='font-size:15px;font-weight:700;line-height:1.05'>UPF Digital Twin</div>"
        "<div style='font-size:10px;color:#7e93aa;letter-spacing:.06em;text-transform:uppercase'>Energy-Aware Orchestration</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px;background:rgba(255,255,255,.07);margin:.8rem 0 .6rem'></div>", unsafe_allow_html=True)

    span_days = N * float(time_step_min) / 60 / 24
    st.markdown(
        "<div style='font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#6f8298;margin-bottom:4px'>Network</div>"
        + _srow("Service", str(fes.get("service", "?")))
        + _srow("Traffic regions", str(K))
        + _srow("Replay span", f"{span_days:.1f} days")
        + _srow("Resolution", f"{time_step_min} min"),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px;background:rgba(255,255,255,.07);margin:.7rem 0 .6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#6f8298;margin-bottom:4px'>Derived operating point</div>"
        + _srow("Decision", f"{spec.decision_gbps*1000:.0f} Mbps", "#22d3ee")
        + _srow("Energy break-even", f"{spec.energy_breakeven_gbps*1000:.0f} Mbps")
        + _srow("QoS limit", f"{spec.qos_limit_gbps*1000:.0f} Mbps")
        + _srow("Hysteresis band", f"{spec.hysteresis_band_gbps*1000:.0f} Mbps"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:10px;color:#6f8298;margin-top:10px;line-height:1.5'>"
        f"Surrogate-grounded · NetMob Lyon<br>{spec.derived_from}</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_demo, tab_orch, tab_curves, tab_clusters, tab_pareto, tab_model = st.tabs([
    "📊  Overview",
    "🛰️  Operations",
    "🔬  Twin Explorer",
    "🗺️  Regions",
    "⚖️  Trade-offs",
    "✅  Model quality",
])

# ═════════════════════════════════════════════════════════════════════════════
# Tab 0 — Investor demo: shadow-mode Ops Replay Cockpit
# ═════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown(
        "<div style='font-size:10px;letter-spacing:.12em;text-transform:uppercase;"
        "color:#888780;font-family:monospace'>UpfDigitalTwin · Shadow mode · NetMob Lyon</div>"
        "<div style='font-size:17px;font-weight:600;margin-bottom:2px'>"
        "Energy-Aware UPF Orchestration — Replay</div>",
        unsafe_allow_html=True,
    )

    # --- Controls ---
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        demo_ctrl = st.radio(
            "Controller", ["threshold", "hysteresis", "oracle"],
            format_func=lambda x: {
                "threshold":  "Threshold (deployable)",
                "hysteresis": "Hysteresis",
                "oracle":     "Oracle (perfect forecast)",
            }[x],
            index=0, horizontal=True, label_visibility="collapsed",
        )
    busiest = int(tgt_gbps[:, 0, :].mean(axis=0).argmax())
    with cc2:
        demo_cluster = st.selectbox("Region", list(range(K)), index=busiest,
                                    format_func=region_label, key="demo_cluster")
    with cc3:
        demo_horizon = st.selectbox("Forecast lead", list(range(H)), index=0,
                                    format_func=lead_label, key="demo_horizon")

    with st.expander("⚙︎ Fleet & cost assumptions (drive the annualized figures)"):
        a1, a2, a3 = st.columns(3)
        fleet_n  = a1.number_input("Fleet size (UPF instances)", 1, 1_000_000, 1000, step=100)
        eur_kwh  = a2.number_input("Electricity price (€/kWh)", 0.0, 2.0, 0.12, step=0.01, format="%.2f")
        gco2_kwh = a3.number_input("Grid intensity (gCO₂e/kWh)", 0.0, 1000.0, 50.0, step=10.0,
                                   help="France/Lyon grid is nuclear-heavy (~50). Higher elsewhere.")

    # --- Episodes: controller vs always-DPDK baseline (cached) ---
    ep      = run_episode(demo_ctrl, demo_cluster, demo_horizon)
    ep_dpdk = run_episode("static_dpdk", demo_cluster, demo_horizon)

    # --- Play / scrub ---
    ss = st.session_state
    ss.setdefault("demo_slider", N - 1)
    ss.setdefault("demo_play", False)
    pcol, scol, icol = st.columns([1, 1, 4])
    if pcol.button("⏸ Pause" if ss.demo_play else "▶ Play", use_container_width=True):
        ss.demo_play = not ss.demo_play
    speed = scol.selectbox("Speed", [1, 2, 4], index=1, label_visibility="collapsed")
    # Advance the cursor BEFORE the slider is instantiated (safe session_state write).
    if ss.demo_play:
        nxt = ss.demo_slider + max(1, (N // 120) * speed)
        if nxt >= N - 1:
            nxt, ss.demo_play = N - 1, False
        ss.demo_slider = nxt
    t = st.slider("scrub", 0, N - 1, key="demo_slider", label_visibility="collapsed")
    with icol:
        hh = (t * time_step_min) // 60
        st.markdown(
            f"<div style='font-size:11px;color:#888780;font-family:monospace;padding-top:8px'>"
            f"replaying slot <b>{t}</b>/{N - 1} · t+{hh // 24:.0f}d {hh % 24:.0f}h · "
            f"<span style='color:{COL_USR}'>shadow mode — no live actuation</span></div>",
            unsafe_allow_html=True,
        )

    # --- Cumulative metrics up to t ---
    m      = cumulative_metrics(ep, t)
    m_dpdk = cumulative_metrics(ep_dpdk, t)
    saved_pct = (m_dpdk["energy_wh"] - m["energy_wh"]) / m_dpdk["energy_wh"] * 100 if m_dpdk["energy_wh"] > 0 else 0.0
    sla_pct   = 100.0 - m["viol_pct"]
    sla_color = COL_USR if sla_pct >= 99.0 else COL_THRESH

    # Annualized fleet impact (EXTRAPOLATED): avg power saved/instance × 8760h × N
    sl = slice(0, t + 1)
    avg_saved_w   = float(ep_dpdk["power"][sl].mean() - ep["power"][sl].mean())
    kwh_yr_inst   = max(0.0, avg_saved_w) * 8760.0 / 1000.0
    fleet_kwh_yr  = kwh_yr_inst * fleet_n
    fleet_eur_yr  = fleet_kwh_yr * eur_kwh
    fleet_tco2_yr = fleet_kwh_yr * gco2_kwh / 1e6  # g → t

    # --- Hero KPI strip ---
    cards = "<div style='display:flex;gap:8px;margin:.4rem 0 .7rem'>"
    cards += metric_card("Energy saved",  f"{saved_pct:.0f}%",                "vs always-DPDK", color=COL_USR)
    cards += metric_card("SLA adherence", f"{sla_pct:.1f}%",                  "QoS ≥ 0.90", color=sla_color)
    cards += metric_card("Fleet energy",  f"{fleet_kwh_yr / 1000:,.0f} MWh/yr", f"@ {fleet_n:,} instances")
    cards += metric_card("OPEX saved",    f"€{fleet_eur_yr:,.0f}/yr",         f"@ €{eur_kwh:.2f}/kWh")
    cards += metric_card("CO₂ avoided",   f"{fleet_tco2_yr:,.0f} t/yr",       f"@ {gco2_kwh:.0f} gCO₂/kWh")
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)
    st.caption(
        "Annualized figures are **extrapolations** (representative cluster load × 8 760 h × fleet size). "
        "Energy saved, SLA, traffic and power are measured by the twin over real NetMob Lyon traffic."
    )

    # --- UPF configuration timeline strip ---
    st.markdown(
        "<div style='display:flex;gap:14px;margin:.2rem 0 4px;align-items:center'>"
        "<span style='font-size:10px;color:#888780;text-transform:uppercase;letter-spacing:.08em'>UPF configuration:</span>"
        f"<span style='font-size:11px;color:#888780'><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:{COL_DPDK};margin-right:4px'></span>DPDK</span>"
        f"<span style='font-size:11px;color:#888780'><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:{COL_USR};margin-right:4px'></span>USR</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(make_strip_html(ep["realised"], t, N), unsafe_allow_html=True)

    # --- Reveal helper (everything draws up to the cursor t) ---
    x = np.arange(N)
    past = x <= t

    def _reveal(arr):
        return np.where(past, arr, np.nan)

    # --- Traffic ---
    st.markdown(
        "<p style='font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:.6rem 0 3px'>Traffic load (Mbps)</p>",
        unsafe_allow_html=True,
    )
    f_traffic = go.Figure()
    f_traffic.add_trace(go.Scatter(x=x, y=_reveal(ep["actual_mbps"]), mode="lines",
                                   line=dict(color=COL_TRAFFIC, width=1.6), name="Traffic"))
    f_traffic.add_hline(y=spec.decision_gbps * 1000, line_dash="dash", line_color=COL_THRESH, line_width=1,
                        annotation_text=f"decision {spec.decision_gbps * 1000:.0f} Mbps",
                        annotation_position="top right",
                        annotation=dict(font_size=9, font_color=COL_THRESH))
    f_traffic.update_layout(margin=dict(l=10, r=10, t=4, b=4), height=140,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                            xaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", tickfont=dict(size=9, color="#888780")),
                            yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", tickfont=dict(size=9, color="#888780")))
    st.plotly_chart(f_traffic, use_container_width=True, config={"displayModeBar": False})

    # --- Power vs baseline (signature 'savings gap') + QoS ---
    pL, pR = st.columns([3, 2])
    with pL:
        st.markdown(
            "<p style='font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 3px'>Attributed power — shaded area = energy saved</p>",
            unsafe_allow_html=True,
        )
        f_pow = go.Figure()
        f_pow.add_trace(go.Scatter(x=x, y=_reveal(ep_dpdk["power"]), mode="lines",
                                   line=dict(color=COL_DPDK, width=1.3, dash="dot"), name="Always-DPDK baseline"))
        f_pow.add_trace(go.Scatter(x=x, y=_reveal(ep["power"]), mode="lines",
                                   line=dict(color=COL_POWER, width=1.8), name="Controller",
                                   fill="tonexty", fillcolor="rgba(29,158,117,0.20)"))
        f_pow.update_layout(margin=dict(l=10, r=10, t=4, b=4), height=160,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.18, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                            xaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", tickfont=dict(size=9, color="#888780")),
                            yaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", tickfont=dict(size=9, color="#888780")))
        st.plotly_chart(f_pow, use_container_width=True, config={"displayModeBar": False})
    with pR:
        st.markdown(
            f"<p style='font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#888780;margin:0 0 3px'>QoS compliance · <span style='color:{sla_color}'>SLA {sla_pct:.1f}%</span></p>",
            unsafe_allow_html=True,
        )
        f_qos = go.Figure()
        f_qos.add_trace(go.Scatter(x=x, y=_reveal(ep["qos"]), mode="lines",
                                   line=dict(color=COL_QOS, width=1.6), name="QoS"))
        f_qos.add_hline(y=0.9, line_dash="dash", line_color=COL_THRESH, line_width=1,
                        annotation_text="τ = 0.90", annotation_position="bottom right",
                        annotation=dict(font_size=9, font_color=COL_THRESH))
        f_qos.update_layout(margin=dict(l=10, r=10, t=4, b=4), height=160,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                            yaxis=dict(range=[0.5, 1.02], showgrid=True, gridcolor="rgba(136,135,128,.12)", tickfont=dict(size=9, color="#888780")),
                            xaxis=dict(showgrid=True, gridcolor="rgba(136,135,128,.12)", tickfont=dict(size=9, color="#888780")))
        st.plotly_chart(f_qos, use_container_width=True, config={"displayModeBar": False})

    # --- Honesty band: forecast vs hindsight ---
    oracle_dec = np.where(ep["actual_mbps"] / 1000.0 < spec.decision_gbps, "USR", "DPDK")
    disagree = ep["decisions"][:t + 1] != oracle_dec[:t + 1]
    n_dis = int(disagree.sum())
    held = int((disagree & (ep["qos"][:t + 1] >= 0.9)).sum())
    st.markdown(
        f"<div style='font-size:11px;color:#888780;border-left:3px solid {COL_USR};padding:6px 10px;margin:.4rem 0;background:rgba(136,135,128,.05)'>"
        f"<b>Forecast honesty:</b> the controller acts on a <i>forecast</i>. So far it disagreed with perfect "
        f"hindsight at <b>{n_dis}</b> of {t + 1} steps — the safety margin held QoS at <b>{held}/{n_dis}</b> of them. "
        f"Shadow-mode replay over real traffic, not a live network.</div>",
        unsafe_allow_html=True,
    )

    # --- How it works / provenance ---
    with st.expander("🛈 How it works · model provenance · path to live"):
        lite_r2 = float(np.mean([e["r2"] for e in manifest.values() if e.get("model_variant") == "lite"]))
        wape = next((r.get("test_wape") for r in fes.get("results", []) if int(r.get("K", -1)) == K), None)
        wape_str = f"{wape:.1f}%" if wape is not None else "n/a"
        st.markdown(
            "**Closed loop (today = step 1, shadow):** traffic forecast → digital-twin what-if "
            "(power / QoS / safety) → policy picks UPF mode → *(live: orchestrator actuates)*.\n\n"
            f"- Surrogate fidelity: mean **R² = {lite_r2:.3f}** across lite models · "
            f"forecast **WAPE = {wape_str}** at K={K}\n"
            "- Twin is **measurement-grounded** (UpfProfilingCampaign); traffic is **real NetMob Lyon**.\n"
            "- Path to live: shadow → human-in-the-loop → guarded closed loop "
            "(SLA circuit-breaker + auto-rollback)."
        )

    # --- Auto-advance the replay ---
    if ss.demo_play:
        time.sleep(0.08)
        st.rerun()

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
        cluster = st.selectbox("Region", list(range(K)), index=9, format_func=region_label)
    with sel2:
        horizon = st.selectbox("Forecast lead", list(range(H)), index=0, format_func=lead_label,
                                help="how far ahead the controller acts on the forecast")

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
# Twin Explorer — single-point what-if (folded in from the old Live tab)
# ═════════════════════════════════════════════════════════════════════════════
with tab_curves:
    st.markdown("##### Single-point what-if — set a load, compare DPDK vs USR")
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
    st.divider()
    st.markdown("##### Power / delay / loss vs offered load")
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
    st.markdown(f"**{len(bs_loc)} base stations** across Lyon, grouped into **{K} traffic regions** by demand similarity.")
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
        st.markdown("##### Region statistics")
        sizes = pd.Series({int(k): len(v) for k, v in bsmap.items()}, name="# BS").sort_index()
        peak  = pd.Series({i: cs_gbps[i].max() for i in range(K)}, name="Peak (Gbps)").sort_index()
        mean  = pd.Series({i: cs_gbps[i].mean() for i in range(K)}, name="Mean (Gbps)").sort_index()
        cstats = pd.concat([sizes, peak.round(3), mean.round(3)], axis=1)
        cstats.index = [region_label(int(i)) for i in cstats.index]
        cstats.index.name = "Region"
        st.dataframe(cstats, use_container_width=True)

    st.divider()
    st.markdown("##### Traffic time series (full historical data)")
    sel_cluster = st.selectbox("Region", list(range(K)), index=0, format_func=region_label, key="ck")
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
# Trade-offs — per-window what-if (folded in from the old Counterfactual tab)
# ═════════════════════════════════════════════════════════════════════════════
with tab_pareto:
    st.divider()
    st.markdown("##### Per-window what-if — what would each policy choose?")
    c1, c2, c3 = st.columns(3)
    with c1: cf_n = st.number_input("Window (sample)", 0, N - 1, 0)
    with c2: cf_h = st.selectbox("Forecast lead", list(range(H)), index=0, format_func=lead_label, key="cf_h")
    with c3: cf_k = st.selectbox("Region", list(range(K)), index=0, format_func=region_label, key="cf_k")

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
