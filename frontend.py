import io
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# Import pipeline components from main backend
from backend import (
    AstronautHealthNet,
    explain_risk_score,
    predict_health_risk,
    predict_health_outputs,
    train_model,
    build_personal_baseline,
    calculate_baseline_deviation,
    detect_sensor_faults,
    validate_external_dataset,
    run_robustness_suite,
)

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="AstroHealth Mission Control",
    page_icon="👩‍🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# NASA SPACE APPS BRAND GUIDELINES CSS INJECTION
# ----------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --ah-bg: #030914;
        --ah-panel: #071321;
        --ah-panel-2: #0a192b;
        --ah-line: #183653;
        --ah-blue: #55b7ff;
        --ah-blue-2: #1685d8;
        --ah-cyan: #69e5ff;
        --ah-lime: #d8ff3e;
        --ah-white: #f4f8fc;
        --ah-muted: #8fa7bd;
        --ah-orange: #ffb84d;
        --ah-red: #ff5f67;
    }

    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
        color: var(--ah-white) !important;
        background: var(--ah-bg) !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 8%, rgba(22,133,216,.18), transparent 28%),
            radial-gradient(circle at 88% 12%, rgba(105,229,255,.08), transparent 24%),
            linear-gradient(180deg, #030914 0%, #05101d 52%, #030914 100%) !important;
        background-attachment: fixed !important;
    }

    /* Subtle mission-grid texture */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .18;
        background-image:
            linear-gradient(rgba(85,183,255,.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(85,183,255,.045) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: linear-gradient(to bottom, black, transparent 82%);
    }

    header[data-testid="stHeader"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: .045em !important;
        text-transform: uppercase !important;
        color: var(--ah-white) !important;
    }
    h1 { font-size: 2.35rem !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.15rem !important; }
    p, li, label { color: #d9e6f0; }

    /* Mission masthead */
    .ah-masthead {
        position: relative;
        overflow: hidden;
        margin: 4px 0 18px 0;
        padding: 20px 24px 18px 24px;
        border: 1px solid #1b4a70;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(7,19,33,.97), rgba(7,25,43,.90));
        box-shadow: 0 14px 34px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
    }
    .ah-masthead::after {
        content: '';
        position: absolute;
        right: -70px;
        top: -100px;
        width: 260px;
        height: 260px;
        border: 1px solid rgba(105,229,255,.16);
        border-radius: 50%;
        box-shadow: 0 0 0 22px rgba(105,229,255,.025), 0 0 0 44px rgba(105,229,255,.018);
    }
    .ah-kicker {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--ah-cyan);
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: .18em;
        text-transform: uppercase;
    }
    .ah-brand-row { display: flex; align-items: center; gap: 14px; margin: 6px 0 2px 0; }
    .ah-mark {
        width: 42px; height: 42px; border-radius: 50%;
        display: grid; place-items: center;
        border: 1px solid var(--ah-blue);
        background: radial-gradient(circle, #10375b 0%, #061321 68%);
        color: var(--ah-lime);
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: .82rem;
        box-shadow: 0 0 18px rgba(85,183,255,.16);
    }
    .ah-brand {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.1rem;
        line-height: 1;
        letter-spacing: .08em;
        font-weight: 700;
        color: #fff;
    }
    .ah-subbrand {
        color: var(--ah-muted);
        font-size: .78rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-top: 3px;
    }
    .ah-mission-line {
        margin-top: 15px;
        max-width: 920px;
        color: #dceaf5;
        font-size: .96rem;
        line-height: 1.55;
    }
    .ah-mission-line b { color: #fff; }
    .ah-status-strip {
        display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;
    }
    .ah-chip {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 5px 9px;
        border: 1px solid #234866;
        border-radius: 4px;
        background: rgba(4,14,25,.72);
        font-family: 'IBM Plex Mono', monospace;
        font-size: .67rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #a9bfd0;
    }
    .ah-chip .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ah-lime); box-shadow: 0 0 8px rgba(216,255,62,.7); }
    .ah-chip.blue .dot { background: var(--ah-cyan); box-shadow: 0 0 8px rgba(105,229,255,.6); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050e19 0%, #071321 100%) !important;
        border-right: 1px solid #15344e !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: var(--ah-blue) !important;
        font-size: .86rem !important;
        letter-spacing: .13em !important;
    }
    section[data-testid="stSidebar"] hr { border-color: #16334c !important; }
    section[data-testid="stSidebar"] .stCaption { color: #7891a7 !important; }

    /* Cards / metrics */
    [data-testid="stMetric"] {
        min-height: 105px;
        background: linear-gradient(145deg, rgba(9,25,43,.96), rgba(5,16,29,.94)) !important;
        border: 1px solid #173b59 !important;
        border-radius: 7px !important;
        padding: 13px 15px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.025), 0 8px 22px rgba(0,0,0,.18) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #7898b1 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .67rem !important;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: var(--ah-white) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
    }

    /* Controls */
    .stButton > button {
        min-height: 38px;
        background: linear-gradient(180deg, #dfff51 0%, #c8ef24 100%) !important;
        color: #06111c !important;
        border: 1px solid #e8ff88 !important;
        border-radius: 5px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .72rem !important;
        font-weight: 600 !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 14px rgba(216,255,62,.10) !important;
    }
    /* Streamlit can render button labels inside <p>/<span>; force dark text on the yellow controls. */
    .stButton > button p,
    .stButton > button span {
        color: #06111c !important;
    }
    .stButton > button:hover {
        background: #f3ffd0 !important;
        color: #03101b !important;
        border-color: #fff !important;
        transform: translateY(-1px);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea,
    input {
        background: #081827 !important;
        border-color: #1a405e !important;
        color: #eaf3f9 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        border-bottom: 1px solid #183650;
        padding: 0 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #7691a8 !important;
        border-radius: 5px 5px 0 0 !important;
        padding: 10px 14px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .68rem !important;
        letter-spacing: .045em;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #dceaf5 !important; background: rgba(22,133,216,.08) !important; }
    .stTabs [aria-selected="true"] {
        color: var(--ah-lime) !important;
        background: rgba(22,133,216,.10) !important;
        box-shadow: inset 0 -2px 0 var(--ah-lime);
    }

    /* Expanders / alerts */
    [data-testid="stExpander"] {
        border: 1px solid #173b59 !important;
        background: rgba(6,18,31,.76) !important;
        border-radius: 7px !important;
    }
    div[data-baseweb="notification"] {
        background: #071726 !important;
        border: 1px solid #204766 !important;
    }
    .stAlert { border-radius: 6px !important; }

    /* Uploaders */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(7,20,34,.75) !important;
        border: 1px dashed #285270 !important;
        border-radius: 6px !important;
    }

    /* Code / diagnostics */
    code, pre { font-family: 'IBM Plex Mono', monospace !important; }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #173b59 !important;
        border-radius: 6px !important;
        overflow: hidden;
    }

    /* Footer */
    .ah-footer {
        margin-top: 26px;
        padding: 12px 0 6px 0;
        border-top: 1px solid #15334c;
        color: #607a91;
        font-family: 'IBM Plex Mono', monospace;
        font-size: .64rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        display: flex; justify-content: space-between; gap: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="ah-masthead">
        <div class="ah-kicker">ASTROHEALTH // MISSION OPERATIONS SYSTEM</div>
        <div class="ah-brand-row">
            <div class="ah-mark">AH</div>
            <div>
                <div class="ah-brand">ASTROHEALTH</div>
                <div class="ah-subbrand">Fleet Analytics &amp; Operations</div>
            </div>
        </div>
        <div class="ah-mission-line">
            <b>When astronauts cannot wait for Earth, AstroHealth detects deterioration, predicts what comes next, and supports action locally.</b>
        </div>
        <div class="ah-status-strip">
            <span class="ah-chip"><span class="dot"></span>EDGE INFERENCE ONLINE</span>
            <span class="ah-chip blue"><span class="dot"></span>TELEMETRY PIPELINE READY</span>
            <span class="ah-chip">DECISION SUPPORT // HUMAN OVERSIGHT</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    model, mean, scale, validation_metrics = train_model()
    df_nom = pd.read_csv("nominal_telemetry.csv")
    df_warn = pd.read_csv("warning_telemetry.csv")
    df_crit = pd.read_csv("critical_telemetry.csv")
    # Add explicit status labels
    df_nom["y_status"] = "nominal"
    df_warn["y_status"] = "warning"
    df_crit["y_status"] = "critical"
    pool = pd.concat(
        [
            df_nom,
            df_warn,
            df_crit
        ],
        ignore_index=True
    )
    return (
        model,
        mean,
        scale,
        pool,
        validation_metrics
    )


model, mean, scale, telemetry_pool, validation_metrics = load_pipeline()

# ----------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ----------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = {
        "tick": [],
        "risk": [],
        "heart_rate": [],
        "hrv": [],
        "spo2": [],
        "resp_rate": [],
        "core_temp": [],
        "ambient_co2": [],
        "pressure": [],
        "rad_rate": [],
        "sleep_debt": [],
    }
if "incident_logs" not in st.session_state:
    st.session_state["incident_logs"] = []
if "active_countermeasure" not in st.session_state:
    st.session_state["active_countermeasure"] = None
if "replay_data" not in st.session_state:
    st.session_state["replay_data"] = None
if "latest_sample" not in st.session_state:
    st.session_state["latest_sample"] = None
if "latest_score" not in st.session_state:
    st.session_state["latest_score"] = None
if "personal_baseline" not in st.session_state:
    st.session_state["personal_baseline"] = None
if "baseline_deviation_score" not in st.session_state:
    st.session_state["baseline_deviation_score"] = None
if "sensor_diagnostics" not in st.session_state:
    st.session_state["sensor_diagnostics"] = None
if "communication_blackout" not in st.session_state:
    st.session_state["communication_blackout"] = False
if "pending_sync_events" not in st.session_state:
    st.session_state["pending_sync_events"] = 0
if "last_sync_count" not in st.session_state:
    st.session_state["last_sync_count"] = 0
if "countermeasure_history" not in st.session_state:
    st.session_state["countermeasure_history"] = []
if "last_countermeasure_effect" not in st.session_state:
    st.session_state["last_countermeasure_effect"] = None
if "tick_speed" not in st.session_state:
    st.session_state["tick_speed"] = 0.5
if "latest_model_outputs" not in st.session_state:
    st.session_state["latest_model_outputs"] = None
if "trajectory_forecast" not in st.session_state:
    st.session_state["trajectory_forecast"] = None
if "external_validation_result" not in st.session_state:
    st.session_state["external_validation_result"] = None
if "robustness_result" not in st.session_state:
    st.session_state["robustness_result"] = None
if "base_sample" not in st.session_state:
    st.session_state["base_sample"] = None
if "mission_mode" not in st.session_state:
    st.session_state["mission_mode"] = "Standard IVA"
if "eva_o2_fraction" not in st.session_state:
    st.session_state["eva_o2_fraction"] = 1.00
if "real_data_sanity_result" not in st.session_state:
    st.session_state["real_data_sanity_result"] = None
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False


def render_digital_twin_status(sample):
    baseline = st.session_state.get(
        "personal_baseline"
    )
    if baseline is None:
        return
    deviation_score, drivers = (
        calculate_baseline_deviation(
            sample,
            baseline
        )
    )
    st.session_state.baseline_deviation_score = (
        deviation_score
    )
    driver_text = ", ".join(
        f"{name}: {value:.0f}%"
        for name, value in drivers
    )
    digital_twin_placeholder.info(
        f"🧬 **PERSONAL DIGITAL TWIN** | "
        f"Baseline Deviation: **{deviation_score:.1f}/100** | "
        f"Top deviations: {driver_text}"
    )


def render_sensor_health():
    result = st.session_state.get(
        "sensor_diagnostics"
    )
    if result is None:
        return
    if result["overall_status"] == "FAULT SUSPECTED":
        suspect_text = ", ".join(
            result["suspect_features"]
        )
        sensor_health_placeholder.warning(
            f"🛰️ **SENSOR INTEGRITY ALERT** | "
            f"Suspected channel(s): **{suspect_text}** | "
            f"Do not interpret the affected channel blindly."
        )
    else:
        sensor_health_placeholder.success(
            "🛰️ **SENSOR INTEGRITY:** "
            "All monitored channels currently consistent."
        )


def apply_sensor_fault(
    sample,
    baseline,
    fault_choice,
    tick
):
    """
    Inject simulated sensor failures for
    engineering fault-detection testing.
    """

    modified = sample.copy()
    if fault_choice == "Pressure Sensor Failure":
        modified["pressure"] = (
            baseline["pressure"]
            + np.random.normal(0, 5.0)
        )
    elif fault_choice == "SpO2 Sensor Dropout":
        # Simulate a missing channel. The model path below uses an
        # explicit hold-last-baseline value so the fault monitor can
        # distinguish missing telemetry from a physiological change.
        modified["spo2"] = np.nan
    elif fault_choice == "Heart Rate Sensor Drift":
        modified["heart_rate"] = (
            baseline["heart_rate"]
            + (tick * 2.5)
        )
    elif fault_choice == "CO2 Sensor Stuck":
        modified["ambient_co2"] = (
            baseline["ambient_co2"]
        )
    return modified


# ----------------------------------------------------
# 3. MISSION ANALYTICS HELPERS
# ----------------------------------------------------
FEATURES = [
    "heart_rate", "hrv", "spo2", "resp_rate", "core_temp",
    "ambient_co2", "pressure", "rad_rate", "sleep_debt",
]


def safe_sample_for_model(sample):
    """Return a finite telemetry sample suitable for model inference."""
    cleaned = dict(sample)
    for feature in FEATURES:
        value = cleaned.get(feature, np.nan)
        if not np.isfinite(value):
            baseline = st.session_state.get("personal_baseline") or {}
            fallback = baseline.get(feature, 0.0)
            cleaned[feature] = float(fallback)
        else:
            cleaned[feature] = float(value)
    return cleaned


def compute_model_outputs(sample):
    """Run the dual-head model and expose its agreement/consensus."""
    clean = safe_sample_for_model(sample)
    outputs = predict_health_outputs(clean, model, mean, scale)
    st.session_state.latest_model_outputs = outputs
    return outputs


def render_model_consensus(outputs, placeholder=None):
    """Display risk-head vs classification-head agreement."""
    target = placeholder or consensus_placeholder
    agreement = outputs.get("model_agreement", False)
    risk_status = outputs.get("risk_status", "UNKNOWN")
    class_status = outputs.get("classification_status", "UNKNOWN")
    confidence = outputs.get("confidence", 0.0)
    if agreement:
        target.success(
            f"🧠 **DUAL-HEAD CONSENSUS: AGREEMENT** | "
            f"Risk head: **{risk_status}** | Classifier: **{class_status}** | "
            f"Confidence: **{confidence:.1f}%**"
        )
    else:
        target.warning(
            f"🧠 **DUAL-HEAD CONSENSUS: DISAGREEMENT** | "
            f"Risk head: **{risk_status}** | Classifier: **{class_status}** | "
            f"Confidence: **{confidence:.1f}%** | Review telemetry + sensor integrity."
        )


def calculate_trajectory_forecast(history, horizon=6):
    """Project near-term risk from recent mission history."""
    ticks = np.asarray(history.get("tick", []), dtype=float)
    risks = np.asarray(history.get("risk", []), dtype=float)
    if len(risks) < 3:
        return None
    window = min(len(risks), 8)
    x = ticks[-window:]
    y = risks[-window:]
    slope, intercept = np.polyfit(x, y, 1)
    future_ticks = np.arange(int(ticks[-1]) + 1, int(ticks[-1]) + horizon + 1)
    projected = np.clip(slope * future_ticks + intercept, 0, 100)
    return {
        "ticks": future_ticks.tolist(),
        "risk": projected.tolist(),
        "slope": float(slope),
        "direction": "RISING" if slope > 0.5 else "FALLING" if slope < -0.5 else "STABLE",
    }


def render_trajectory_forecast(history):
    forecast = calculate_trajectory_forecast(history)
    st.session_state.trajectory_forecast = forecast
    if forecast is None:
        trajectory_placeholder.info(
            "🔮 **HEALTH TRAJECTORY:** Collect at least 3 telemetry ticks to generate a near-term engineering projection."
        )
        return
    last_tick = history["tick"][-1]
    last_risk = history["risk"][-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["tick"], y=history["risk"],
        name="Observed Risk", mode="lines+markers",
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[last_tick] + forecast["ticks"],
        y=[last_risk] + forecast["risk"],
        name="Projected Risk",
        line=dict(dash="dash", width=3),
    ))
    fig.add_hline(y=25, line_dash="dot", annotation_text="Warning threshold")
    fig.add_hline(y=55, line_dash="dot", annotation_text="Critical threshold")
    fig.update_layout(
        title=f"Near-Term Risk Trajectory — {forecast['direction']}",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,23,63,0.45)",
        font=dict(color="#FFFFFF", family="Overpass"),
        hovermode="x unified",
        yaxis=dict(range=[0, 100], title="Risk Score"),
        xaxis_title="Mission Tick",
    )
    trajectory_placeholder.plotly_chart(
        fig, use_container_width=True,
        key=f"trajectory_{last_tick}_{len(history['tick'])}"
    )
    trajectory_placeholder.caption(
        f"Engineering projection: {forecast['direction']} trend, slope {forecast['slope']:+.2f} risk points/tick. "
        "Not a clinical prediction or flight certification."
    )


def apply_countermeasure_to_sample(sample, countermeasure):
    """Apply the simulated closed-loop response to one telemetry sample."""
    result = safe_sample_for_model(sample)
    if countermeasure == "CO2_FLUSH":
        result["ambient_co2"] = max(result["ambient_co2"] - 180, 400)
        result["resp_rate"] = max(result["resp_rate"] - 0.25, 12)
    elif countermeasure == "O2_BOOST":
        result["spo2"] = min(result["spo2"] + 0.8, 99.5)
    elif countermeasure == "RAD_SHIELD":
        result["rad_rate"] = max(result["rad_rate"] - 10, 5)
    return result


def record_countermeasure(name, before_score, after_score):
    effect = float(before_score - after_score)
    entry = {
        "Countermeasure": name,
        "Before Risk": round(float(before_score), 1),
        "After Risk": round(float(after_score), 1),
        "Risk Change": round(effect, 1),
        "Outcome": "IMPROVED" if effect > 0.5 else "NO SIGNIFICANT CHANGE" if effect >= -0.5 else "WORSENED",
        "Tick": st.session_state.history["tick"][-1] if st.session_state.history["tick"] else None,
    }
    st.session_state.countermeasure_history.append(entry)
    st.session_state.last_countermeasure_effect = entry


def append_incident_log(tick, astronaut, sample, outputs, explanation, sensor_result=None, anomaly="None", countermeasure=None, latency_ms=None):
    st.session_state.incident_logs.append({
        "Tick": tick,
        "Astronaut": astronaut,
        "Anomaly Scenario": anomaly,
        "Countermeasure": countermeasure or "None",
        "Status": outputs["risk_status"],
        "Classification": outputs["classification_status"],
        "Model Agreement": bool(outputs["model_agreement"]),
        "Confidence (%)": round(outputs["confidence"], 1),
        "Risk Score": round(outputs["risk_score"], 1),
        "Sensor Integrity": sensor_result["overall_status"] if sensor_result else "NOT CHECKED",
        "Suspect Sensors": ", ".join(sensor_result.get("suspect_features", [])) if sensor_result else "",
        "Primary Driver": explanation,
        "HR (bpm)": int(sample["heart_rate"]),
        "HRV (ms)": round(sample["hrv"], 1),
        "SpO2 (%)": round(sample["spo2"], 1),
        "Resp Rate": round(sample["resp_rate"], 1),
        "Core Temp (°C)": round(sample["core_temp"], 1),
        "CO2 (ppm)": int(sample["ambient_co2"]),
        "Pressure (kPa)": round(sample["pressure"], 1),
        "Radiation (uSv/h)": round(sample["rad_rate"], 1),
        "Sleep Debt (h)": round(sample["sleep_debt"], 1),
        "Inference Latency (ms)": round(latency_ms, 2) if latency_ms is not None else None,
        "Earth Link": "LOST / AUTONOMOUS" if st.session_state.communication_blackout else "CONNECTED",
    })


# ----------------------------------------------------
# 4. TRUST, EVA & REAL-DATA HELPERS
# ----------------------------------------------------
NASA_SOURCES = {
    "NASA-STD-3001 Vol. 2": "https://www.nasa.gov/reference/nasa-std-3001v2/",
    "NASA Vol. 2 — Natural & Induced Environments": "https://www.nasa.gov/reference/6-0-natural-and-induced-environments-vol-2/",
    "NASA Vol. 2 — Spacesuits": "https://www.nasa.gov/reference/11-0-spacesuits-vol-2/",
    "NASA OSDR": "https://osdr.nasa.gov/",
    "PhysioNet": "https://physionet.org/",
}

REAL_DATA_ALIASES = {
    "heart_rate": ["heart_rate", "hr", "HR", "Heart Rate", "HeartRate"],
    "spo2": ["spo2", "SpO2", "oxygen_saturation", "oxygen saturation", "O2Sat"],
    "resp_rate": ["resp_rate", "respiratory_rate", "respiratory rate", "RR", "RespRate"],
    "core_temp": ["core_temp", "temperature", "temp", "Temperature", "Temp"],
}


def find_real_data_column(df, aliases):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def real_data_sanity_check(df):
    """Descriptive sanity check only; does not claim model validation."""
    rows = []
    for feature, aliases in REAL_DATA_ALIASES.items():
        column = find_real_data_column(df, aliases)
        if column is None:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        rows.append({
            "AstroHealth Feature": feature,
            "Source Column": str(column),
            "Valid Samples": int(len(numeric)),
            "Median": round(float(numeric.median()), 3),
            "Minimum": round(float(numeric.min()), 3),
            "Maximum": round(float(numeric.max()), 3),
        })
    return pd.DataFrame(rows)


def eva_screening(sample, o2_fraction):
    """EVA atmosphere screening using NASA-published context, not certification limits."""
    pressure_kpa = float(sample["pressure"])
    co2_ppm = max(float(sample["ambient_co2"]), 0.0)
    pp_o2_mmhg = pressure_kpa * 7.50062 * float(o2_fraction)
    pp_co2_mmhg = pressure_kpa * 7.50062 * (co2_ppm / 1_000_000.0)

    # Current NASA Vol. 2 text discusses guiding PIO2 values of 127–333 mmHg.
    # These are used here only as a transparent engineering screening band.
    pio2_flag = "SCREENING FLAG" if not (
        127.0 <= pp_o2_mmhg <= 333.0) else "WITHIN SCREENING BAND"
    pressure_flag = "REVIEW" if not (
        34.5 < pressure_kpa <= 103.0) else "WITHIN GENERAL PRESSURE BAND"
    return {
        "pressure_kpa": pressure_kpa,
        "pp_o2_mmhg": pp_o2_mmhg,
        "pp_co2_mmhg": pp_co2_mmhg,
        "pio2_flag": pio2_flag,
        "pressure_flag": pressure_flag,
    }


def render_eva_monitor(sample):
    if st.session_state.mission_mode != "EVA Specialist":
        return
    result = eva_screening(sample, st.session_state.eva_o2_fraction)
    st.markdown("### 🧑‍🚀 EVA Atmospheric Monitor")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suit / Local Pressure", f"{result['pressure_kpa']:.1f} kPa")
    c2.metric("Calculated ppO₂", f"{result['pp_o2_mmhg']:.0f} mmHg")
    c3.metric("Calculated ppCO₂", f"{result['pp_co2_mmhg']:.2f} mmHg")
    c4.metric("O₂ Fraction", f"{st.session_state.eva_o2_fraction * 100:.0f}%")
    if result["pio2_flag"] == "SCREENING FLAG":
        st.warning(
            f"⚠️ EVA ppO₂ screening flag: {result['pp_o2_mmhg']:.0f} mmHg")
    else:
        st.success(f"✓ EVA ppO₂ screening: {result['pp_o2_mmhg']:.0f} mmHg")
    st.caption(
        "EVA monitor follows the current NASA-STD-3001 Vol. 2 emphasis on recording and alerting for suit pressure, ppO₂ and ppCO₂. "
        "The 127–333 mmHg PIO₂ values are used only as an engineering screening band from NASA's explanatory text; "
        "this app is not a flight certification system."
    )


def render_ethics_privacy():
    with st.expander("🛡️ Ethics, Privacy & Human Oversight"):
        st.markdown(
            """
            **AstroHealth is decision support, not an autonomous medical authority.**

            - Telemetry is processed locally in this demonstration wherever possible.
            - The system reports risk estimates, sensor-quality warnings and simulated countermeasure effects.
            - It does **not** diagnose disease or replace qualified crew, flight surgeons or mission controllers.
            - Synthetic telemetry is clearly distinguished from external research data.
            - External datasets should be used only under their own licenses, access rules and de-identification requirements.
            """
        )


# ----------------------------------------------------
# 4. SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:4px 0 16px 0; border-bottom:1px solid #16334c; margin-bottom:16px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.16em; color:#69e5ff;">AHMC // FLIGHT DESK</div>
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.35rem; letter-spacing:.08em; color:#f4f8fc; margin-top:3px;">MISSION CONTROL</div>
        <div style="font-size:.66rem; color:#6f899f; margin-top:3px;">LOCAL EDGE OPERATIONS CONSOLE</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### 01 // MISSION PROFILE")
mission_mode = st.sidebar.selectbox(
    "Operational Mode",
    ["Standard IVA", "EVA Specialist"],
    key="mission_mode",
)
if mission_mode == "EVA Specialist":
    st.sidebar.slider(
        "EVA O₂ Fraction",
        min_value=0.21,
        max_value=1.00,
        value=1.00,
        step=0.01,
        key="eva_o2_fraction",
        help="Used only for the EVA atmospheric screening overlay; it is not a neural-network input."
    )
st.sidebar.markdown("### 02 // CREW FLEET")
selected_astronaut = st.sidebar.selectbox(
    "Active Crew Member",
    [
        "Cmdr. Vance (EVA Specialist)",
        "Dr. Aris (Payload Specialist)",
        "Lt. Kovacs (Flight Engineer)",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("### 03 // STREAM CONTROL")
total_ticks = st.sidebar.slider(
    "Total Ticks", min_value=10, max_value=50, value=20)
tick_speed = st.sidebar.slider(
    "Tick Interval (s)", min_value=0.1, max_value=2.0, value=0.5
)
st.sidebar.divider()
st.sidebar.markdown("### 04 // ANOMALY INJECTION")
anomaly_choice = st.sidebar.selectbox(
    "Force Telemetry Anomaly",
    [
        "None (Nominal Operation)",
        "Cabin Depressurization",
        "CO2 Scrubber Failure",
        "Solar Particle Event (High Radiation)",
        "Extreme Heat Exposure",
    ],
)
sensor_fault_choice = st.sidebar.selectbox(
    "Sensor Integrity Scenario",
    [
        "No Sensor Fault",
        "Pressure Sensor Failure",
        "SpO2 Sensor Dropout",
        "Heart Rate Sensor Drift",
        "CO2 Sensor Stuck",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("### 05 // EARTH LINK")
if st.sidebar.button(
    "TOGGLE EARTH LINK",
    key="toggle_earth_link"
):
    was_blackout = (
        st.session_state.communication_blackout
    )
    st.session_state.communication_blackout = (
        not was_blackout
    )
    # Earth link restored
    if (
        was_blackout
        and not st.session_state.communication_blackout
    ):
        st.session_state.last_sync_count = (
            st.session_state.pending_sync_events
        )
        st.session_state.pending_sync_events = 0
st.sidebar.caption(
    "Local AI inference continues during communications loss."
)
st.session_state.tick_speed = tick_speed
start_stream = st.sidebar.button("START TELEMETRY STREAM")
run_demo = st.sidebar.button("RUN JUDGE DEMO SEQUENCE", key="run_demo")
st.sidebar.divider()
st.sidebar.markdown("### 06 // MISSION REPLAY")
uploaded_file = st.sidebar.file_uploader(
    "Upload Telemetry Session File", type=["json", "csv"]
)
if uploaded_file is not None:
    if uploaded_file.name.endswith(".json"):
        st.session_state.replay_data = pd.read_json(uploaded_file)
    else:
        st.session_state.replay_data = pd.read_csv(uploaded_file)
    st.sidebar.success(
        f"Loaded {len(st.session_state.replay_data)} ticks from log."
    )
start_replay = st.sidebar.button("PLAYBACK MISSION SESSION")

# ----------------------------------------------------
# 4. DASHBOARD TABS
# ----------------------------------------------------
tab_live, tab_impact, tab_countermeasures, tab_logs, tab_diagnostics = st.tabs([
    "01 // LIVE TELEMETRY",
    "02 // SYSTEM IMPACT",
    "03 // COUNTERMEASURES",
    "04 // INCIDENT LOGS",
    "05 // EDGE DIAGNOSTICS",
])

# --- TAB 1: LIVE TELEMETRY UI ---
with tab_live:
    status_alert = st.empty()
    communication_placeholder = st.empty()
    explain_box = st.empty()
    consensus_placeholder = st.empty()
    audio_placeholder = st.empty()
    st.subheader(f"LIVE TELEMETRY // {selected_astronaut}")
    with st.container():
        r1_cols = st.columns(5)
        card_risk = r1_cols[0].empty()
        card_hr = r1_cols[1].empty()
        card_hrv = r1_cols[2].empty()
        card_spo2 = r1_cols[3].empty()
        card_resp = r1_cols[4].empty()
        r2_cols = st.columns(5)
        card_temp = r2_cols[0].empty()
        card_co2 = r2_cols[1].empty()
        card_press = r2_cols[2].empty()
        card_rad = r2_cols[3].empty()
        card_sleep = r2_cols[4].empty()
    st.subheader("📈 Multi-Parameter Tracking & Trend Forecast")
    chart_placeholder = st.empty()
    trajectory_placeholder = st.empty()
    digital_twin_placeholder = st.empty()

# --- TAB 2: SYSTEM IMPACT MAP ---
with tab_impact:
    st.subheader("🎯 Physiological & Environmental Domain Impact Map")
    st.write(
        "Real-time assessment of system-level stress across major physiological"
        " and environmental vectors."
    )
    radar_chart_placeholder = st.empty()
    sensor_health_placeholder = st.empty()
    eva_placeholder = st.empty()

# --- TAB 3: CLOSED-LOOP COUNTERMEASURES ---
with tab_countermeasures:
    st.subheader("🛠️ Active Mission Subsystem Control")
    st.caption(
        "Simulated closed-loop controls modify the current telemetry state and immediately re-score it. "
        "This is an engineering demonstration, not a medical treatment recommendation."
    )
    c1, c2, c3 = st.columns(3)
    countermeasure_buttons = [
        (c1, "🛈 Inject High O2 Stream (SpO2 Assist)",
         "O2_BOOST", "High O2 Flow Activated."),
        (c2, "💨 Flush Atmospheric CO2 (Scrubber Boost)",
         "CO2_FLUSH", "Atmospheric CO2 Scrubber Boost Activated."),
        (c3, "🛡️ Deploy Radiation Shielding Protocols",
         "RAD_SHIELD", "Radiation Storm Protocol Implemented."),
    ]
    for column, label, action, message in countermeasure_buttons:
        if column.button(label, key=f"cm_{action}"):
            if st.session_state.latest_sample is None or st.session_state.latest_score is None:
                st.warning(
                    "No active telemetry sample is available yet. Start a telemetry stream or replay first.")
            else:
                before = float(st.session_state.latest_score)
                sample_before = st.session_state.latest_sample.copy()
                sample_after = apply_countermeasure_to_sample(
                    sample_before, action)
                after_outputs = compute_model_outputs(sample_after)
                st.session_state.active_countermeasure = action
                st.session_state.latest_sample = sample_after
                st.session_state.latest_score = after_outputs["risk_score"]
                record_countermeasure(
                    action, before, after_outputs["risk_score"])
                st.success(
                    f"{message} Risk changed {before:.1f} → {after_outputs['risk_score']:.1f}.")
                st.rerun()
    if st.button("🔄 Clear Active Countermeasure", key="clear_countermeasure"):
        st.session_state.active_countermeasure = None
        st.success(
            "Active countermeasure cleared. Existing telemetry history is preserved.")
        st.rerun()
    st.divider()
    st.subheader("📊 Countermeasure Effectiveness")
    cm_history = st.session_state.get("countermeasure_history", [])
    if cm_history:
        st.dataframe(pd.DataFrame(cm_history), use_container_width=True)
    else:
        st.info("No countermeasure actions evaluated yet.")
    if st.session_state.get("last_countermeasure_effect"):
        effect = st.session_state.last_countermeasure_effect
        st.metric(
            "Latest Risk Change",
            f"{effect['Risk Change']:+.1f}",
            delta=f"{effect['Outcome']}"
        )

# --- TAB 4: INCIDENT LOGS ---
with tab_logs:
    st.subheader("📄 Automated Incident Logging & Forensic Export")
    logs = st.session_state.get("incident_logs", [])
    if len(logs) > 0:
        df_incidents = pd.DataFrame(logs)
        st.dataframe(df_incidents, use_container_width=True)
        csv_buffer = df_incidents.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Mission Incident Log (CSV)",
            data=csv_buffer,
            file_name="astrohealth_mission_incidents.csv",
            mime="text/csv",
            key="export_incident_log_static",
        )
    else:
        st.info(
            "No incident logs recorded for the current session yet. "
            "Click '🚀 Start Telemetry Stream' in the sidebar to generate logs."
        )
    st.divider()
    # Full telemetry session export
    hist_ticks = st.session_state.get("history", {}).get("tick", [])
    if len(hist_ticks) > 0:
        history = st.session_state["history"]
        df_full_history = pd.DataFrame(history)
        json_str = df_full_history.to_json(
            orient="records",
            indent=2
        )
        st.download_button(
            label="💾 Export Full Telemetry Session (JSON for Replay)",
            data=json_str,
            file_name=f"telemetry_session_{int(time.time())}.json",
            mime="application/json",
            key="full_session_export",
        )
    else:
        st.info("No telemetry session available for export yet.")

# --- TAB 5: MODEL DIAGNOSTICS ---
with tab_diagnostics:
    st.subheader(
        "⚡ Edge Performance & Neural Net Specifications"
    )
    # ---------------------------------------------
    # MODEL SPECIFICATIONS
    # ---------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    param_count = sum(
        p.numel()
        for p in model.parameters()
    )
    model_size_kb = (
        param_count * 4
    ) / 1024
    m1.metric(
        "Model Architecture",
        "MLP Deep Neural Net"
    )
    m2.metric(
        "Total Parameters",
        f"{param_count:,}"
    )
    m3.metric(
        "Memory Footprint",
        f"{model_size_kb:.2f} KB"
    )
    m4.metric(
        "Target Platform",
        "Spacecraft Edge Microcontroller"
    )

    # ---------------------------------------------
    # VALIDATION RESULTS
    # ---------------------------------------------
    st.divider()
    st.subheader(
        "🧪 Independent Test-Set Validation"
    )
    st.caption(
        "Validation uses a stratified 80/20 split of the "
        "synthetic telemetry dataset. These metrics measure "
        "agreement with the simulation-generated labels; "
        "they do not establish clinical or flight certification."
    )
    v1, v2, v3, v4 = st.columns(4)
    v1.metric(
        "Test Accuracy",
        f"{validation_metrics['accuracy'] * 100:.2f}%"
    )
    v2.metric(
        "Macro Precision",
        f"{validation_metrics['precision'] * 100:.2f}%"
    )
    v3.metric(
        "Macro Recall",
        f"{validation_metrics['recall'] * 100:.2f}%"
    )
    v4.metric(
        "Macro F1",
        f"{validation_metrics['f1'] * 100:.2f}%"
    )
    v5, v6, v7, v8 = st.columns(4)
    v5.metric(
        "Risk MAE",
        f"{validation_metrics['risk_mae']:.2f}"
    )
    v6.metric(
        "Risk RMSE",
        f"{validation_metrics['risk_rmse']:.2f}"
    )
    v7.metric(
        "Training Samples",
        f"{validation_metrics['train_samples']:,}"
    )
    v8.metric(
        "Test Samples",
        f"{validation_metrics['test_samples']:,}"
    )
    st.subheader("🧩 Confusion Matrix — Synthetic Test Split")
    confusion = validation_metrics.get("confusion_matrix")
    if confusion is not None:
        cm = pd.DataFrame(
            confusion,
            index=["Actual: Nominal", "Actual: Warning", "Actual: Critical"],
            columns=["Pred: Nominal", "Pred: Warning", "Pred: Critical"],
        )
        st.dataframe(cm, use_container_width=True)
        st.caption(
            "This matrix comes from the same synthetic stratified test split as the metrics above; it is not clinical validation.")
    else:
        st.info(
            "Confusion matrix will appear after the backend validation metrics include it.")
    st.subheader("📚 Threshold & Standards Provenance")
    st.markdown(
        "- **NASA-STD-3001 Vol. 2** — human-system requirements and environmental monitoring.  "
        "The current standard emphasizes trend analysis and monitoring of pressure, ppO₂ and ppCO₂, including EVA suit data.\n"
        "- **NASA Vol. 2 §6.2** — atmospheric composition, ppO₂, CO₂ and pressure context.\n"
        "- **NASA Vol. 2 §11.3** — suited atmospheric monitoring and alerting.\n"
        "- **Important:** AstroHealth's 25/55 risk bands are **model-status thresholds**, not NASA medical limits.\n"
    )
    for label, url in NASA_SOURCES.items():
        st.markdown(f"- [{label}]({url})")
    st.divider()
    st.subheader("🌍 External / Research Data Validation")
    st.caption(
        "Upload telemetry that has been normalized to the AstroHealth schema. This validates the model against an independent dataset; "
        "it does not imply the uploaded data is NASA-owned or clinically certified."
    )
    real_data_file = st.file_uploader(
        "Optional: upload a real research CSV for a vital-sign sanity check",
        type=["csv"],
        key="real_data_sanity_upload",
        help="For example, a CSV exported from an appropriately licensed PhysioNet or NASA OSDR dataset. This check is descriptive only."
    )
    if real_data_file is not None:
        real_df = pd.read_csv(real_data_file)
        if st.button("🔎 Run Real-Data Sanity Check", key="run_real_data_sanity"):
            st.session_state.real_data_sanity_result = real_data_sanity_check(
                real_df)
        sanity = st.session_state.get("real_data_sanity_result")
        if sanity is not None:
            if sanity.empty:
                st.warning(
                    "No supported vital-sign columns were detected. No claim of validation is made.")
            else:
                st.dataframe(sanity, use_container_width=True)
                st.success(
                    "Real-data sanity check complete. This is a descriptive data check, not model or clinical validation.")
    external_file = st.file_uploader(
        "Upload normalized external telemetry (CSV) for model validation",
        type=["csv"],
        key="external_validation_upload",
    )
    if external_file is not None:
        external_df = pd.read_csv(external_file)
        st.write(f"Loaded **{len(external_df):,} rows**.")
        if st.button("🧪 Run External Validation", key="run_external_validation"):
            result = validate_external_dataset(external_df, model, mean, scale)
            st.session_state.external_validation_result = result
    external_result = st.session_state.get("external_validation_result")
    if external_result:
        if not external_result.get("success"):
            st.error(
                f"External validation failed: {external_result.get('error', 'Unknown error')}")
        else:
            metrics = external_result.get("metrics", {})
            cols = st.columns(4)
            cols[0].metric("Rows Evaluated",
                           f"{metrics.get('rows_evaluated', 0):,}")
            if "risk_mae" in metrics:
                cols[1].metric("Risk MAE", f"{metrics['risk_mae']:.2f}")
            if "risk_rmse" in metrics:
                cols[2].metric("Risk RMSE", f"{metrics['risk_rmse']:.2f}")
            if "status_accuracy" in metrics:
                cols[3].metric("Status Accuracy",
                               f"{metrics['status_accuracy'] * 100:.2f}%")
            st.dataframe(external_result["predictions"].head(
                100), use_container_width=True)
            st.download_button(
                "📥 Export External Validation Results",
                external_result["predictions"].to_csv(
                    index=False).encode("utf-8"),
                file_name="astrohealth_external_validation.csv",
                mime="text/csv",
                key="export_external_validation",
            )
    st.divider()
    st.subheader("🧪 Robustness & Generalization Test Suite")
    st.caption(
        "Engineering stress tests probe behavior under sensor noise, dropout, compound anomalies and sensor spikes. "
        "These are robustness tests, not clinical validation."
    )
    if st.button("▶️ Run Robustness Suite", key="run_robustness"):
        baseline = st.session_state.get("personal_baseline")
        if baseline is None:
            baseline = (
                telemetry_pool[telemetry_pool["y_status"] == "nominal"]
                .drop(columns=["y_status"])
                .iloc[0]
                .to_dict()
            )
        st.session_state.robustness_result = run_robustness_suite(
            model, mean, scale, baseline
        )
    robustness_result = st.session_state.get("robustness_result")
    if robustness_result:
        if isinstance(robustness_result, dict) and "scenarios" in robustness_result:
            robustness_df = pd.DataFrame(robustness_result["scenarios"])
        elif isinstance(robustness_result, list):
            robustness_df = pd.DataFrame(robustness_result)
        else:
            robustness_df = pd.DataFrame(robustness_result)
        if not robustness_df.empty:
            st.dataframe(robustness_df, use_container_width=True)
            st.download_button(
                "📥 Export Robustness Report",
                robustness_df.to_csv(index=False).encode("utf-8"),
                file_name="astrohealth_robustness_report.csv",
                mime="text/csv",
                key="export_robustness_report",
            )
    st.divider()
    st.subheader("🧭 One-Slide System Architecture")
    st.code(
        "Crew Sensors / External Data\n"
        "        │\n"
        "        ▼\n"
        "Sensor Integrity ──► Personal Digital Twin\n"
        "        │                    │\n"
        "        └──────────┬─────────┘\n"
        "                   ▼\n"
        "        Dual-Head Health Model\n"
        "          ├── Risk Score\n"
        "          └── Status + Confidence\n"
        "                   │\n"
        "          ┌────────┴────────┐\n"
        "          ▼                 ▼\n"
        "   Trajectory Forecast   Decision Support\n"
        "                            │\n"
        "                            ▼\n"
        "                Simulated Countermeasure\n"
        "                            │\n"
        "                            ▼\n"
        "                     Before → After\n"
        "Earth Link Lost → Local Inference → Buffered Sync",
        language="text",
    )
    render_ethics_privacy()


def render_communication_status():
    if st.session_state.communication_blackout:
        communication_placeholder.error(
            "📡 **EARTH LINK LOST** — "
            "**ASTROHEALTH AUTONOMOUS MODE ACTIVE** | "
            "Local inference, monitoring and incident logging "
            "continue onboard."
        )
    else:
        sync_count = (
            st.session_state.last_sync_count
        )
        if sync_count > 0:
            communication_placeholder.success(
                f"📡 **EARTH LINK RESTORED** — "
                f"{sync_count} locally buffered event(s) "
                f"ready for synchronization."
            )
        else:
            communication_placeholder.success(
                "📡 **EARTH LINK: CONNECTED**"
            )

# ----------------------------------------------------
# DISPLAY & CHART HELPER FUNCTIONS
# ----------------------------------------------------


def update_metric_cards(sample, score):
    card_risk.metric("Risk Score", f"{score:.1f} / 100")
    card_hr.metric("Heart Rate", f"{int(sample['heart_rate'])} bpm")
    card_hrv.metric("HRV", f"{sample['hrv']:.1f} ms")
    card_spo2.metric("SpO2", f"{sample['spo2']:.1f}%")
    card_resp.metric("Resp Rate", f"{sample['resp_rate']:.1f} br/m")
    card_temp.metric("Core Temp", f"{sample['core_temp']:.1f} °C")
    card_co2.metric("CO2 Level", f"{int(sample['ambient_co2'])} ppm")
    card_press.metric("Pressure", f"{sample['pressure']:.1f} kPa")
    card_rad.metric("Radiation", f"{sample['rad_rate']:.1f} uSv/h")
    card_sleep.metric("Sleep Debt", f"{sample['sleep_debt']:.1f} hrs")


def render_dashboard_charts(df_hist, sample, tick):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Model Risk Score & Predictive Forecast",
            "Cardiovascular & Vital Metrics",
            "Environmental Parameters & Stressors",
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
    )
    # --- SUBPLOT 1: RISK & FORECAST ---
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["risk"],
            name="Risk Score",
            line=dict(color="#E43700", width=3),
        ),
        row=1,
        col=1,
    )
    if len(df_hist) >= 3:
        z = np.polyfit(df_hist["tick"], df_hist["risk"], 1)
        p = np.poly1d(z)
        future_ticks = list(range(tick, tick + 6))
        future_risk = [min(100.0, max(0.0, p(t))) for t in future_ticks]
        fig.add_trace(
            go.Scatter(
                x=future_ticks,
                y=future_risk,
                name="Projected Trend",
                line=dict(color="#EAFE07", dash="dot", width=2),
            ),
            row=1,
            col=1,
        )

    # --- SUBPLOT 2: VITALS & CORE TEMP ---
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["heart_rate"],
            name="HR (bpm)",
            line=dict(color="#e74c3c"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["spo2"],
            name="SpO2 (%)",
            line=dict(color="#2ecc71"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["hrv"],
            name="HRV (ms)",
            line=dict(color="#8e44ad"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["resp_rate"],
            name="Resp Rate (br/m)",
            line=dict(color="#3498db", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["core_temp"],
            name="Core Temp (°C)",
            line=dict(color="#e67e22", dash="dot"),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # --- SUBPLOT 3: ENVIRONMENTAL & FATIGUE ---
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["ambient_co2"],
            name="CO2 (ppm)",
            line=dict(color="#16a085"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["pressure"],
            name="Pressure (kPa)",
            line=dict(color="#2E96F5"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["rad_rate"],
            name="Radiation (uSv/h)",
            line=dict(color="#f39c12", dash="dot"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["tick"],
            y=df_hist["sleep_debt"],
            name="Sleep Debt (h)",
            line=dict(color="#bdc3c7", dash="dashdot"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    # Axis and Layout Settings
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 23, 63, 0.5)",
        font=dict(color="#FFFFFF", family="Overpass"),
        legend=dict(bgcolor="rgba(7, 23, 63, 0.8)", bordercolor="#0960E1"),
    )
    fig.update_yaxes(title_text="Risk Score", range=[-5, 105], row=1, col=1)
    fig.update_yaxes(title_text="Vitals (HR/SpO2/HRV/Resp)", row=2, col=1)
    fig.update_yaxes(title_text="Core Temp (°C)",
                     row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="CO2 (ppm)", row=3, col=1)
    fig.update_yaxes(title_text="Pressure / Rad / Sleep",
                     row=3, col=1, secondary_y=True)
    chart_placeholder.plotly_chart(
        fig,
        use_container_width=True,
        key=f"main_chart_{tick}_{len(df_hist)}"
    )

    # --- RADAR CHART (SYSTEM IMPACT MAP) ---
    cardio_stress = min(
        100, (abs(sample["heart_rate"] - 75) * 1.2 + max(0, 50 - sample["hrv"]) * 0.8))
    resp_stress = min(
        100, (max(0, 98 - sample["spo2"]) * 5 + abs(sample["resp_rate"] - 16) * 3))
    env_stress = min(100, (max(
        0, sample["ambient_co2"] - 600) / 30 + max(0, 101.3 - sample["pressure"]) * 2))
    neuro_stress = min(
        100, (sample["sleep_debt"] * 8 + sample["rad_rate"] * 0.4))
    fig_radar = go.Figure(
        data=go.Scatterpolar(
            r=[cardio_stress, resp_stress, env_stress, neuro_stress],
            theta=["Cardiovascular", "Respiratory",
                   "Environment", "Neurological"],
            fill="toself",
            fillcolor="rgba(228, 55, 0, 0.4)",
            line=dict(color="#E43700", width=2),
        )
    )
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", family="Overpass"),
        polar=dict(
            bgcolor="rgba(7, 23, 63, 0.5)",
            radialaxis=dict(visible=True, range=[0, 100], color="#FFFFFF"),
        ),
        showlegend=False,
        height=400,
    )
    radar_chart_placeholder.plotly_chart(
        fig_radar,
        use_container_width=True,
        key=f"radar_chart_{tick}_{len(df_hist)}"
    )


# ----------------------------------------------------
# RESTORE DASHBOARD AFTER STREAM / RERUN
# ----------------------------------------------------
if (
    len(st.session_state.history["tick"]) > 0
    and not start_stream
    and not start_replay
):
    history = st.session_state.history
    # Latest telemetry sample
    latest_sample = {
        "heart_rate": history["heart_rate"][-1],
        "hrv": history["hrv"][-1],
        "spo2": history["spo2"][-1],
        "resp_rate": history["resp_rate"][-1],
        "core_temp": history["core_temp"][-1],
        "ambient_co2": history["ambient_co2"][-1],
        "pressure": history["pressure"][-1],
        "rad_rate": history["rad_rate"][-1],
        "sleep_debt": history["sleep_debt"][-1],
    }
    latest_score = history["risk"][-1]
    latest_tick = history["tick"][-1]
    # Determine latest status
    if latest_score < 25:
        latest_status = "NOMINAL"
    elif latest_score < 55:
        latest_status = "WARNING"
    else:
        latest_status = "CRITICAL"
    # Restore alert
    if latest_status == "CRITICAL":
        status_alert.error(
            f"🚨 **CRITICAL ALERT**: Risk Level is {latest_score:.1f}/100!"
        )
    elif latest_status == "WARNING":
        status_alert.warning(
            f"⚠️ **WARNING**: Risk Level is {latest_score:.1f}/100!"
        )
    else:
        status_alert.success(
            "✅ **NOMINAL**: All vital parameters within expected ranges."
        )
    # Restore metric cards
    update_metric_cards(
        latest_sample,
        latest_score
    )
    # Restore graph
    render_dashboard_charts(
        pd.DataFrame(history),
        latest_sample,
        latest_tick
    )
    render_trajectory_forecast(history)
    render_communication_status()
    if st.session_state.get("latest_model_outputs"):
        render_model_consensus(st.session_state.latest_model_outputs)

# ----------------------------------------------------
# 5. LIVE STREAM ENGINE
# ----------------------------------------------------
if start_stream:
    # ------------------------------------------------
    # START A COMPLETELY NEW TELEMETRY RUN
    # ------------------------------------------------
    st.session_state.history = {
        "tick": [],
        "risk": [],
        "heart_rate": [],
        "hrv": [],
        "spo2": [],
        "resp_rate": [],
        "core_temp": [],
        "ambient_co2": [],
        "pressure": [],
        "rad_rate": [],
        "sleep_debt": [],
    }
    st.session_state.incident_logs = []
    st.session_state.countermeasure_history = []
    st.session_state.last_countermeasure_effect = None
    st.session_state.latest_model_outputs = None
    st.session_state.pending_sync_events = 0
    st.session_state.last_sync_count = 0
    # ------------------------------------------------
    # PERSISTENT SIMULATION STATE
    # ------------------------------------------------
    for tick in range(1, total_ticks + 1):
        if tick == 1:
            raw_base_sample = (
                telemetry_pool[
                    telemetry_pool["y_status"] == "nominal"
                ]
                .sample(n=1)
                .drop(columns=["y_status"])
                .iloc[0]
                .to_dict()
            )
            st.session_state.base_sample = raw_base_sample
            st.session_state.personal_baseline = (
                build_personal_baseline(
                    raw_base_sample,
                    selected_astronaut
                )
            )
        # Use the personalized physical baseline throughout the mission.
        base_sample = st.session_state.personal_baseline.copy()
        # Progress goes from 0 → 1 during the mission
        progress = (tick - 1) / max(total_ticks - 1, 1)
        sample = base_sample.copy()

        # ============================================================
        # ANOMALY SIMULATION
        # ============================================================
        if anomaly_choice == "Cabin Depressurization":
            # Environmental effects
            sample["pressure"] = 101.0 - (16.0 * progress)
            sample["spo2"] = 98.0 - (8.0 * progress)
            # Physiological response
            sample["heart_rate"] = 75.0 + (20.0 * progress)
            sample["resp_rate"] = 15.0 + (6.0 * progress)
            sample["hrv"] = 50.0 - (20.0 * progress)
        elif anomaly_choice == "CO2 Scrubber Failure":
            # CO2 accumulation
            sample["ambient_co2"] = 600.0 + (2200.0 * progress)
            # Physiological response to rising CO2
            sample["resp_rate"] = 15.0 + (10.0 * progress)
            sample["heart_rate"] = 75.0 + (15.0 * progress)
            sample["spo2"] = 98.0 - (4.0 * progress)
            sample["hrv"] = 50.0 - (15.0 * progress)
        elif anomaly_choice == "Solar Particle Event (High Radiation)":
            # Radiation increases sharply
            sample["rad_rate"] = 15.0 + (105.0 * progress)
            # Physiological stress response
            sample["heart_rate"] = 75.0 + (10.0 * progress)
            sample["hrv"] = 50.0 - (12.0 * progress)
            # Small secondary effects
            sample["resp_rate"] = 15.0 + (3.0 * progress)
        elif anomaly_choice == "Extreme Heat Exposure":
            # Core temperature rises
            sample["core_temp"] = 37.0 + (2.0 * progress)
            # Physiological response to heat
            sample["heart_rate"] = 75.0 + (25.0 * progress)
            sample["resp_rate"] = 15.0 + (7.0 * progress)
            sample["hrv"] = 50.0 - (18.0 * progress)
            # Small SpO2 reduction at severe heat
            sample["spo2"] = 98.0 - (2.0 * progress)
        elif anomaly_choice == "None (Nominal Operation)":
            # Small natural sensor variation during normal operation
            sample["heart_rate"] += np.random.normal(0, 0.8)
            sample["hrv"] += np.random.normal(0, 1.0)
            sample["spo2"] += np.random.normal(0, 0.08)
            sample["resp_rate"] += np.random.normal(0, 0.15)
            sample["core_temp"] += np.random.normal(0, 0.02)
            sample["ambient_co2"] += np.random.normal(0, 3.0)
            sample["pressure"] += np.random.normal(0, 0.03)
            sample["rad_rate"] += np.random.normal(0, 0.1)

        # ============================================================
        # SENSOR FAULT INJECTION
        # ============================================================
        sample = apply_sensor_fault(
            sample,
            st.session_state.personal_baseline,
            sensor_fault_choice,
            tick
        )
        # Keep a copy with missing channels intact for sensor diagnostics,
        # then impute only for neural-network inference.
        diagnostic_sample = sample.copy()

        # ============================================================
        # COUNTERMEASURE RESPONSE
        # ============================================================
        active_countermeasure = st.session_state.get(
            "active_countermeasure"
        )
        if active_countermeasure == "CO2_FLUSH":
            sample["ambient_co2"] -= 180
            sample["ambient_co2"] = max(sample["ambient_co2"], 400)
            sample["resp_rate"] -= 0.25
            sample["resp_rate"] = max(sample["resp_rate"], 12)
        elif active_countermeasure == "O2_BOOST":
            sample["spo2"] += 0.8
            sample["spo2"] = min(sample["spo2"], 99.5)
        elif active_countermeasure == "RAD_SHIELD":
            sample["rad_rate"] -= 10
            sample["rad_rate"] = max(sample["rad_rate"], 5)
        # Sensor integrity runs before model inference.
        sensor_result = detect_sensor_faults(
            diagnostic_sample,
            st.session_state.history,
            st.session_state.personal_baseline
        )
        st.session_state.sensor_diagnostics = sensor_result
        sample = safe_sample_for_model(sample)
        st.session_state.latest_sample = sample.copy()
        t_start = time.time()
        model_outputs = compute_model_outputs(sample)
        score = model_outputs["risk_score"]
        status = model_outputs["risk_status"]
        inference_latency = (time.time() - t_start) * 1000
        explanation_summary, _ = explain_risk_score(sample, model, mean, scale)
        st.session_state.latest_score = score

        # ------------------------------------------------
        # UPDATE HISTORY
        # ------------------------------------------------
        history = st.session_state.history
        history["tick"].append(tick)
        history["risk"].append(score)
        history["heart_rate"].append(sample["heart_rate"])
        history["hrv"].append(sample["hrv"])
        history["spo2"].append(sample["spo2"])
        history["resp_rate"].append(sample["resp_rate"])
        history["core_temp"].append(sample["core_temp"])
        history["ambient_co2"].append(sample["ambient_co2"])
        history["pressure"].append(sample["pressure"])
        history["rad_rate"].append(sample["rad_rate"])
        history["sleep_debt"].append(sample["sleep_debt"])
        append_incident_log(
            tick=tick,
            astronaut=selected_astronaut,
            sample=sample,
            outputs=model_outputs,
            explanation=explanation_summary,
            sensor_result=sensor_result,
            anomaly=anomaly_choice,
            countermeasure=st.session_state.get("active_countermeasure"),
            latency_ms=inference_latency,
        )
        if status == "CRITICAL":
            status_alert.error(
                f"🚨 **CRITICAL ALERT**: Risk Level is {score:.1f}/100!")
        elif status == "WARNING":
            status_alert.warning(
                f"⚠️ **WARNING**: Risk Level is {score:.1f}/100!")
        else:
            status_alert.success(
                f"✅ **NOMINAL**: All vital parameters within expected ranges.")
        update_metric_cards(sample, score)
        explain_box.info(
            f"🧠 **Driver Analysis**: {explanation_summary} | ⚡ **Latency**: {inference_latency:.2f} ms")
        render_model_consensus(model_outputs)
        render_sensor_health()
        render_digital_twin_status(sample)
        render_eva_monitor(sample)
        render_communication_status()
        render_dashboard_charts(pd.DataFrame(
            st.session_state.history), sample, tick)
        render_trajectory_forecast(st.session_state.history)
        if st.session_state.communication_blackout:
            st.session_state.pending_sync_events += 1
        time.sleep(tick_speed)
    st.rerun()

# ----------------------------------------------------
# 6. ONE-CLICK 3-MINUTE JUDGE DEMO
# ----------------------------------------------------
if run_demo:
    st.session_state.history = {
        "tick": [], "risk": [], "heart_rate": [], "hrv": [], "spo2": [],
        "resp_rate": [], "core_temp": [], "ambient_co2": [], "pressure": [],
        "rad_rate": [], "sleep_debt": [],
    }
    st.session_state.incident_logs = []
    st.session_state.countermeasure_history = []
    st.session_state.last_countermeasure_effect = None
    st.session_state.communication_blackout = False
    st.session_state.pending_sync_events = 0
    st.session_state.last_sync_count = 0
    raw_base = (
        telemetry_pool[telemetry_pool["y_status"] == "nominal"]
        .sample(n=1)
        .drop(columns=["y_status"])
        .iloc[0]
        .to_dict()
    )
    st.session_state.base_sample = raw_base
    st.session_state.personal_baseline = build_personal_baseline(
        raw_base, selected_astronaut)
    demo_ticks = 18
    st.info("🎬 JUDGE DEMO: nominal → CO₂ scrubber failure → forecast → CO₂ flush → Earth-link blackout → autonomous operation. Use this sequence as the core of a ~3-minute narrated presentation.")
    progress_bar = st.progress(0)
    for tick in range(1, demo_ticks + 1):
        base = st.session_state.personal_baseline.copy()
        sample = base.copy()
        progress = (tick - 1) / 5.0
        if tick <= 4:
            phase = "NOMINAL"
            sample["heart_rate"] += np.random.normal(0, 0.6)
            sample["ambient_co2"] += np.random.normal(0, 2.0)
        elif tick <= 10:
            phase = "CO2 SCRUBBER FAILURE"
            p = min(1.0, progress)
            sample["ambient_co2"] = 600.0 + 2400.0 * p
            sample["resp_rate"] = 15.0 + 11.0 * p
            sample["heart_rate"] = 75.0 + 18.0 * p
            sample["spo2"] = 98.0 - 4.5 * p
            sample["hrv"] = 50.0 - 16.0 * p
        elif tick <= 14:
            phase = "CO2 FLUSH COUNTERMEASURE"
            p = min(1.0, (tick - 11) / 3.0)
            sample["ambient_co2"] = max(600.0 + 2400.0 - 500.0 * p, 400.0)
            sample["resp_rate"] = max(15.0 + 11.0 - 2.0 * p, 12.0)
            sample["heart_rate"] = max(75.0 + 18.0 - 4.0 * p, 75.0)
            sample["spo2"] = min(98.0 - 4.5 + 0.8 * p, 99.5)
            sample["hrv"] = min(50.0 - 16.0 + 3.0 * p, 50.0)
            st.session_state.active_countermeasure = "CO2_FLUSH"
        else:
            phase = "EARTH LINK BLACKOUT / AUTONOMOUS"
            st.session_state.communication_blackout = True
            sample["ambient_co2"] = max(1900.0 - 80.0 * (tick - 15), 400.0)
            sample["resp_rate"] = max(21.0 - 0.5 * (tick - 15), 12.0)
            sample["heart_rate"] = max(88.0 - 1.5 * (tick - 15), 75.0)
            sample["spo2"] = min(95.0 + 0.2 * (tick - 15), 99.5)
        diagnostic_sample = sample.copy()
        sensor_result = detect_sensor_faults(
            diagnostic_sample,
            st.session_state.history,
            st.session_state.personal_baseline,
        )
        st.session_state.sensor_diagnostics = sensor_result
        sample = safe_sample_for_model(sample)
        outputs = compute_model_outputs(sample)
        score = outputs["risk_score"]
        st.session_state.latest_sample = sample.copy()
        st.session_state.latest_score = score
        history = st.session_state.history
        history["tick"].append(tick)
        history["risk"].append(score)
        for feature in FEATURES:
            history[feature].append(sample[feature])
        explanation, _ = explain_risk_score(sample, model, mean, scale)
        append_incident_log(
            tick=tick,
            astronaut=selected_astronaut,
            sample=sample,
            outputs=outputs,
            explanation=explanation,
            sensor_result=sensor_result,
            anomaly=phase,
            countermeasure=st.session_state.get("active_countermeasure"),
        )
        status_alert.info(
            f"🎬 **DEMO TICK {tick}/{demo_ticks} — {phase}** | Risk {score:.1f}/100")
        update_metric_cards(sample, score)
        render_model_consensus(outputs)
        render_sensor_health()
        render_digital_twin_status(sample)
        render_eva_monitor(sample)
        render_communication_status()
        render_dashboard_charts(pd.DataFrame(history), sample, tick)
        render_trajectory_forecast(history)
        if st.session_state.communication_blackout:
            st.session_state.pending_sync_events += 1
        progress_bar.progress(tick / demo_ticks)
        time.sleep(0.5)
    st.success("🎬 DEMO COMPLETE — this automated sequence is intentionally short; use it as the backbone of a ~3-minute narrated judge presentation.")
    st.rerun()

# ----------------------------------------------------
# 6. FORENSIC MISSION REPLAY ENGINE
# ----------------------------------------------------
if start_replay and st.session_state.replay_data is not None:
    # ------------------------------------------------
    # START A COMPLETELY NEW REPLAY SESSION
    # ------------------------------------------------
    st.session_state.history = {
        "tick": [],
        "risk": [],
        "heart_rate": [],
        "hrv": [],
        "spo2": [],
        "resp_rate": [],
        "core_temp": [],
        "ambient_co2": [],
        "pressure": [],
        "rad_rate": [],
        "sleep_debt": [],
    }
    # Clear previous incident logs
    st.session_state.incident_logs = []
    st.session_state.countermeasure_history = []
    st.session_state.last_countermeasure_effect = None
    st.session_state.latest_model_outputs = None
    # Load replay dataset
    df_replay = st.session_state.replay_data
    if st.session_state.personal_baseline is None:
        first_valid = df_replay.iloc[0].to_dict()
        st.session_state.personal_baseline = build_personal_baseline(
            safe_sample_for_model(first_valid), selected_astronaut
        )
    for idx, row in df_replay.iterrows():
        raw_sample = row.to_dict()
        tick = int(raw_sample.get("tick", idx + 1))
        diagnostic_sample = raw_sample.copy()
        sample = safe_sample_for_model(raw_sample)

        # --------------------------------------------
        # AI RISK + DUAL-HEAD PREDICTION
        # --------------------------------------------
        t_start = time.time()
        model_outputs = compute_model_outputs(sample)
        score = model_outputs["risk_score"]
        status = model_outputs["risk_status"]

        # ------------------------------------------------
        # DIGITAL TWIN
        # ------------------------------------------------
        render_digital_twin_status(
            sample
        )

        # ------------------------------------------------
        # SENSOR INTEGRITY
        # ------------------------------------------------
        sensor_result = detect_sensor_faults(
            sample,
            st.session_state.history,
            st.session_state.personal_baseline
        )
        st.session_state.sensor_diagnostics = (
            sensor_result
        )
        render_sensor_health()

        # ------------------------------------------------
        # DUAL-HEAD MODEL OUTPUT
        # ------------------------------------------------
        inference_latency = (time.time() - t_start) * 1000
        explanation_summary, _ = explain_risk_score(
            sample,
            model,
            mean,
            scale
        )

        # ------------------------------------------------
        # UPDATE HISTORY
        # ------------------------------------------------
        history = st.session_state.history
        history["tick"].append(tick)
        history["risk"].append(score)
        history["heart_rate"].append(sample["heart_rate"])
        history["hrv"].append(sample["hrv"])
        history["spo2"].append(sample["spo2"])
        history["resp_rate"].append(sample["resp_rate"])
        history["core_temp"].append(sample["core_temp"])
        history["ambient_co2"].append(sample["ambient_co2"])
        history["pressure"].append(sample["pressure"])
        history["rad_rate"].append(sample["rad_rate"])
        history["sleep_debt"].append(sample["sleep_debt"])

        # --------------------------------------------
        # INCIDENT LOG
        # --------------------------------------------
        append_incident_log(
            tick=tick,
            astronaut=selected_astronaut,
            sample=sample,
            outputs=model_outputs,
            explanation=explanation_summary,
            sensor_result=sensor_result,
            anomaly="Replay / External Session",
            countermeasure=None,
            latency_ms=inference_latency,
        )

        # --------------------------------------------
        # STATUS ALERT
        # --------------------------------------------
        if status == "CRITICAL":
            status_alert.error(
                f"🚨 **CRITICAL ALERT [TICK {tick}/{len(df_replay)}]**: "
                f"Risk Level is {score:.1f}/100!"
            )
        elif status == "WARNING":
            status_alert.warning(
                f"⚠️ **WARNING [TICK {tick}/{len(df_replay)}]**: "
                f"Risk Level is {score:.1f}/100!"
            )
        else:
            status_alert.success(
                f"✅ **NOMINAL [TICK {tick}/{len(df_replay)}]**: "
                f"All vital parameters within expected ranges."
            )

        # --------------------------------------------
        # UPDATE DASHBOARD
        # --------------------------------------------
        update_metric_cards(sample, score)
        explain_box.info(
            f"🧠 **Driver Analysis**: {explanation_summary} | "
            f"⚡ **Latency**: {inference_latency:.2f} ms"
        )
        render_model_consensus(model_outputs)
        render_sensor_health()
        render_communication_status()
        render_dashboard_charts(
            pd.DataFrame(st.session_state.history),
            sample,
            tick
        )
        render_trajectory_forecast(st.session_state.history)
        if st.session_state.communication_blackout:
            st.session_state.pending_sync_events += 1

        # --------------------------------------------
        # REPLAY SPEED
        # --------------------------------------------
        time.sleep(tick_speed)
    # Replay finished
    st.rerun()

# ----------------------------------------------------
# 9. VISUAL IDENTITY FOOTER
# ----------------------------------------------------
st.markdown(
    """
    <div class="ah-footer">
        <span>ASTROHEALTH // AHMC-01</span>
        <span>ENGINEERING DEMONSTRATOR • HUMAN OVERSIGHT</span>
    </div>
    """,
    unsafe_allow_html=True,
)
