import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GlucoTwin AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, #e8f5f3, #eef4ff);
    border: 1px solid #dce7ee;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 750;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 17px;
    color: #536471;
}

.status {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    background-color: #dff5e9;
    color: #137333;
    font-weight: 650;
    font-size: 14px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}

.prediction-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #edf5ff, #f7fbff);
    border: 1px solid #d9e7f5;
    text-align: center;
}

.prediction-label {
    font-size: 15px;
    color: #657482;
}

.prediction-value {
    font-size: 42px;
    font-weight: 800;
    margin: 8px 0;
}

.risk-high {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    background-color: #ffe5e5;
    color: #b42318;
    font-weight: 700;
}

.risk-normal {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    background-color: #e5f7ed;
    color: #137333;
    font-weight: 700;
}

.risk-low {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    background-color: #fff3d6;
    color: #8a5a00;
    font-weight: 700;
}

.scenario-box {
    padding: 22px;
    border-radius: 16px;
    background-color: #ffffff;
    border: 1px solid #e0e6ed;
}

.footer {
    text-align: center;
    color: #7a8793;
    padding-top: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv("data/all_patients.csv")

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["patient_id", "time"]
).reset_index(drop=True)

model = joblib.load(
    "models/multi_patient_model.pkl"
)


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(patient_df):

    patient_df = patient_df.copy()

    patient_df = patient_df.sort_values(
        "time"
    ).reset_index(drop=True)

    patient_df["glucose_5min"] = \
        patient_df["glucose"].shift(1)

    patient_df["glucose_10min"] = \
        patient_df["glucose"].shift(2)

    patient_df["glucose_15min"] = \
        patient_df["glucose"].shift(3)

    patient_df["glucose_30min"] = \
        patient_df["glucose"].shift(6)

    patient_df["glucose_60min"] = \
        patient_df["glucose"].shift(12)

    patient_df["glucose_trend_30min"] = (
        patient_df["glucose"]
        - patient_df["glucose_30min"]
    )

    patient_df["glucose_avg_30min"] = (
        patient_df["glucose"]
        .rolling(6)
        .mean()
    )

    patient_df["glucose_avg_60min"] = (
        patient_df["glucose"]
        .rolling(12)
        .mean()
    )

    return patient_df


features = [
    "glucose",
    "calories",
    "heart_rate",
    "steps",
    "basal_rate",
    "bolus_volume_delivered",
    "carb_input",
    "glucose_5min",
    "glucose_10min",
    "glucose_15min",
    "glucose_30min",
    "glucose_60min",
    "glucose_trend_30min",
    "glucose_avg_30min",
    "glucose_avg_60min"
]


# ============================================================
# RISK FUNCTION
# ============================================================

def risk_level(glucose):

    if glucose < 70:
        return "LOW GLUCOSE RISK"

    elif glucose >= 180:
        return "HIGH GLUCOSE RISK"

    return "NORMAL RANGE"


def risk_class(risk):

    if "HIGH" in risk:
        return "risk-high"

    if "LOW" in risk:
        return "risk-low"

    return "risk-normal"


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🧬 GlucoTwin AI
</div>

<div class="hero-subtitle">
Personalized Digital Twin for Early Blood-Glucose Prediction
</div>

<br>

<span class="status">
● Digital Twin Active
</span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🧬 GlucoTwin AI")

st.sidebar.markdown(
    "### Patient Selection"
)

patients = sorted(
    df["patient_id"].unique()
)

patient_id = st.sidebar.selectbox(
    "Select Patient",
    patients
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**Model:** Random Forest"
)

st.sidebar.markdown(
    "**Prediction Horizon:** 2 Hours"
)

st.sidebar.markdown(
    "**Patients:** 25"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI research prototype for Digital Twin Challenge 2026"
)


# ============================================================
# PREPARE PATIENT DATA
# ============================================================

patient_df = df[
    df["patient_id"] == patient_id
]

patient_features = create_features(
    patient_df
).dropna()

latest = patient_features.iloc[-1].copy()


# ============================================================
# PATIENT HEADER
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient Digital Twin</div>',
    unsafe_allow_html=True
)

st.write(
    f"Active patient profile: **{patient_id}**"
)


# ============================================================
# CURRENT STATE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Current Patient State</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🩸 Glucose",
        f"{latest['glucose']:.1f} mg/dL"
    )

with col2:
    st.metric(
        "❤️ Heart Rate",
        f"{latest['heart_rate']:.1f} bpm"
    )

with col3:
    st.metric(
        "🚶 Steps",
        f"{latest['steps']:.0f}"
    )

with col4:
    st.metric(
        "🥗 Carbohydrates",
        f"{latest['carb_input']:.1f} g"
    )


# ============================================================
# PREDICTION
# ============================================================

input_data = latest[
    features
].to_frame().T

current_prediction = model.predict(
    input_data
)[0]

current_risk = risk_level(
    current_prediction
)

st.markdown(
    '<div class="section-title">🔮 2-Hour Glucose Prediction</div>',
    unsafe_allow_html=True
)

pred_col1, pred_col2 = st.columns([1.5, 1])

with pred_col1:

    st.markdown(
        f"""
        <div class="prediction-box">

        <div class="prediction-label">
        Predicted Glucose in ~2 Hours
        </div>

        <div class="prediction-value">
        {current_prediction:.1f} mg/dL
        </div>

        <div class="{risk_class(current_risk)}">
        {current_risk}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with pred_col2:

    st.info(
        "The Digital Twin uses the patient's recent "
        "glucose history, physiological signals and "
        "lifestyle inputs to estimate future glucose."
    )


# ============================================================
# GLUCOSE TREND
# ============================================================

st.markdown(
    '<div class="section-title">📈 Recent Glucose Trend</div>',
    unsafe_allow_html=True
)

chart_data = patient_df.tail(100)

fig = px.line(
    chart_data,
    x="time",
    y="glucose",
    markers=False
)

fig.update_layout(
    height=380,
    xaxis_title="Time",
    yaxis_title="Glucose (mg/dL)",
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified"
)

fig.add_hline(
    y=180,
    line_dash="dash",
    annotation_text="High-risk threshold"
)

fig.add_hline(
    y=70,
    line_dash="dash",
    annotation_text="Low-risk threshold"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# WHAT-IF SIMULATION
# ============================================================

st.markdown(
    '<div class="section-title">🧪 What-If Digital Twin Simulation</div>',
    unsafe_allow_html=True
)

st.write(
    "Modify lifestyle inputs and observe how the trained "
    "model's prediction changes."
)

sim_col1, sim_col2 = st.columns(2)

with sim_col1:

    carb_change = st.slider(
        "🥗 Additional Carbohydrates (g)",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )

with sim_col2:

    step_change = st.slider(
        "🚶 Additional Steps",
        min_value=0,
        max_value=3000,
        value=0,
        step=100
    )


scenario = latest.copy()

scenario["carb_input"] += carb_change

scenario["steps"] += step_change

scenario_input = scenario[
    features
].to_frame().T

scenario_prediction = model.predict(
    scenario_input
)[0]

scenario_risk = risk_level(
    scenario_prediction
)

difference = (
    scenario_prediction
    - current_prediction
)


# ============================================================
# SIMULATION RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📋 Simulation Result</div>',
    unsafe_allow_html=True
)

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:

    st.metric(
        "Current Prediction",
        f"{current_prediction:.1f} mg/dL"
    )

with result_col2:

    st.metric(
        "Scenario Prediction",
        f"{scenario_prediction:.1f} mg/dL",
        delta=f"{difference:+.1f} mg/dL"
    )

with result_col3:

    st.markdown(
        f"""
        <div class="{risk_class(scenario_risk)}">
        {scenario_risk}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DIGITAL TWIN SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Digital Twin Summary</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="scenario-box">

    <b>Patient:</b> {patient_id}<br><br>

    <b>Current glucose:</b>
    {latest['glucose']:.1f} mg/dL<br>

    <b>Predicted glucose:</b>
    {current_prediction:.1f} mg/dL<br>

    <b>Scenario change:</b>
    {difference:+.1f} mg/dL<br>

    <b>Simulation inputs:</b>
    +{carb_change} g carbohydrates,
    +{step_change} steps

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ GlucoTwin AI is a research and hackathon prototype. "
    "Predictions and risk indicators are not medical advice "
    "and have not been clinically validated."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    GlucoTwin AI • Digital Twin Challenge 2026
    </div>
    """,
    unsafe_allow_html=True
)