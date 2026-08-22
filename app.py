import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Failure Prognosis Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model = joblib.load("random_forest_model.pkl")
feature_names = joblib.load("feature_names.pkl")
threshold = float(joblib.load("threshold.pkl"))

TRAINING_RANGES = {
    "air_temp": (295.3, 304.5),
    "process_temp": (305.7, 313.8),
    "rotational_speed": (1168, 2886),
    "torque": (3.8, 76.6),
    "tool_wear": (0, 253)
}

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Orbitron:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
}

.stApp {
    background:
        radial-gradient(circle at 50% 0%, #182018 0%, #090b09 45%, #050605 100%);
    color: #e8e6d9;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.15;
    background-image:
        linear-gradient(#26352d 1px, transparent 1px),
        linear-gradient(90deg, #26352d 1px, transparent 1px);
    background-size: 45px 45px;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.console-header {
    border: 1px solid #41483b;
    background: linear-gradient(180deg, #1a1e16, #10130e);
    padding: 22px 28px;
    margin-bottom: 20px;
    border-radius: 6px;
    box-shadow: 0 0 30px rgba(0,0,0,0.35);
}

.console-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #eeeadd;
}

.console-subtitle {
    color: #8d9388;
    font-size: 12px;
    letter-spacing: 2px;
    margin-top: 5px;
}

.status-online {
    color: #43df89;
    font-weight: 700;
    letter-spacing: 1px;
}

.panel {
    border: 1px solid #3a4032;
    background: linear-gradient(180deg, #181c14, #10130e);
    padding: 22px;
    border-radius: 5px;
    margin-bottom: 20px;
}

.panel-title {
    color: #f0aa00;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    border-bottom: 1px solid #343a2f;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

.status-box {
    text-align: center;
    padding: 25px;
    border: 1px solid #3d4638;
    background: #0d100c;
    border-radius: 5px;
}

.status-normal {
    color: #42df88;
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    font-weight: 700;
}

.status-danger {
    color: #ff4034;
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    font-weight: 700;
}

.metric-card {
    border: 1px solid #343a2f;
    background: #11140f;
    padding: 16px;
    text-align: center;
    border-radius: 4px;
}

.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 23px;
    color: #e9e6da;
    font-weight: 700;
}

.metric-label {
    color: #858b81;
    font-size: 10px;
    letter-spacing: 1px;
}

.stButton > button {
    width: 100%;
    background: #f0a900;
    color: #080908;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 12px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #ffc13a;
    box-shadow: 0 0 20px rgba(240,169,0,0.3);
}

div[data-baseweb="input"] {
    background-color: #0d100c;
}

div[data-baseweb="select"] {
    background-color: #0d100c;
}

.footer {
    text-align: center;
    color: #555b52;
    font-size: 10px;
    letter-spacing: 1px;
    padding-top: 20px;
}

.warning-box {
    border: 1px solid #a87800;
    background: #191507;
    color: #f0c45c;
    padding: 12px;
    margin-top: 15px;
    border-radius: 4px;
    font-size: 11px;
}

.range-box {
    border: 1px solid #343a2f;
    background: #0b0e0a;
    color: #777f73;
    padding: 10px;
    margin-top: 8px;
    font-size: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="console-header">

<div style="display:flex; justify-content:space-between; align-items:center;">

<div>
<div class="console-title">⚙ FAILURE PROGNOSIS CONSOLE</div>

<div class="console-subtitle">
PREDICTIVE MAINTENANCE SYSTEM // RANDOM FOREST ENGINE
</div>

</div>

<div class="status-online">
● SYSTEM ONLINE
</div>

</div>

</div>
""", unsafe_allow_html=True)

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown(
    '<div class="panel-title">◈ QUICK MACHINE SCENARIOS</div>',
    unsafe_allow_html=True
)

scenario = st.selectbox(
    "Load a predefined operating condition",
    [
        "Manual Input",
        "Normal Operation",
        "High Torque",
        "High Tool Wear",
        "High Speed",
        "High Risk Condition"
    ]
)

scenarios = {

    "Normal Operation": {
        "air": 298.1,
        "process": 308.6,
        "speed": 1500,
        "torque": 40.0,
        "wear": 20
    },

    "High Torque": {
        "air": 298.5,
        "process": 309.2,
        "speed": 1400,
        "torque": 65.0,
        "wear": 60
    },

    "High Tool Wear": {
        "air": 298.8,
        "process": 309.5,
        "speed": 1450,
        "torque": 48.0,
        "wear": 180
    },

    "High Speed": {
        "air": 299.0,
        "process": 309.8,
        "speed": 1900,
        "torque": 48.0,
        "wear": 80
    },

    "High Risk Condition": {
        "air": 301.0,
        "process": 312.0,
        "speed": 1900,
        "torque": 70.0,
        "wear": 220
    }
}

st.markdown('</div>', unsafe_allow_html=True)

if scenario != "Manual Input":

    selected = scenarios[scenario]

    air_default = selected["air"]
    process_default = selected["process"]
    speed_default = selected["speed"]
    torque_default = selected["torque"]
    wear_default = selected["wear"]

else:

    air_default = 298.1
    process_default = 308.6
    speed_default = 1500
    torque_default = 40.0
    wear_default = 20

st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown(
    '<div class="panel-title">◈ LIVE SENSOR TELEMETRY</div>',
    unsafe_allow_html=True
)

st.caption(
    "Input fields accept values beyond the training distribution. "
    "The model will flag such inputs before prediction."
)

col1, col2 = st.columns(2)

with col1:

    air_temp = st.number_input(
        "AIR TEMPERATURE [K]",
        min_value=-1000000.0,
        max_value=1000000.0,
        value=float(air_default),
        step=0.1
    )

    process_temp = st.number_input(
        "PROCESS TEMPERATURE [K]",
        min_value=-1000000.0,
        max_value=1000000.0,
        value=float(process_default),
        step=0.1
    )

    rotational_speed = st.number_input(
        "ROTATIONAL SPEED [RPM]",
        min_value=-1000000,
        max_value=1000000,
        value=int(speed_default),
        step=10
    )

with col2:

    torque = st.number_input(
        "TORQUE [Nm]",
        min_value=-1000000.0,
        max_value=1000000.0,
        value=float(torque_default),
        step=0.5
    )

    tool_wear = st.number_input(
        "TOOL WEAR [min]",
        min_value=-1000000,
        max_value=1000000,
        value=int(wear_default),
        step=1
    )

    st.markdown(
        f"""
        <div class="range-box">
        TRAINING DATA RANGES<br><br>
        Air Temperature: {TRAINING_RANGES["air_temp"][0]} – {TRAINING_RANGES["air_temp"][1]} K<br>
        Process Temperature: {TRAINING_RANGES["process_temp"][0]} – {TRAINING_RANGES["process_temp"][1]} K<br>
        Rotational Speed: {TRAINING_RANGES["rotational_speed"][0]} – {TRAINING_RANGES["rotational_speed"][1]} RPM<br>
        Torque: {TRAINING_RANGES["torque"][0]} – {TRAINING_RANGES["torque"][1]} Nm<br>
        Tool Wear: {TRAINING_RANGES["tool_wear"][0]} – {TRAINING_RANGES["tool_wear"][1]} min
        </div>
        """,
        unsafe_allow_html=True
    )

out_of_range = []

if not (TRAINING_RANGES["air_temp"][0] <= air_temp <= TRAINING_RANGES["air_temp"][1]):
    out_of_range.append("Air Temperature")

if not (TRAINING_RANGES["process_temp"][0] <= process_temp <= TRAINING_RANGES["process_temp"][1]):
    out_of_range.append("Process Temperature")

if not (TRAINING_RANGES["rotational_speed"][0] <= rotational_speed <= TRAINING_RANGES["rotational_speed"][1]):
    out_of_range.append("Rotational Speed")

if not (TRAINING_RANGES["torque"][0] <= torque <= TRAINING_RANGES["torque"][1]):
    out_of_range.append("Torque")

if not (TRAINING_RANGES["tool_wear"][0] <= tool_wear <= TRAINING_RANGES["tool_wear"][1]):
    out_of_range.append("Tool Wear")

if out_of_range:

    st.markdown(
        f"""
        <div class="warning-box">
        ⚠ OUTSIDE TRAINING DISTRIBUTION<br><br>
        The following inputs are outside the values observed during model training:
        <b>{", ".join(out_of_range)}</b><br><br>
        Prediction will still be performed, but reliability may be lower because
        the Random Forest has not seen these operating conditions during training.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    f"""
    <div style="
        margin-top:20px;
        padding:12px;
        border:1px solid #343a2f;
        background:#0b0e0a;
        color:#888f84;
        font-size:11px;
    ">
    MODEL DECISION THRESHOLD<br>
    <span style="color:#f0a900;font-size:20px;">
    {threshold:.2f}
    </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown(
    '<div class="panel-title">◈ DIAGNOSTIC ENGINE</div>',
    unsafe_allow_html=True
)

if st.button("⚡ ANALYZE MACHINE"):

    input_data = np.array([
        [
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]
    ])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(probability >= threshold)

    st.session_state.probability = probability
    st.session_state.prediction = prediction

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.probability is not None:

    probability = st.session_state.probability
    prediction = st.session_state.prediction

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">◈ DIAGNOSTIC RESULT</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1])

    with left:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 40,
                        "color": "#e9e6da"
                    }
                },

                title={
                    "text": "FAILURE PROBABILITY",
                    "font": {
                        "size": 13,
                        "color": "#888f84"
                    }
                },

                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#666d61"
                    },

                    "bar": {
                        "color": "#f0a900"
                    },

                    "bgcolor": "#0c0f0b",

                    "borderwidth": 1,

                    "bordercolor": "#343a2f",

                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#10251a"
                        },
                        {
                            "range": [30, 60],
                            "color": "#2b230b"
                        },
                        {
                            "range": [60, 100],
                            "color": "#35120f"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=330,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e9e6da"
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:

        if prediction == 0:

            st.markdown("""
            <div class="status-box">

            <div style="font-size:45px;">
            ✓
            </div>

            <div class="status-normal">
            MACHINE NORMAL
            </div>

            <p style="color:#777f73;">
            No failure condition detected
            </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="status-box">

            <div style="font-size:45px;">
            ⚠
            </div>

            <div class="status-danger">
            FAILURE DETECTED
            </div>

            <p style="color:#a8aaa0;">
            Potential machine failure condition
            </p>

            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-value">
            {probability * 100:.2f}%
            </div>

            <div class="metric-label">
            MODEL FAILURE PROBABILITY
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.probability is not None:

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">◈ CURRENT SENSOR STATE</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    values = [
        (c1, "AIR TEMP", f"{air_temp:.1f} K"),
        (c2, "PROCESS TEMP", f"{process_temp:.1f} K"),
        (c3, "ROTATIONAL SPEED", f"{rotational_speed} RPM"),
        (c4, "TORQUE", f"{torque:.1f} Nm"),
        (c5, "TOOL WEAR", f"{tool_wear} min")
    ]

    for col, label, value in values:

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {value}
                </div>

                <div class="metric-label">
                {label}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown(
    '<div class="panel-title">◈ MODEL FEATURE IMPORTANCE</div>',
    unsafe_allow_html=True
)

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": [
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],
    "Importance": importance
}).sort_values("Importance")

fig = go.Figure(
    go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker=dict(
            color=importance_df["Importance"],
            colorscale=[
                [0, "#343b30"],
                [0.5, "#a87800"],
                [1, "#f0a900"]
            ]
        )
    )
)

fig.update_layout(
    height=350,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    xaxis=dict(
        title="Importance",
        gridcolor="#242a22",
        color="#858b81"
    ),

    yaxis=dict(
        color="#c9c8bd"
    ),

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=40
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": False}
)

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("MODEL INFORMATION"):

    st.markdown("""
    ### Random Forest Classifier

    **Model:** Random Forest

    **Number of estimators:** 300

    **Class weighting:** Balanced

    **Decision threshold:** 0.35

    The final Random Forest model was selected after comparing
    multiple approaches for the highly imbalanced machine-failure
    prediction problem.

    ### Input Features

    - Air Temperature
    - Process Temperature
    - Rotational Speed
    - Torque
    - Tool Wear

    ### Evaluation Metrics

    Machine failure is a highly imbalanced classification problem.
    Therefore, accuracy alone is not sufficient.

    The project evaluates:

    - ROC-AUC
    - PR-AUC
    - Precision
    - Recall
    - F1-score

    The final model prioritizes detection of actual failures rather
    than simply maximizing normal-operation accuracy.
    """)

st.markdown("""
<div class="footer">

PREDICTIVE MAINTENANCE // FAILURE PROGNOSIS CONSOLE

<br>

RANDOM FOREST INFERENCE ENGINE // MECHANICAL SYSTEM MONITORING

</div>
""", unsafe_allow_html=True)