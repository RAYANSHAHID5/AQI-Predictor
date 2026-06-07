"""
streamlit web dashboard
connects to Hopsworks feature store to retrieve historical AQI data and the
registered XGBoost model provides real time AQI predictions and an interactive EDA
suite proving environmental correlations.
"""
import streamlit as st
import hopsworks
import joblib
import pandas as pd
import os
from PIL import Image

# UI Configuration
st.set_page_config(page_title="Lahore AQI Predictor", page_icon="🌤️", layout="wide")


st.markdown(
    """
    <style>
    /* 1. Container Width */
    .block-container {
        max-width: 1800px; 
        padding-top: 2rem;
    }

    /* 2. Global Text: Targets all standard text, lists, and markdown */
    html, body, [class*="css"] {
        font-size: 24px !important;
    }
    p, li, span {
        font-size: 24px !important;
    }

    /* 3. Headers: Massive scaling for titles */
    h1 { font-size: 4.5rem !important; padding-bottom: 1rem; }
    h2 { font-size: 3.5rem !important; padding-bottom: 1rem; }
    h3 { font-size: 2.5rem !important; }
    h4 { font-size: 2.0rem !important; }

    /* 4. Metrics: The numbers at the top */
    [data-testid="stMetricLabel"] > div > p {
        font-size: 28px !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 85px !important; /* Huge numbers */
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 24px !important;
    }

    /* 5. Tabs: The interactive buttons for the charts */
    button[data-baseweb="tab"] > div > p {
        font-size: 24px !important;
        font-weight: bold !important;
    }

    /* 6. Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        font-size: 22px !important;
    }

    /* 7. Big Red Button */
    .stButton>button {
        font-size: 24px !important;
        font-weight: bold !important;
        height: 3rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
@st.cache_resource
def load_model():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    mr = project.get_model_registry()
    model_handle = mr.get_model("lahore_aqi_model", version=3)
    model_dir = model_handle.download()
    return joblib.load(model_dir + "/aqi_model.pkl")


@st.cache_data
def load_history():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="lahore_aqi_features", version=2)
    return aqi_fg.read()


# Sidebar Controls
st.sidebar.title("🌍 AQI Predictor Hub")
st.sidebar.markdown("### Project Specifications")
st.sidebar.info("Continuous serverless forecasting for DHA Phase 8, Lahore, Pakistan.")
st.sidebar.markdown("### Engineering Team")
st.sidebar.text("Rayan Shahid")

# Main Application Header
st.title("🌍 Lahore AQI Predictor (DHA Phase 8)")
st.markdown("An automated, end-to-end MLOps engine utilizing serverless features and predictive AI.")

with st.spinner("Synchronizing with Hopsworks Cloud Feature Store..."):
    try:
        model = load_model()
        history_df = load_history()
        history_df = history_df.sort_values("timestamp")
        latest_data = history_df.tail(1).iloc[0]
        st.toast("Feature store synchronized successfully.", icon="✅")
    except Exception as e:
        st.error(f"Cloud Connection Failed: {e}")
        st.stop()

# Section 1: Real-Time Environmental Telemetry
st.subheader("📡 Live Cloud Telemetry")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Current AQI",
              value=int(latest_data['target_aqi']),
              delta=int(latest_data.get('aqi_change_rate', 0)),
              delta_color="inverse")
with col2:
    st.metric(label="PM 2.5 Concentration", value=f"{latest_data['pm25']} µg/m³")
with col3:
    st.metric(label="Ambient Temperature", value=f"{latest_data['temp']} °C")
with col4:
    st.metric(label="Relative Humidity", value=f"{latest_data['humidity']}%")

st.divider()

# Section 2: Predictive Inference Engine
st.subheader("🔮 Predictive Machine Learning Inference")
predict_col, result_col = st.columns([1, 2])

with predict_col:
    st.markdown("Compute the immediate forward prediction using the active cloud feature vector.")
    predict_trigger = st.button("Execute Next-Hour Forecast", type="primary", use_container_width=True)

with result_col:
    if predict_trigger:
        features = history_df.tail(1).copy().drop(columns=["timestamp", "target_aqi"], errors='ignore')
        prediction = model.predict(features)[0]

        if prediction < 50:
            st.success(f"**Predicted AQI: {round(prediction, 1)}** — Category: Good 🌿")
        elif prediction < 100:
            st.warning(f"**Predicted AQI: {round(prediction, 1)}** — Category: Moderate 🟡")
        else:
            st.error(f"**Predicted AQI: {round(prediction, 1)}** — Category: Unhealthy 😷")
    else:
        st.info("System idling. Awaiting inference trigger execution.")

st.divider()

# Section 3: Exploratory Data Analysis
st.subheader("📊 Interactive Exploratory Data Analysis (EDA)")
st.markdown(
    "This section provides empirical, historical data analysis verifying environmental and behavioral patterns.")

tab1, tab2, tab3 = st.tabs(["🕒 Traffic & Time Analysis", "🌤️ Weather Correlation Analytics", "🧠 Explainable AI (SHAP)"])

with tab1:
    st.markdown("#### **Empirical Proof of Human Activity Cycles**")
    st.markdown(
        "The chart below aggregates all historical records by the hour of the day. Notice the distinct, repeatable peaks during morning and evening rush hours, confirming the impact of local vehicle emissions.")

    # Calculate historical average AQI per hour to prove traffic peaks
    hourly_profile = history_df.groupby("hour")["target_aqi"].mean().reset_index()
    hourly_profile = hourly_profile.rename(columns={"target_aqi": "Average AQI"})
    st.bar_chart(hourly_profile.set_index("hour"))

with tab2:
    st.markdown("#### **Multi-Feature Historical Explorer**")
    st.markdown(
        "Select an environmental feature below to investigate its raw timeline variance alongside historical air quality trends.")

    # Allow user to toggle columns interactively to analyze raw trends
    available_features = [col for col in history_df.columns if col not in ["timestamp"]]
    selected_feature = st.selectbox("Select Target Feature for Visualization:", available_features, index=0)
    st.line_chart(history_df.set_index("timestamp")[selected_feature])

    st.markdown("#### **Feature Correlation Insights**")
    # Generate an interactive numeric correlation data frame to prove atmospheric effects
    corr_df = history_df[['target_aqi', 'pm25', 'pm10', 'temp', 'humidity', 'wind_speed']].corr()[
        ['target_aqi']].reset_index()
    corr_df = corr_df.rename(
        columns={"index": "Environmental Feature", "target_aqi": "Correlation Coefficient with AQI"})
    st.dataframe(corr_df.style.background_gradient(cmap="Blues"), use_container_width=True)

with tab3:
    st.markdown("#### **Algorithmic Attribution via SHAP values**")
    st.markdown(
        "This summary plot illustrates the global mathematical weight assigned to each feature by the trained XGBoost model.")

    if os.path.exists("shap_summary.png"):
        image = Image.open("shap_summary.png")
        st.image(image, use_container_width=True)
    else:
        st.warning("SHAP analysis matrix image ('shap_summary.png') not detected in local directory root.")