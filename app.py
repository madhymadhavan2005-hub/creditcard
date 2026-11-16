from datetime import datetime
import sqlite3
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="SentinelGuard Advanced SOC Cockpit",
    page_icon="🛡️",
    layout="wide",
)

# Custom Charcoal & Amber Gold SOC Styling with High-Contrast Metrics
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #1E2129;
        border-left: 4px solid #F39C12;
        padding: 12px;
        border-radius: 6px;
    }
    .stMetric label {
        color: #A6ACAF !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load ML Artifacts (Cached for performance)
@st.cache_resource
def load_artifacts():
  model = joblib.load("xgb_fraud_model.pkl")
  scaler = joblib.load("scaler.pkl")
  feature_order = joblib.load("feature_order.pkl")
  return model, scaler, feature_order


model, scaler, feature_order = load_artifacts()


# Initialize SQLite Database for SOC Audit Trail
def init_db():
  conn = sqlite3.connect("sentinel_audit.db", check_same_thread=False)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            time_val REAL,
            amount REAL,
            probability REAL,
            prediction TEXT,
            status TEXT
        )
    """)
  conn.commit()
  conn.close()


init_db()


def log_transaction(time_val, amount, prob, pred_label):
  conn = sqlite3.connect("sentinel_audit.db", check_same_thread=False)
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO audit_logs (timestamp, time_val, amount, probability, prediction, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
      (
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          time_val,
          amount,
          prob,
          pred_label,
          "Pending Review",
      ),
  )
  conn.commit()
  conn.close()


st.title("🛡️ SentinelGuard: Enterprise SOC Telemetry & Fraud Cockpit")
st.write(
    "Automated risk telemetry evaluated directly from transaction Time and"
    " Amount."
)

st.markdown("---")

dashboard_mode = st.radio(
    "Select System Module:",
    [
        "Single Transaction Analyzer",
        "Batch CSV Processor",
        "SOC Analyst Review Queue",
    ],
    horizontal=True,
)

# ==============================================================================
# MODULE 1: SINGLE TRANSACTION ANALYZER
# ==============================================================================
if dashboard_mode == "Single Transaction Analyzer":
  st.subheader("🔍 Automated Transaction Telemetry")

  col1, col2 = st.columns(2)
  with col1:
    time_val = st.number_input(
        "Transaction Time (Seconds)", value=406.0, format="%.2f"
    )
  with col2:
    amount_val = st.number_input(
        "Transaction Amount ($)", value=45.00, format="%.2f"
    )

  if st.button("🚀 Run Full Telemetry Scan", use_container_width=True):
    with st.spinner("Analyzing transaction vectors..."):
      # Input data strictly from Time and Amount
      input_data = pd.DataFrame([[time_val, amount_val]], columns=feature_order)
      input_scaled = scaler.transform(input_data)

      prediction = int(model.predict(input_scaled)[0])
      probability = float(model.predict_proba(input_scaled)[0][1])
      probability_pct = round(probability * 100, 2)
      pred_label = "Fraud" if prediction == 1 else "Legitimate"

      log_transaction(time_val, amount_val, probability_pct, pred_label)
      confidence = round(abs(probability_pct - 50.0) * 2, 2)

      st.markdown("---")
      m1, m2, m3, m4 = st.columns(4)
      m1.metric("Fraud Probability", f"{probability_pct}%")
      m2.metric("Model Certainty", f"{confidence}%")
      m3.metric(
          "Threat Tier",
          "HIGH RISK"
          if probability_pct >= 75
          else ("MEDIUM RISK" if probability_pct >= 20 else "LOW RISK"),
      )
      m4.metric("Primary Driver", "Amount / Time Pattern")

      st.markdown("")

      g1_col, g2_col, v_col = st.columns([1, 1, 1.2])

      with g1_col:
        fig_risk_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_pct,
                title={"text": "Threat Risk Index", "font": {"size": 15}},
                number={"suffix": "%", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#F39C12"},
                    "steps": [
                        {"range": [0, 20], "color": "#27AE60"},
                        {"range": [20, 75], "color": "#D68910"},
                        {"range": [75, 100], "color": "#C0392B"},
                    ],
                },
            )
        )
        fig_risk_gauge.update_layout(
            height=280, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(
            fig_risk_gauge, use_container_width=True, key="risk_gauge"
        )

      with g2_col:
        fig_conf_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Classifier Confidence", "font": {"size": 15}},
                number={"suffix": "%", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2980B9"},
                    "steps": [
                        {"range": [0, 40], "color": "#5D6D7E"},
                        {"range": [40, 80], "color": "#2E86C1"},
                        {"range": [80, 100], "color": "#1ABC9C"},
                    ],
                },
            )
        )
        fig_conf_gauge.update_layout(
            height=280, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(
            fig_conf_gauge, use_container_width=True, key="conf_gauge"
        )

      with v_col:
        st.markdown("### 📋 Automated Security Verdict")
        if probability_pct < 20:
          st.success(
              "🟢 **LOW RISK: Authorized Pattern**\n\nTransaction amount and"
              " timing align with normal behavior."
          )
        elif probability_pct < 75:
          st.warning(
              "🟡 **MEDIUM RISK: Flagged for Triage**\n\nModerate anomaly"
              " detected. Queued for analyst review."
          )
        else:
          st.error(
              "🔴 **HIGH RISK: Fraud Signature Locked**\n\nHigh-risk amount/time"
              " signature detected. Immediate block recommended."
          )

# ==============================================================================
# MODULE 2: BATCH CSV PROCESSOR
# ==============================================================================
elif dashboard_mode == "Batch CSV Processor":
  st.subheader("📁 Bulk Transaction Batch Telemetry")
  uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

  if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded **{len(input_df)}** records from file.")

    if st.button("⚡ Process & Visualize Batch", use_container_width=True):
      with st.spinner("Running batch inference..."):
        batch_results = []
        for _, tx in input_df.head(50).iterrows():
          t_val = tx.get("Time", 0.0)
          a_val = tx.get("Amount", 0.0)

          row_df = pd.DataFrame([[t_val, a_val]], columns=feature_order)
          row_scaled = scaler.transform(row_df)

          pred = int(model.predict(row_scaled)[0])
          prob = round(float(model.predict_proba(row_scaled)[0][1]) * 100, 2)
          pred_label = "Fraud" if pred == 1 else "Legitimate"

          log_transaction(t_val, a_val, prob, pred_label)
          batch_results.append({
              "Time": t_val,
              "Amount": a_val,
              "Prediction": pred_label,
              "Risk_Probability_Percent": prob,
          })

        result_df = pd.DataFrame(batch_results)
        st.success("Batch evaluation complete!")
        st.dataframe(result_df, use_container_width=True)

# ==============================================================================
# MODULE 3: SOC ANALYST REVIEW QUEUE
# ==============================================================================
else:
  st.subheader("🛡️ SOC Live Audit Analytics & Case Triage")
  conn = sqlite3.connect("sentinel_audit.db", check_same_thread=False)
  try:
    logs_df = pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)
    if not logs_df.empty:
      st.dataframe(logs_df, use_container_width=True)
    else:
      st.info("No audit logs found yet. Run a scan first!")
  except Exception as e:
    st.error(f"Error reading database: {e}")
  finally:
    conn.close()