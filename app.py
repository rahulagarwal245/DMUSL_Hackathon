import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(120deg, #f6f9fc 0%, #eef2f7 100%);
        padding: 2rem;
    }

    h1 {
        color: #0f172a;
        font-weight: 700;
        font-size: 42px;
    }

    h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }

    .subtext {
        color: #475569;
        font-size: 16px;
    }

    .card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .cluster-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load models ----------------
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
pca = pickle.load(open("artifacts/pca.pkl", "rb"))
kmeans = pickle.load(open("artifacts/kmeans.pkl", "rb"))

# ---------------- Cluster Profiles ----------------
CLUSTERS = {
    0: {
        "name": "Cash Crunch Users",
        "risk": "High Risk",
        "risk_color": "#dc2626",
        "points": [
            "Primarily use credit cards for cash advances rather than purchases",
            "Highest credit utilization (~75%)",
            "Low full-payment rate; balances are usually carried forward"
        ]
    },
    1: {
        "name": "Light & Responsible Users",
        "risk": "Low Risk",
        "risk_color": "#16a34a",
        "points": [
            "Lowest overall card usage",
            "Lowest credit utilization (~16%)",
            "Disciplined repayment behaviour"
        ]
    },
    2: {
        "name": "Power Spenders",
        "risk": "Medium Risk",
        "risk_color": "#f59e0b",
        "points": [
            "Highest purchases and frequent spending",
            "Active use of both one-time and installment purchases",
            "Moderate credit utilization (~40%)"
        ]
    }
}

# ---------------- Page Header ----------------
st.title("💳 Customer Segmentation Dashboard")
st.markdown(
    "<p class='subtext'>Advanced customer profiling using PCA & clustering for credit card behaviour analysis</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- Inputs ----------------
st.subheader("📥 Customer Behaviour Inputs")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        BALANCE = st.number_input("BALANCE", min_value=0.0)
        PURCHASES = st.number_input("PURCHASES", min_value=0.0)
        CASH_ADVANCE = st.number_input("CASH_ADVANCE", min_value=0.0)

    with col2:
        CREDIT_LIMIT = st.number_input("CREDIT_LIMIT", min_value=0.0)
        PAYMENTS = st.number_input("PAYMENTS", min_value=0.0)
        PRCFULLPAYMENT = st.number_input("PRC_FULL_PAYMENT (0–1)", 0.0, 1.0)

    with col3:
        PURCHASES_FREQUENCY = st.number_input("PURCHASES_FREQUENCY (0–1)", 0.0, 1.0)
        CASHADVANCEFREQUENCY = st.number_input("CASH_ADVANCE_FREQUENCY (0–1)", 0.0, 1.0)
        TENURE = st.number_input("TENURE (months)", min_value=1)

st.markdown("---")

# ---------------- Prediction ----------------
if st.button("🔍 Identify Customer Segment"):
    input_data = np.zeros((1, 24))  # placeholder to match scaler shape
    input_data[0, 0] = BALANCE
    input_data[0, 2] = PURCHASES
    input_data[0, 5] = CASH_ADVANCE
    input_data[0, 6] = PURCHASES_FREQUENCY
    input_data[0, 9] = CASHADVANCEFREQUENCY
    input_data[0, 12] = CREDIT_LIMIT
    input_data[0, 15] = PRCFULLPAYMENT
    input_data[0, 16] = TENURE

    scaled = scaler.transform(input_data)
    cluster = int(kmeans.predict(pca.transform(scaled))[0])
    profile = CLUSTERS[cluster]

    # -------- Result Card --------
    st.markdown(
        f"""
        <div class="card">
            <div class="cluster-title">Cluster {cluster} — {profile['name']}</div>
            <span class="risk-badge" style="background:{profile['risk_color']}">
                {profile['risk']}
            </span>
            <ul>
                {''.join([f"<li>{p}</li>" for p in profile['points']])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- Chart --------
    chart_df = pd.DataFrame({
        "Metric": ["Credit Utilization", "Cash Advance Usage", "Purchase Frequency", "Full Payment Rate"],
        "Value": [
            BALANCE / CREDIT_LIMIT if CREDIT_LIMIT > 0 else 0,
            CASHADVANCEFREQUENCY,
            PURCHASES_FREQUENCY,
            PRCFULLPAYMENT
        ]
    })

    fig = px.bar(
        chart_df,
        x="Metric",
        y="Value",
        range_y=[0, 1],
        title="Behaviour Snapshot"
    )

    st.plotly_chart(fig, use_container_width=True)
