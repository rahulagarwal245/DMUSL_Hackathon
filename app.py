import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
    padding: 2rem;
}
h1, h2, h3 { color: #f8fafc; }
label { color: #cbd5f5 !important; }
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
}
.card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 25px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load models ----------------
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
pca = pickle.load(open("artifacts/pca.pkl", "rb"))
kmeans = pickle.load(open("artifacts/kmeans.pkl", "rb"))

# ---------------- Cluster Profiles ----------------
CLUSTERS = {
    0: ("Cash Crunch Users", "High cash advance usage, high utilization, low repayment discipline."),
    1: ("Light & Responsible Users", "Low card usage, low utilization, disciplined repayment."),
    2: ("Power Spenders", "High purchases, frequent usage, moderate utilization.")
}

# ---------------- Header ----------------
st.title("💳 Customer Segmentation Dashboard")
st.caption("PCA + Clustering based credit card customer profiling")

st.divider()

# ================= INPUT SECTIONS =================

with st.expander("📘 Core Credit Behaviour Variables", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        BALANCE = st.number_input("BALANCE", min_value=0.0)
        BALANCE_FREQUENCY = st.number_input("BALANCE_FREQUENCY (0–1)", 0.0, 1.0)
        PURCHASES = st.number_input("PURCHASES", min_value=0.0)
        ONEOFF_PURCHASES = st.number_input("ONEOFF_PURCHASES", min_value=0.0)
        INSTALLMENTS_PURCHASES = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0)

    with col2:
        CASH_ADVANCE = st.number_input("CASH_ADVANCE", min_value=0.0)
        PURCHASES_FREQUENCY = st.number_input("PURCHASES_FREQUENCY (0–1)", 0.0, 1.0)
        ONEOFFPURCHASESFREQUENCY = st.number_input("ONEOFF_PURCHASES_FREQUENCY (0–1)", 0.0, 1.0)
        PURCHASESINSTALLMENTSFREQUENCY = st.number_input("INSTALLMENTS_PURCHASES_FREQUENCY (0–1)", 0.0, 1.0)

    with col3:
        CASHADVANCEFREQUENCY = st.number_input("CASH_ADVANCE_FREQUENCY (0–1)", 0.0, 1.0)
        CASHADVANCETRX = st.number_input("CASH_ADVANCE_TRX", min_value=0)
        PURCHASES_TRX = st.number_input("PURCHASES_TRX", min_value=0)
        CREDIT_LIMIT = st.number_input("CREDIT_LIMIT", min_value=0.0)
        PAYMENTS = st.number_input("PAYMENTS", min_value=0.0)

with st.expander("📊 Repayment & Tenure Variables"):
    col4, col5 = st.columns(2)
    with col4:
        MINIMUM_PAYMENTS = st.number_input("MINIMUM_PAYMENTS", min_value=0.0)
        PRCFULLPAYMENT = st.number_input("PRC_FULL_PAYMENT (0–1)", 0.0, 1.0)
    with col5:
        TENURE = st.number_input("TENURE (months)", min_value=1)

with st.expander("📈 Derived Financial Metrics (Used During Training)"):
    col6, col7 = st.columns(2)
    with col6:
        AVG_MONTHLY_SPEND = st.number_input("Avg Monthly Spend", min_value=0.0)
        CREDIT_UTILIZATION = st.number_input("Credit Utilization Ratio", min_value=0.0)
        CASH_ADVANCE_RATIO = st.number_input("Cash Advance Ratio", min_value=0.0)
        PAYMENT_BALANCE_RATIO = st.number_input("Payment–Balance Ratio", min_value=0.0)
    with col7:
        AVG_TRX_PER_MONTH = st.number_input("Avg Transactions / Month", min_value=0.0)
        CASH_ADV_TRX_RATIO = st.number_input("Cash Advance Transaction Ratio", min_value=0.0)
        PAY_FULL_BALANCE_RATIO = st.number_input("Pay Full Balance Ratio", min_value=0.0)

st.divider()

# ================= PREDICTION =================
if st.button("🔍 Identify Customer Segment"):
    input_data = np.array([[
        BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
        INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
        ONEOFFPURCHASESFREQUENCY, PURCHASESINSTALLMENTSFREQUENCY,
        CASHADVANCEFREQUENCY, CASHADVANCETRX, PURCHASES_TRX,
        CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRCFULLPAYMENT,
        TENURE, AVG_MONTHLY_SPEND, CREDIT_UTILIZATION,
        CASH_ADVANCE_RATIO, PAYMENT_BALANCE_RATIO,
        AVG_TRX_PER_MONTH, CASH_ADV_TRX_RATIO, PAY_FULL_BALANCE_RATIO
    ]])

    scaled = scaler.transform(input_data)
    cluster = int(kmeans.predict(pca.transform(scaled))[0])
    name, desc = CLUSTERS[cluster]

    st.markdown(f"""
    <div class="card">
        <h2>Cluster {cluster} — {name}</h2>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)
