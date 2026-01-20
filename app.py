import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# ---------------- Load artifacts ----------------
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
pca = pickle.load(open("artifacts/pca.pkl", "rb"))
kmeans = pickle.load(open("artifacts/kmeans.pkl", "rb"))

# ---------------- Cluster Profiles ----------------
CLUSTERS = {
    0: {
        "name": "Cash Crunch Users",
        "description": [
            "Primarily use credit cards for cash advances rather than purchases",
            "Highest credit utilization (~75%)",
            "Low full-payment rate; balances are often carried forward",
            "High dependency on minimum payments"
        ],
        "risk": "High",
        "color": "#E74C3C"
    },
    1: {
        "name": "Light & Responsible Users",
        "description": [
            "Lowest overall card usage (low purchases and cash advances)",
            "Lowest credit utilization (~16%)",
            "Better repayment behaviour than Cluster 0",
            "Typically disciplined, low-risk customers"
        ],
        "risk": "Low",
        "color": "#2ECC71"
    },
    2: {
        "name": "Power Spenders",
        "description": [
            "Highest purchase volumes with frequent card usage",
            "Active in both one-time and installment purchases",
            "Higher credit limits with moderate utilization (~40%)",
            "Payments are regular but not always full"
        ],
        "risk": "Medium",
        "color": "#F39C12"
    }
}

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.title("💳 Customer Segmentation & Behaviour Dashboard")
st.caption("PCA + Clustering based customer profiling for credit card users")

st.divider()

# ---------------- Input Section ----------------
st.subheader("📥 Customer Behaviour Inputs")

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
    ONEOFFPURCHASESFREQUENCY = st.number_input("ONEOFFPURCHASESFREQUENCY (0–1)", 0.0, 1.0)
    PURCHASESINSTALLMENTSFREQUENCY = st.number_input("INSTALLMENTS_FREQUENCY (0–1)", 0.0, 1.0)
    CASHADVANCEFREQUENCY = st.number_input("CASH_ADVANCE_FREQUENCY (0–1)", 0.0, 1.0)

with col3:
    CASHADVANCETRX = st.number_input("CASH_ADVANCE_TRX", min_value=0)
    PURCHASES_TRX = st.number_input("PURCHASES_TRX", min_value=0)
    CREDIT_LIMIT = st.number_input("CREDIT_LIMIT", min_value=0.0)
    PAYMENTS = st.number_input("PAYMENTS", min_value=0.0)
    MINIMUM_PAYMENTS = st.number_input("MINIMUM_PAYMENTS", min_value=0.0)
    PRCFULLPAYMENT = st.number_input("PRC_FULL_PAYMENT (0–1)", 0.0, 1.0)
    TENURE = st.number_input("TENURE (months)", min_value=1)

st.divider()

st.subheader("📊 Derived Financial Metrics")

col4, col5, col6 = st.columns(3)

with col4:
    AVG_MONTHLY_SPEND = st.number_input("Avg Monthly Spend", min_value=0.0)
    CREDIT_UTILIZATION = st.number_input("Credit Utilization Ratio", min_value=0.0)

with col5:
    CASH_ADVANCE_RATIO = st.number_input("Cash Advance Ratio", min_value=0.0)
    PAYMENT_BALANCE_RATIO = st.number_input("Payment–Balance Ratio", min_value=0.0)

with col6:
    AVG_TRX_PER_MONTH = st.number_input("Avg Transactions / Month", min_value=0.0)
    CASH_ADV_TRX_RATIO = st.number_input("Cash Advance Transaction Ratio", min_value=0.0)
    PAY_FULL_BALANCE_RATIO = st.number_input("Pay Full Balance Ratio", min_value=0.0)

st.divider()

# ---------------- Prediction ----------------
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
    pca_out = pca.transform(scaled)
    cluster = int(kmeans.predict(pca_out)[0])

    profile = CLUSTERS[cluster]

    st.markdown("---")
    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;background-color:{profile['color']}20">
            <h2>Cluster {cluster} — {profile['name']}</h2>
            <b>Risk Profile:</b> {profile['risk']}
        </div>
        """,
        unsafe_allow_html=True
    )

    for point in profile["description"]:
        st.write(f"• {point}")

    st.markdown("### 📈 Behaviour Snapshot")

    radar_df = pd.DataFrame({
        "Metric": ["Credit Utilization", "Cash Advance Usage", "Purchase Activity", "Repayment Discipline"],
        "Value": [
            CREDIT_UTILIZATION,
            CASH_ADVANCE_RATIO,
            PURCHASES_FREQUENCY,
            PRCFULLPAYMENT
        ]
    })

    fig = px.bar(
        radar_df,
        x="Metric",
        y="Value",
        color="Metric",
        range_y=[0, 1],
        title="Customer Behaviour Indicators"
    )

    st.plotly_chart(fig, use_container_width=True)
