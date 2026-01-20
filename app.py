import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------- Load artifacts ----------------
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
pca = pickle.load(open("artifacts/pca.pkl", "rb"))
kmeans = pickle.load(open("artifacts/kmeans.pkl", "rb"))

# ---------------- Cluster descriptions ----------------
CLUSTER_DESCRIPTION = {
    0: "Low usage and low engagement customers with conservative spending behaviour.",
    1: "High spending and active customers with strong repayment discipline.",
    2: "Customers with heavy cash advance usage and higher credit risk.",
    3: "Stable long-tenure customers with balanced spending and repayments."
}

# ---------------- Page config ----------------
st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")
st.title("Customer Segmentation & Credit Behaviour Analysis")

st.info(
    "All inputs below exactly match the variables used during model training "
    "(17 original + 7 derived = 24 features)."
)

st.divider()

# ---------------- User Inputs ----------------
st.subheader("Customer Behaviour Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    BALANCE = st.number_input("BALANCE", min_value=0.0)
    BALANCE_FREQUENCY = st.number_input("BALANCE_FREQUENCY", 0.0, 1.0)
    PURCHASES = st.number_input("PURCHASES", min_value=0.0)
    ONEOFF_PURCHASES = st.number_input("ONEOFF_PURCHASES", min_value=0.0)
    INSTALLMENTS_PURCHASES = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0)
    CASH_ADVANCE = st.number_input("CASH_ADVANCE", min_value=0.0)

with col2:
    PURCHASES_FREQUENCY = st.number_input("PURCHASES_FREQUENCY", 0.0, 1.0)
    ONEOFFPURCHASESFREQUENCY = st.number_input("ONEOFFPURCHASESFREQUENCY", 0.0, 1.0)
    PURCHASESINSTALLMENTSFREQUENCY = st.number_input("PURCHASESINSTALLMENTSFREQUENCY", 0.0, 1.0)
    CASHADVANCEFREQUENCY = st.number_input("CASHADVANCEFREQUENCY", 0.0, 1.0)
    CASHADVANCETRX = st.number_input("CASHADVANCETRX", min_value=0)
    PURCHASES_TRX = st.number_input("PURCHASES_TRX", min_value=0)

with col3:
    CREDIT_LIMIT = st.number_input("CREDIT_LIMIT", min_value=0.0)
    PAYMENTS = st.number_input("PAYMENTS", min_value=0.0)
    MINIMUM_PAYMENTS = st.number_input("MINIMUM_PAYMENTS", min_value=0.0)
    PRCFULLPAYMENT = st.number_input("PRCFULLPAYMENT", 0.0, 1.0)
    TENURE = st.number_input("TENURE (months)", min_value=1)

st.divider()
st.subheader("Derived Financial Metrics (Used During Training)")

AVG_MONTHLY_SPEND = st.number_input("Avg Monthly Spend", min_value=0.0)
CREDIT_UTILIZATION = st.number_input("Credit Utilization Ratio", min_value=0.0)
CASH_ADVANCE_RATIO = st.number_input("Cash Advance Ratio", min_value=0.0)
PAYMENT_BALANCE_RATIO = st.number_input("Payment Balance Ratio", min_value=0.0)
AVG_TRX_PER_MONTH = st.number_input("Avg Transactions per Month", min_value=0.0)
CASH_ADV_TRX_RATIO = st.number_input("Cash Advance Transaction Ratio", min_value=0.0)
PAY_FULL_BALANCE_RATIO = st.number_input("Pay Full Balance Ratio", min_value=0.0)

# ---------------- Prediction ----------------
if st.button("Identify Customer Segment"):
    try:
        input_data = np.array([[
            BALANCE,
            BALANCE_FREQUENCY,
            PURCHASES,
            ONEOFF_PURCHASES,
            INSTALLMENTS_PURCHASES,
            CASH_ADVANCE,
            PURCHASES_FREQUENCY,
            ONEOFFPURCHASESFREQUENCY,
            PURCHASESINSTALLMENTSFREQUENCY,
            CASHADVANCEFREQUENCY,
            CASHADVANCETRX,
            PURCHASES_TRX,
            CREDIT_LIMIT,
            PAYMENTS,
            MINIMUM_PAYMENTS,
            PRCFULLPAYMENT,
            TENURE,
            AVG_MONTHLY_SPEND,
            CREDIT_UTILIZATION,
            CASH_ADVANCE_RATIO,
            PAYMENT_BALANCE_RATIO,
            AVG_TRX_PER_MONTH,
            CASH_ADV_TRX_RATIO,
            PAY_FULL_BALANCE_RATIO
        ]])

        scaled = scaler.transform(input_data)
        pca_out = pca.transform(scaled)
        cluster = int(kmeans.predict(pca_out)[0])

        st.success(f"Customer Segment: Cluster {cluster}")
        st.write(CLUSTER_DESCRIPTION[cluster])

    except Exception as e:
        st.error(f"Processing error: {e}")
