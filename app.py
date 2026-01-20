import streamlit as st
import numpy as np
import pickle
import pandas as pd

# --------------------------------------------------
# Load trained artifacts
# --------------------------------------------------
with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("artifacts/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

# --------------------------------------------------
# Cluster descriptions (business interpretation)
# --------------------------------------------------
CLUSTER_DESCRIPTION = {
    0: "Low spending, low engagement customers with conservative card usage.",
    1: "High spending customers who actively use credit cards and repay regularly.",
    2: "Customers with high cash advance usage and higher credit utilisation risk.",
    3: "Long-tenure customers with stable usage and disciplined repayment behaviour."
}

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Strategy Tool",
    layout="wide"
)

st.title("Customer Segmentation & Behaviour Analysis")
st.markdown(
    "This tool identifies the **customer segment** based on credit card behaviour "
    "and provides a **business-level interpretation**."
)

st.divider()

# --------------------------------------------------
# Input guidance table
# --------------------------------------------------
st.subheader("📋 Input Variable Reference")

input_reference = pd.DataFrame({
    "Variable": [
        "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
        "ONEOFFPURCHASESFREQUENCY", "PURCHASESINSTALLMENTSFREQUENCY",
        "CASHADVANCEFREQUENCY", "CASHADVANCETRX", "PURCHASES_TRX",
        "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
        "PRCFULLPAYMENT", "TENURE"
    ],
    "Expected Range": [
        "≥ 0", "0 – 1", "≥ 0", "≥ 0",
        "≥ 0", "≥ 0", "0 – 1",
        "0 – 1", "0 – 1",
        "0 – 1", "≥ 0", "≥ 0",
        "≥ 0", "≥ 0", "≥ 0",
        "0 – 1", "≥ 1 (months)"
    ]
})

st.dataframe(input_reference, use_container_width=True)

st.divider()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.subheader("🧮 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    BALANCE = st.number_input("BALANCE", min_value=0.0)
    PURCHASES = st.number_input("PURCHASES", min_value=0.0)
    INSTALLMENTS_PURCHASES = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0)
    PURCHASES_FREQUENCY = st.number_input("PURCHASES_FREQUENCY (0–1)", min_value=0.0, max_value=1.0)
    CASHADVANCEFREQUENCY = st.number_input("CASHADVANCEFREQUENCY (0–1)", min_value=0.0, max_value=1.0)
    CREDIT_LIMIT = st.number_input("CREDIT_LIMIT", min_value=0.0)

with col2:
    BALANCE_FREQUENCY = st.number_input("BALANCE_FREQUENCY (0–1)", min_value=0.0, max_value=1.0)
    ONEOFF_PURCHASES = st.number_input("ONEOFF_PURCHASES", min_value=0.0)
    CASH_ADVANCE = st.number_input("CASH_ADVANCE", min_value=0.0)
    ONEOFFPURCHASESFREQUENCY = st.number_input("ONEOFFPURCHASESFREQUENCY (0–1)", min_value=0.0, max_value=1.0)
    CASHADVANCETRX = st.number_input("CASHADVANCETRX", min_value=0)
    PAYMENTS = st.number_input("PAYMENTS", min_value=0.0)

with col3:
    PURCHASESINSTALLMENTSFREQUENCY = st.number_input(
        "PURCHASESINSTALLMENTSFREQUENCY (0–1)", min_value=0.0, max_value=1.0
    )
    PURCHASES_TRX = st.number_input("PURCHASES_TRX", min_value=0)
    MINIMUM_PAYMENTS = st.number_input("MINIMUM_PAYMENTS", min_value=0.0)
    PRCFULLPAYMENT = st.number_input("PRCFULLPAYMENT (0–1)", min_value=0.0, max_value=1.0)
    TENURE = st.number_input("TENURE (months)", min_value=1)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
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
            TENURE
        ]])

        scaled_input = scaler.transform(input_data)
        pca_input = pca.transform(scaled_input)
        cluster = int(kmeans.predict(pca_input)[0])

        st.success(f"Customer belongs to **Cluster {cluster}**")
        st.markdown(f"**Customer Profile:** {CLUSTER_DESCRIPTION[cluster]}")

    except Exception as e:
        st.error("Unable to process input. Please verify all values.")
