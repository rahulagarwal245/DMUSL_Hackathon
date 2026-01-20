import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ------------------ Load Models ------------------
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
pca = pickle.load(open("artifacts/pca.pkl", "rb"))
kmeans = pickle.load(open("artifacts/kmeans.pkl", "rb"))

# ------------------ Cluster Descriptions ------------------
CLUSTER_DESCRIPTION = {
    0: "Low spending and low engagement customers with conservative card usage.",
    1: "High spending customers with active card usage and timely repayments.",
    2: "Customers with high cash advance usage and elevated credit risk.",
    3: "Stable long-tenure customers with disciplined repayment behaviour."
}

# ------------------ Page Config ------------------
st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")
st.title("Customer Segmentation & Credit Behaviour Analysis")

st.info(
    "All inputs below correspond exactly to the variables used during model training. "
    "Derived ratios are NOT used for prediction."
)

st.divider()

# ------------------ Input Table ------------------
st.subheader("Input Variables and Expected Ranges")

ref = pd.DataFrame({
    "Variable": [
        "BALANCE","BALANCE_FREQUENCY","PURCHASES","ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES","CASH_ADVANCE","PURCHASES_FREQUENCY",
        "ONEOFFPURCHASESFREQUENCY","PURCHASESINSTALLMENTSFREQUENCY",
        "CASHADVANCEFREQUENCY","CASHADVANCETRX","PURCHASES_TRX",
        "CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS",
        "PRCFULLPAYMENT","TENURE"
    ],
    "Range": [
        "≥0","0–1","≥0","≥0","≥0","≥0","0–1","0–1","0–1",
        "0–1","≥0","≥0","≥0","≥0","≥0","0–1","≥1"
    ]
})

st.dataframe(ref, use_container_width=True)

st.divider()

# ------------------ Inputs ------------------
col1, col2, col3 = st.columns(3)

with col1:
    BALANCE = st.number_input("BALANCE", min_value=0.0)
    PURCHASES = st.number_input("PURCHASES", min_value=0.0)
    INSTALLMENTS_PURCHASES = st.number_input("INSTALLMENTS_PURCHASES", min_value=0.0)
    PURCHASES_FREQUENCY = st.number_input("PURCHASES_FREQUENCY", 0.0, 1.0)

with col2:
    BALANCE_FREQUENCY = st.number_input("BALANCE_FREQUENCY", 0.0, 1.0)
    ONEOFF_PURCHASES = st.number_input("ONEOFF_PURCHASES", min_value=0.0)
    CASH_ADVANCE = st.number_input("CASH_ADVANCE", min_value=0.0)
    ONEOFFPURCHASESFREQUENCY = st.number_input("ONEOFFPURCHASESFREQUENCY", 0.0, 1.0)

with col3:
    PURCHASESINSTALLMENTSFREQUENCY = st.number_input("PURCHASESINSTALLMENTSFREQUENCY", 0.0, 1.0)
    CASHADVANCEFREQUENCY = st.number_input("CASHADVANCEFREQUENCY", 0.0, 1.0)
    CASHADVANCETRX = st.number_input("CASHADVANCETRX", min_value=0)
    PURCHASES_TRX = st.number_input("PURCHASES_TRX", min_value=0)

CREDIT_LIMIT = st.number_input("CREDIT_LIMIT", min_value=0.0)
PAYMENTS = st.number_input("PAYMENTS", min_value=0.0)
MINIMUM_PAYMENTS = st.number_input("MINIMUM_PAYMENTS", min_value=0.0)
PRCFULLPAYMENT = st.number_input("PRCFULLPAYMENT", 0.0, 1.0)
TENURE = st.number_input("TENURE (months)", min_value=1)

# ------------------ Prediction ------------------
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

        scaled = scaler.transform(input_data)
        pca_out = pca.transform(scaled)
        cluster = int(kmeans.predict(pca_out)[0])

        st.success(f"Customer Segment: Cluster {cluster}")
        st.write(CLUSTER_DESCRIPTION[cluster])

    except Exception as e:
        st.error(f"Processing error: {e}")
