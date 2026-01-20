import streamlit as st
import numpy as np
import pickle

# -------------------- Load Artifacts --------------------
with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("artifacts/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

from config.cluster_strategy import CLUSTER_STRATEGY

# -------------------- UI CONFIG --------------------
st.set_page_config(
    page_title="Customer Segmentation Tool",
    layout="centered"
)

st.title("Customer Segmentation & Marketing Strategy Engine")
st.markdown("Enter customer financial attributes to identify segment and strategy.")

st.divider()

# -------------------- USER INPUTS --------------------
# 👉 USE ONLY VARIABLES THAT YOU ALREADY USED IN PCA
BALANCE = st.number_input("Balance", min_value=0.0)
PURCHASES = st.number_input("Purchases", min_value=0.0)
CASH_ADVANCE = st.number_input("Cash Advance", min_value=0.0)
CREDIT_LIMIT = st.number_input("Credit Limit", min_value=0.0)
PAYMENTS = st.number_input("Payments", min_value=0.0)

# Keep order EXACTLY as used during training
input_data = np.array([[BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS]])

# -------------------- PREDICTION --------------------
if st.button("Identify Customer Segment"):
    try:
        scaled_input = scaler.transform(input_data)
        pca_input = pca.transform(scaled_input)
        cluster = kmeans.predict(pca_input)[0]

        st.success(f"Customer belongs to **Cluster {cluster}**")

        st.subheader("Recommended Marketing Strategy")
        for point in CLUSTER_STRATEGY[cluster]:
            st.write(f"• {point}")

    except Exception as e:
        st.error("Input processing failed. Please verify values.")
