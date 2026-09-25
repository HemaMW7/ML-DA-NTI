# ============================================
# Streamlit App - Telco Churn Prediction
# يقارن بين Logistic Regression و Random Forest
# ============================================

import streamlit as st
import pandas as pd
import pickle

# ============================================
# تحميل الموديلات والملفات
# ============================================
with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📞")
st.title("📞 Telco Customer Churn Prediction")
st.write("Enter customer details to compare predictions from two models.")

# ============================================
# Sidebar - Inputs
# ============================================
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 18.0, 9000.0, 1000.0)

# ============================================
# تجهيز المدخلات
# ============================================
input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=columns, fill_value=0)

# ============================================
# التنبؤ بالموديلين
# ============================================
if st.button("Predict"):

    pred_rf = model_rf.predict(input_df)[0]
    prob_rf = model_rf.predict_proba(input_df)[0][1]

    input_scaled = scaler.transform(input_df)
    pred_lr = model_lr.predict(input_scaled)[0]
    prob_lr = model_lr.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest ⭐")
        if pred_rf == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")
        st.metric("Churn Probability", f"{prob_rf:.2%}")

    with col2:
        st.markdown("### 📈 Logistic Regression")
        if pred_lr == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")
        st.metric("Churn Probability", f"{prob_lr:.2%}")

    # القرار النهائي
    st.markdown("---")
    st.subheader("🎯 Final Decision (Random Forest)")
    if pred_rf == 1:
        st.error(f"Customer is likely to **CHURN** — Probability: {prob_rf:.2%}")
    else:
        st.success(f"Customer is likely to **STAY** — Probability: {prob_rf:.2%}")

    if pred_rf != pred_lr:
        st.caption("ℹ️ Models disagree — final decision follows Random Forest (higher accuracy).")