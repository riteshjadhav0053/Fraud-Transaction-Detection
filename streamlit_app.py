import streamlit as st
import pandas as pd
from fraud_detection.pipeline.prediction_pipeline import PredictionPipeline, CustomData

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ MLOps Fraud Detection System")
st.write("Enter the transaction details below to evaluate whether it is legitimate or fraudulent.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input("Step (Hour simulation)", min_value=1, value=1)
        type_ = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Transaction Amount", min_value=0.0, value=181000.0, format="%.2f")
        oldbalanceOrg = st.number_input("Sender's Old Balance", min_value=0.0, value=181000.0, format="%.2f")
        
    with col2:
        newbalanceOrig = st.number_input("Sender's New Balance", min_value=0.0, value=0.0, format="%.2f")
        oldbalanceDest = st.number_input("Receiver's Old Balance", min_value=0.0, value=0.0, format="%.2f")
        newbalanceDest = st.number_input("Receiver's New Balance", min_value=0.0, value=181000.0, format="%.2f")

    submit_button = st.form_submit_button(label="Predict Fraud")

# Run Prediction
if submit_button:
    try:
        data = CustomData(
            step=int(step),
            type=str(type_),
            amount=float(amount),
            oldbalanceOrg=float(oldbalanceOrg),
            newbalanceOrig=float(newbalanceOrig),
            oldbalanceDest=float(oldbalanceDest),
            newbalanceDest=float(newbalanceDest)
        )
        
        final_df = data.get_data_as_dataframe()
        
        with st.spinner("Analyzing transaction..."):
            prediction_pipeline = PredictionPipeline()
            prediction, probability = prediction_pipeline.predict(final_df)
            
            is_fraud = prediction[0] == 1
            score = float(probability[0][1]) if probability is not None else 0.0
            
        st.divider()
        st.subheader("Prediction Results")
        
        if is_fraud:
            st.error("🚨 **Fraudulent Transaction Detected!**")
        else:
            st.success("✅ **Legitimate Transaction**")
            
        st.metric(label="Fraud Probability", value=f"{score * 100:.2f}%")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")