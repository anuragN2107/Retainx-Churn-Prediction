import gradio as gr
import pandas as pd
import numpy as np
import joblib

# 1. Load the exported production artifacts
artifacts = joblib.load('churn_pipeline_artifacts.joblib')
scaler = artifacts['scaler']
categorical_cols = artifacts['categorical_cols']
numerical_cols = artifacts['numerical_cols']
xgb_model = artifacts['xgboost_model']
lgb_model = artifacts['lightgbm_model']
features_list = artifacts['features_list']

def predict_churn(
    gender, senior_citizen, partner, dependents, tenure, phone_service,
    multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method, monthly_charges, total_charges
):
    # 2. Structure user inputs into a single row DataFrame
    input_data = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': int(senior_citizen == "Yes"), 'Partner': partner, 
        'Dependents': dependents, 'tenure': float(tenure), 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service, 
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 
        'DeviceProtection': device_protection, 'TechSupport': tech_support, 
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 
        'Contract': contract, 'PaperlessBilling': paperless_billing, 
        'PaymentMethod': payment_method, 'MonthlyCharges': float(monthly_charges), 
        'TotalCharges': float(total_charges)
    }])
    
    # Ensure correct column order matching original training features
    input_data = input_data[features_list]
    
    # 3. Apply operational transformations using the saved scikit-learn states
    for col in categorical_cols:
        # Recreate internal encoding labels matching label encoder assumptions mapping
        mapping = {val: idx for idx, val in enumerate(np.sort(input_data[col].unique()))}
        input_data[col] = input_data[col].map(mapping).fillna(0).astype(int)
        
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # 4. Generate explicit pipeline predictions
    xgb_prob = float(xgb_model.predict_proba(input_data)[0][1])
    lgb_prob = float(lgb_model.predict_proba(input_data)[0][1])
    
    # Format descriptive metrics for business user displays
    xgb_status = "⚠️ HIGH RISK" if xgb_prob > 0.50 else "✅ STABLE"
    lgb_status = "⚠️ HIGH RISK" if lgb_prob > 0.50 else "✅ STABLE"
    
    return (
        f"{xgb_status} ({xgb_prob*100:.1f}% Churn Probability)",
        f"{lgb_status} ({lgb_prob*100:.1f}% Churn Probability)"
    )

# 5. Construct the UI Layout using Gradio Components
interface = gr.Interface(
    fn=predict_churn,
    title="RetainX: Real-Time Customer Attrition Engine",
    description="Enterprise predictive dashboard using dual Gradient Boosting (XGBoost vs. LightGBM) to forecast subscriber churn risks.",
    inputs=[
        gr.Dropdown(["Female", "Male"], label="Gender"),
        gr.Radio(["No", "Yes"], label="Senior Citizen Status"),
        gr.Radio(["No", "Yes"], label="Has Partner"),
        gr.Radio(["No", "Yes"], label="Has Dependents"),
        gr.Slider(minimum=0, maximum=72, value=12, step=1, label="Tenure (Months)"),
        gr.Radio(["No", "Yes"], label="Phone Service"),
        gr.Dropdown(["No phone service", "No", "Yes"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service Type"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Online Security Addon"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Online Backup Addon"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Device Protection Plan"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Tech Support Plan"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Streaming TV service"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Streaming Movies service"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type"),
        gr.Radio(["No", "Yes"], label="Paperless Billing"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(value=65.0, label="Monthly Charges ($)"),
        gr.Number(value=500.0, label="Total Charges ($)")
    ],
    outputs=[
        gr.Textbox(label="XGBoost Engine Forecast Output"),
        gr.Textbox(label="LightGBM Engine Forecast Output")
    ],
    theme="soft"
)

if __name__ == "__main__":
    interface.launch()