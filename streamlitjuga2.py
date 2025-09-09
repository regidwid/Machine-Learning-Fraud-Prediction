import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import sklearn
import category_encoders



# Set page config
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('insurance_fraud.sav', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'insurance_fraud.sav' not found. Please ensure the file is in the same directory.")
        return None

# Title and description
st.title("🛡️ Insurance Fraud Detection System")
st.markdown("This application predicts the likelihood of insurance fraud based on claim and policy information.")

# Load model
model = load_model()

if model is not None:
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Date & Time Information")
        
        month = st.selectbox("Month", 
                           ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug', 'Jul', 'May', 'Sep'])
        
        day_of_week = st.selectbox("Day of Week", 
                                 ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'])
        
        week_of_month = st.selectbox("Week of Month", [1, 2, 3, 4, 5])
        
        day_of_week_claimed = st.selectbox("Day of Week Claimed", 
                                         ['Tuesday', 'Monday', 'Thursday', 'Friday', 'Wednesday', 'Saturday', 'Sunday', '0'])
        
        month_claimed = st.selectbox("Month Claimed", 
                                   ['Jan', 'Nov', 'Jul', 'Feb', 'Mar', 'Dec', 'Apr', 'Aug', 'May', 'Jun', 'Sep', 'Oct', '0'])
        
        week_of_month_claimed = st.selectbox("Week of Month Claimed", [1, 2, 3, 4, 5])
        
        year = st.selectbox("Year", [1994, 1995, 1996])

    with col2:
        st.subheader("👤 Personal Information")
        
        age = st.slider("Age", min_value=16, max_value=80, value=35)
        
        sex = st.selectbox("Sex", ['Female', 'Male'])
        
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Widow', 'Divorced'])
        
        age_of_policy_holder = st.selectbox("Age Group of Policy Holder", 
                                          ['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25', 
                                           '36 to 40', '16 to 17', 'over 65', '18 to 20'])

    with col3:
        st.subheader("🚗 Vehicle Information")
        
        make = st.selectbox("Make", 
                          ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 
                           'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 
                           'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
        
        vehicle_category = st.selectbox("Vehicle Category", ['Sport', 'Utility', 'Sedan'])
        
        vehicle_price = st.selectbox("Vehicle Price Range", 
                                   ['more than 69000', '20000 to 29000', '30000 to 39000', 
                                    'less than 20000', '40000 to 59000', '60000 to 69000'])
        
        age_of_vehicle = st.selectbox("Age of Vehicle", 
                                    ['3 years', '6 years', '7 years', 'more than 7', '5 years', 
                                     'new', '4 years', '2 years'])
        
        number_of_cars = st.selectbox("Number of Cars", ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'])

    # Second row of columns
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("📋 Policy Information")
        
        policy_number = st.number_input("Policy Number", min_value=1, max_value=1000, value=1)
        
        policy_type = st.selectbox("Policy Type", 
                                 ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 
                                  'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision', 
                                  'Utility - Collision', 'Utility - Liability', 'Sport - All Perils'])
        
        base_policy = st.selectbox("Base Policy", ['Liability', 'Collision', 'All Perils'])
        
        deductible = st.selectbox("Deductible", [300, 400, 500, 700])
        
        days_policy_accident = st.selectbox("Days Policy to Accident", 
                                          ['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'])
        
        days_policy_claim = st.selectbox("Days Policy to Claim", 
                                       ['more than 30', '15 to 30', '8 to 15', 'none'])

    with col5:
        st.subheader("📊 Claim Information")
        
        accident_area = st.selectbox("Accident Area", ['Urban', 'Rural'])
        
        fault = st.selectbox("Fault", ['Policy Holder', 'Third Party'])
        
        police_report_filed = st.selectbox("Police Report Filed", ['No', 'Yes'])
        
        witness_present = st.selectbox("Witness Present", ['No', 'Yes'])
        
        past_number_of_claims = st.selectbox("Past Number of Claims", ['none', '1', '2 to 4', 'more than 4'])
        
        number_of_suppliments = st.selectbox("Number of Supplements", ['none', 'more than 5', '3 to 5', '1 to 2'])

    with col6:
        st.subheader("🏢 Agent & Address Information")
        
        agent_type = st.selectbox("Agent Type", ['External', 'Internal'])
        
        rep_number = st.selectbox("Rep Number", [12, 15, 7, 4, 3, 14, 1, 13, 11, 16, 6, 2, 8, 5, 9, 10])
        
        driver_rating = st.selectbox("Driver Rating", [1, 2, 3, 4])
        
        address_change_claim = st.selectbox("Address Change Claim", 
                                          ['1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months'])

    # Prediction button
    st.markdown("---")
    
    if st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Month': [month],
            'WeekOfMonth': [week_of_month],
            'DayOfWeek': [day_of_week],
            'Make': [make],
            'AccidentArea': [accident_area],
            'DayOfWeekClaimed': [day_of_week_claimed],
            'MonthClaimed': [month_claimed],
            'WeekOfMonthClaimed': [week_of_month_claimed],
            'Sex': [sex],
            'MaritalStatus': [marital_status],
            'Age': [age],
            'Fault': [fault],
            'PolicyType': [policy_type],
            'VehicleCategory': [vehicle_category],
            'VehiclePrice': [vehicle_price],
            'Days_Policy_Accident': [days_policy_accident],
            'Days_Policy_Claim': [days_policy_claim],
            'PastNumberOfClaims': [past_number_of_claims],
            'AgeOfVehicle': [age_of_vehicle],
            'AgeOfPolicyHolder': [age_of_policy_holder],
            'PoliceReportFiled': [police_report_filed],
            'WitnessPresent': [witness_present],
            'AgentType': [agent_type],
            'NumberOfSuppliments': [number_of_suppliments],
            'AddressChange_Claim': [address_change_claim],
            'NumberOfCars': [number_of_cars],
            'BasePolicy': [base_policy],
            'PolicyNumber': [policy_number],
            'RepNumber': [rep_number],
            'Deductible': [deductible],
            'DriverRating': [driver_rating],
            'Year': [year]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.error("⚠️ **HIGH FRAUD RISK DETECTED**")
                    st.markdown(f"**Fraud Probability:** {prediction_proba[1]:.2%}")
                else:
                    st.success("✅ **LOW FRAUD RISK**")
                    st.markdown(f"**Legitimate Claim Probability:** {prediction_proba[0]:.2%}")
            
            with col_result2:
                st.markdown("**Probability Breakdown:**")
                st.markdown(f"- Legitimate: {prediction_proba[0]:.2%}")
                st.markdown(f"- Fraudulent: {prediction_proba[1]:.2%}")
            
            # Risk level indicator
            risk_level = prediction_proba[1]
            if risk_level < 0.3:
                st.success("Risk Level: LOW")
            elif risk_level < 0.7:
                st.warning("Risk Level: MEDIUM")
            else:
                st.error("Risk Level: HIGH")
                
            # Progress bar for fraud probability
            # st.markdown("**Fraud Risk Meter:**")

            # st.progress(risk_level)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all input values are valid and the model file is compatible.")

    # Additional information
    st.markdown("---")
    st.markdown("### ℹ️ About This Model")
    st.info("""
    This fraud detection model analyzes various factors including:
    - **Temporal patterns**: When accidents and claims occur
    - **Personal demographics**: Age, marital status, etc.
    - **Vehicle information**: Make, age, category, and value
    - **Policy details**: Coverage type, deductibles, claim history
    - **Claim circumstances**: Fault determination, witnesses, police reports
    - **Agent and address information**: Internal vs external agents, address changes
    
    The model uses XGBoost algorithm with preprocessing pipeline for optimal performance.
    """)

else:
    st.error("Unable to load the model. Please check if 'insurance_fraud.sav' file exists in the application directory.")