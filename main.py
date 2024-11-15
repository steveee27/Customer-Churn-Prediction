import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Define paths to the model and encoders
MODEL_PATH = os.path.join('models', 'xgb_classifier_model.pkl')
GENDER_ENCODER_PATH = os.path.join('models', 'gender_encoder.pkl')
HAS_CR_CARD_ENCODER_PATH = os.path.join('models', 'hasCrCard_encoder.pkl')
IS_ACTIVE_MEMBER_ENCODER_PATH = os.path.join('models', 'isActiveMember_encoder.pkl')

# Load the model and encoders
model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)
hasCrCard_encoder = joblib.load(HAS_CR_CARD_ENCODER_PATH)
isActiveMember_encoder = joblib.load(IS_ACTIVE_MEMBER_ENCODER_PATH)

def main():
    st.markdown("<h1 style='text-align: center;'>Churn Model Deployment</h1>", unsafe_allow_html=True)

    # Input fields
    Surname = st.text_input("Surname: ")
    Age = st.number_input("Age: ", 0, 100)
    Gender = st.radio("Input Gender: ", ["Male", "Female"])
    Geography = st.radio("Geography: ", ['France', 'Spain', 'Germany'])
    Tenure = st.selectbox("Tenure: ", list(range(1, 11)))
    Balance = st.number_input("Balance: ", 0, 10000000)
    NumOfProducts = st.selectbox("Number Of Products:", [1, 2, 3, 4])
    HasCrCard = st.radio("I Have a Credit Card: ", ["Yes", "No"])
    IsActiveMember = st.radio("I am an Active Member: ", ["Yes", "No"])
    EstimatedSalary = st.number_input("Estimated Salary: ", 0, 10000000)
    CreditScore = st.number_input("Credit Score: ", 0, 1000)

    # Encoding Geography
    geography_encoding = {'France': [0, 0, 1], 'Spain': [0, 1, 0], 'Germany': [1, 0, 0]}
    geography_encoded = geography_encoding[Geography]

    # Data preparation
    data = {
        'Surname': Surname, 
        'Age': int(Age), 
        'Gender': Gender, 
        'CreditScore': int(CreditScore),
        'Tenure': int(Tenure), 
        'Balance': int(Balance),
        'NumOfProducts': NumOfProducts, 
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': int(EstimatedSalary),
        'Geography_France': geography_encoded[0],
        'Geography_Spain': geography_encoded[1],
        'Geography_Germany': geography_encoded[2]
    }

    df = pd.DataFrame([list(data.values())], columns=[
        'Surname', 'Age', 'Gender', 'CreditScore', 'Tenure', 
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
        'EstimatedSalary', 'Geography_France', 'Geography_Spain', 'Geography_Germany'
    ])

    # Feature scaling
    scaler = StandardScaler()
    df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(
        df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']]
    )

    # Replace with encoders
    df = df.replace(gender_encoder)
    df = df.replace(hasCrCard_encoder)
    df = df.replace(isActiveMember_encoder)

    # Prediction button
    if st.button('Make Prediction'):
        features = df.drop('Surname', axis=1)      
        result = makePrediction(features)
        prediction_text = "Churn" if result == 1 else "Not Churn"
        st.success(f"Mr./Mrs. {Surname} is {prediction_text}")

    st.markdown("<p style='text-align: center; font-size: small;'>Created by Steve Marcello Liem</p>", unsafe_allow_html=True)

# Prediction function
def makePrediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()