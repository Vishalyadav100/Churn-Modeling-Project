import streamlit as st
import pandas as pd
import joblib

def customer_input():  # input from function 
    st.subheader("Customer Details")

    credit_score = st.number_input("Credit Score",300,900,650)
    geography = st.selectbox("Geography",["France","Germany","Spain"])

    gender = st.selectbox("Gender",["Male","Female"])

    age = st.slider("Age",18,100,35)

    tenure = st.slider("Tenure",0,10,5)

    balance = st.number_input("Balance",value=50000.0)

    products = st.slider("Number of Products",1,4,1)

    has_card = st.selectbox("Has Credit Card",[0,1])

    active = st.selectbox("Is Active Member",[0,1])

    salary = st.number_input("Estimated Salary",value=50000.0)

    return pd.DataFrame({

        "CreditScore":[credit_score],
        "Geography":[geography],
        "Gender":[gender],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[products],
        "HasCrCard":[has_card],
        "IsActiveMember":[active],
        "EstimatedSalary":[salary]
    })

def encode_prediction(df): # encoding
    df["Gender"] = df["Gender"].map({
        "Male":1,
        "Female":0
    })

    df["Geography"] = df["Geography"].map({

        "France":0,

        "Germany":1,

        "Spain":2

    })

    return df

def load_model():  #Load Model Function
    model = joblib.load("saved_models/Best_Model.pkl")
    return model

def predict_customer(model, input_df): # Prediction function

    prediction = model.predict(input_df)

    probability = None
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_df)
    return prediction, probability

def show_prediction(prediction):
    if prediction[0] == 1:
        st.error("❌ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")


def show_probability(probability):
    if probability is not None:
        st.subheader("Prediction Probability")

        st.write(probability)
        st.progress(float(probability[0][1]))

