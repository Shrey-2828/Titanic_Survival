import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# load model
model = pickle.load(open('pipe.pkl','rb'))

def predict(data):
    ans = model.predict(data)
    return ans[0]

st.title("🚢 Titanic Survival Predictor")

Pclass = st.selectbox("Passenger Class", [1,2,3])
Gender = st.selectbox("Gender", ["male","female"])
Age = st.number_input("Age", min_value=0, max_value=110)
fare = st.number_input("Fare", min_value=0.0)
Embarked = st.selectbox("Embarked", ["S","Q","C"])

data = pd.DataFrame({
    "Pclass":[Pclass],
    "Sex":[Gender],
    "Age":[Age],
    "Fare":[fare],
    "Embarked":[Embarked]
})

if st.button("Predict Survival"):

    result = predict(data)

    # probability
    prob = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if result == 1:
        st.success("Passenger Survived ✅")
    else:
        st.error("Passenger Did Not Survive ❌")
        
    st.write(f"Survival Probability: **{prob*100:.2f}%**")    

