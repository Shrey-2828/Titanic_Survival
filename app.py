import streamlit as st
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open('pipe.pkl', 'rb'))

def predict(data):
    ans = model.predict(data)
    return ans[0]  


def main():
    st.title('Titanic Survival Chance !!!')

    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Gender = st.selectbox("Gender", ["male", "female"])
    Age = st.number_input("Age", min_value=0, max_value=110)
    fare = st.number_input("Fare", min_value=0.0)
    Embarked = st.selectbox("Embarked", ["S", "Q", "C"])

    data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Gender],
        "Age": [Age],
        "Fare": [fare],
        "Embarked": [Embarked]
    })

    if st.button('Predict'):
        result = predict(data)

        if result == 1:
            st.success("Survived ✅")
        else:
            st.error("Not Survived ❌")


if __name__ == '__main__':
    main()
