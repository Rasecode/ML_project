#1. Loading libraries

import json
import math
import requests
import streamlit as st

st.title("Heart Disease web app")

#2. CREATING BOXES

data = {}

data["Age"] = st.number_input(
    "Age",
    min_value=0,
    step=1,
    value=63,
    help="Customer's Age",
)

data["Sex"] = st.selectbox(
     'Sex',
     ('F', 'M'))

data["ChestPainType"] = st.selectbox(
     'ChestPainType',
     ('ATA', 'NAP','ASY', 'TA'))

data["RestingBP"] = st.number_input(
    "RestingBP",
    min_value=0,
    step=1,
    value=140,
    help="Customer's RestingBP",
)
data["Cholesterol"] = st.number_input(
    "Cholesterol",
    min_value=0,
    step=1,
    value=195,
    help="Customer's RestingBP",
)
data["FastingBS"] = st.number_input(
    "FastingBS",
    min_value=0,
    max_value=1,
    step=1,
    value=0,
    help="Customer's FastingBS",
)


data["RestingECG"] = st.selectbox(
     'RestingECG',
     ('Normal', 'ST','LVH'))

data["MaxHR"] = st.number_input(
    "MaxHR",
    min_value=0,
    step=1,
    value=179,
    help="Customer's MaxHR",
)

data["ExerciseAngina"] = st.selectbox(
     'ExerciseAngina',
     ('N', 'Y'))

data["Oldpeak"] = st.number_input(
    "Oldpeak",
    min_value=-100,
    step=1,
    value=0,
    help="Customer's Oldpeak",
)
data["ST_Slope"] = st.selectbox(
     'ST_Slope',
     ('Up', 'Down', 'Flat'))

if st.button("Get the probability"):
    data_json = json.dumps(data)

    prediction = requests.post(
        "http://localhost:3000/predict",
        headers={"content-type": "application/json"},
        data=data_json,
    ).text
    st.write(f"This customer has a heart disease probability of: {prediction}%")