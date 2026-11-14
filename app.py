

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Price Prediction",
    
)

@st.cache_resource
def load_model():
    return joblib.load("Car_price_prediction_model.pkl")

model = load_model()
preprocessor = model.named_steps["preprocessor"]

numeric_features = []
categorical_features = {}

for name, transformer, columns in preprocessor.transformers_:

    if name == "numerical":
        numeric_features = list(columns)

    elif name == "categorical":
        categorical_features = {
            column: list(categories)
            for column, categories in zip(
                columns, transformer.categories_
            )
        }

st.title("Car Price Prediction")
st.write("Enter your car details to estimate its price.")

with st.form("car_prediction_form"):
    inputs = {}

    st.subheader("Numerical Features")

    for column in numeric_features:
        inputs[column] = st.number_input(
            str(column),
            value=0.0
        )

    st.subheader("Categorical Features")

    for column, options in categorical_features.items():
        inputs[column] = st.selectbox(
            str(column),
            options=options
        )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_df = pd.DataFrame([inputs])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Car Price: {prediction:,.2f}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
