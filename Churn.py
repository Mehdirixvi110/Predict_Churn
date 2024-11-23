import streamlit as st
import pandas as pd
import joblib
import dill
from sklearn.pipeline import Pipeline

# Load the pretrained model
with open("C:/Users/Mehdi/Downloads/pipeline.pkl", "rb") as file:
    model = dill.load(file)

# Load the feature dictionary
with open("C:/Users/Mehdi/Downloads/my_feature_dict.pkl", "rb") as file:
    my_feature_dict = joblib.load(file)

# Creating prediction function
def predict_churn(data):
    # Debugging: Check input data
    st.write("Input Data for Prediction:")
    st.write(data.head())

    # Make prediction
    prediction = model.predict(data)
    st.write(f"Prediction: {prediction}")

    return prediction[0]  # Return the first element directly

# Streamlit UI
st.title('CUSTOMER CHURN PREDICTION')
st.subheader('Based on Customer churn Dataset')

# Creating Categorical inputs
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}
for i, col in enumerate(categorical_input.get('Column Name').values()):
    options = categorical_input.get('Members')[i]
    categorical_input_vals[col] = st.selectbox(col, options)

# Creating Numerical inputs
st.subheader('Numerical Features')
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}
for col in numerical_input.get('Column Name'):
    if col == 'AGE':
        min_value = 1
        max_value = 100
    elif col == 'EXPERIENCEINCURRENTDOMAIN':
        min_value = 0
        max_value = 20
    else:
        min_value = 2000
        max_value = 2030
    numerical_input_vals[col] = st.slider(col, min_value, max_value, min_value)

# Combine categorical and numerical input dictionaries
input_data = {**categorical_input_vals, **numerical_input_vals}

# Convert input_data to DataFrame
input_data = pd.DataFrame([input_data])

# Churn Prediction
if st.button('Predict'):
    prediction = predict_churn(input_data)  # Now it directly returns a scalar prediction
    translation_dict={1: 'teir 1', 2: 'teir 2', 3: 'teir 3'}
    prediction_translate = translation_dict.get(prediction)
    st.write(f'There is a possibility that the employee will **{prediction}**, according to my prediction.')
