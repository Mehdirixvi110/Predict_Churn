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
    prediction = model.predict(data)
    return prediction[0]  # Return the first element directly

# Set Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.write("Use the navigation bar to explore options.")

# Add a custom header
st.markdown(
    """
    <style>
    .main-header {
        font-size:40px;
        text-align:center;
        font-weight: bold;
        color:#f73c3c;
    }
    .sub-header {
        text-align: center;
        font-style: italic;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">CUSTOMER CHURN PREDICTION</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover customer retention insights with interactive tools.</p>', unsafe_allow_html=True)

# Add colorful sections for inputs
st.markdown("### Provide Input Features 📝")

# Creating Categorical inputs
st.markdown("#### Categorical Features 📊")
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}
for i, col in enumerate(categorical_input.get('Column Name').values()):
    options = categorical_input.get('Members')[i]
    categorical_input_vals[col] = st.selectbox(
        col,
        options,
        help=f"Choose the most relevant option for {col}"
    )

# Creating Numerical inputs
st.markdown("#### Numerical Features 🔢")
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}
for col in numerical_input.get('Column Name'):
    if col == 'AGE':
        min_value = 20
        max_value = 100
    elif col == 'EXPERIENCEINCURRENTDOMAIN':
        min_value = 0
        max_value = 20
    else:
        min_value = 2010
        max_value = 2030
    numerical_input_vals[col] = st.slider(
        f"Adjust {col}:",
        min_value,
        max_value,
        min_value,
        help=f"Slide to set the value for {col}"
    )

# Combine categorical and numerical input dictionaries
input_data = {**categorical_input_vals, **numerical_input_vals}

# Convert input_data to DataFrame
input_data = pd.DataFrame([input_data])

# Add interactivity with a collapsible section
with st.expander("See Input Data 📋", expanded=False):
    st.write("Preview the input data before making predictions:")
    st.write(input_data)

# Churn Prediction Button Styling
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #007da3;
        color: white;
        font-size: 20px;
        height: 2em;
        width: 20em;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Churn Prediction
if st.button('Predict Churn 🚀'):
    with st.spinner("Predicting..."):
        prediction = predict_churn(input_data)  # Direct scalar prediction
        translation_dict = {1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3'}
        prediction_translate = translation_dict.get(prediction, "Unknown")
        
        if prediction == 'LEAVE':  # Employee likely to leave
            st.error(f"🔴 ALERT: The employee is likely to **{prediction_translate}**!")
            st.markdown(
                """
                <style>
                .red-alert {
                    color: #FF0000;
                    font-weight: bold;
                }
                </style>
                <p class="red-alert">This is a critical situation. Please consider taking immediate action to retain the employee!</p>
                """,
                unsafe_allow_html=True
            )
        else:  # Employee likely to stay
            st.success(f"🟢 GREAT NEWS: The employee is predicted to **{prediction_translate}**!")
            st.balloons()

# Add a footer
st.markdown(
    """
    <hr style="border-top: 2px solid #bbb;">
    <div style="text-align: center;">
        <small>2024 Customer Insights. All rights reserved.</small>
    </div>
    """,
    unsafe_allow_html=True
)

# Add a background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpapers.com/images/hd/dark-gray-background-31zgslm940epcocw.jpg"); /* Replace with your image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white; /* Ensures text is visible */
    }
    </style>
    """,
    unsafe_allow_html=True
)