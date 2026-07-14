import streamlit as st
import pandas as pd
import pickle
import os

# --- Page Setup ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

# --- Load the Model ---
# This looks for the model in the same folder as this script
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'wine_quality_model.pkl')
    
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- User Interface ---
st.title("🍷 Wine Quality Predictor")
st.markdown("Enter the wine attributes to estimate the quality score.")

st.sidebar.header("Input Features")

# Inputs: type is mapped to 0 (Red) or 1 (White)
type_val = st.sidebar.selectbox("Wine Type", [0, 1], format_func=lambda x: "Red" if x == 0 else "White")
fixed_acidity = st.sidebar.slider("Fixed Acidity", 3.0, 16.0, 7.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 2.0, 0.3)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 2.0, 0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 70.0, 5.0)
chlorides = st.sidebar.slider("Chlorides", 0.0, 1.0, 0.05)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1.0, 300.0, 30.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 5.0, 500.0, 100.0)
density = st.sidebar.slider("Density", 0.9, 1.1, 0.99)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.2)
sulphates = st.sidebar.slider("Sulphates", 0.2, 2.0, 0.5)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)

# --- Prediction ---
if st.button("Predict Quality"):
    input_data = pd.DataFrame([[
        type_val, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]], columns=[
        'type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ])
    
    prediction = model.predict(input_data)
    
    st.subheader("Results")
    st.metric(label="Predicted Quality Score", value=f"{prediction[0]:.2f}")