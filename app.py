import streamlit as st
import pandas as pd
import joblib
import os
import gdown
import requests

# -----------------------------
# Step 1: Download model from Google Drive
# -----------------------------
MODEL_URL = "https://vivek-practics-ds-bucket.s3.ap-southeast-2.amazonaws.com/random_forest_model.pkl"
MODEL_PATH = "random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Please check the Google Drive link.")
    st.stop()

# -----------------------------
# Step 2: Streamlit UI
# -----------------------------
st.title('HDB Resale Price Predictor')
st.write('Enter the details of the flat to get a predicted resale price.')

# Replace df-based unique values with hard-coded lists or load from a file
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH']
flat_type_options = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
flat_model_options = ['Model A', 'Model B', 'Model C']

# Create input fields
town = st.selectbox('Town', town_options)
flat_type = st.selectbox('Flat Type', flat_type_options)
block = st.text_input('Block')
street_name = st.text_input('Street Name')
storey_mid = st.slider('Storey Level', 1, 50, 10)
floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=1.0)
flat_model = st.selectbox('Flat Model', flat_model_options)
lease_commence_date = st.number_input('Lease Commencement Year', min_value=1960, max_value=2025)
year = st.number_input('Resale Year', min_value=1990, max_value=2025)

if st.button('Predict Resale Price'):
    input_data = pd.DataFrame([[town, flat_type, block, street_name, floor_area_sqm, flat_model, lease_commence_date, year, storey_mid]],
                              columns=['town', 'flat_type', 'block', 'street_name', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'year', 'storey_mid'])
    
    # Make prediction
    predicted_price = model.predict(input_data)
    st.success(f'Predicted Resale Price: ${predicted_price[0]:,.2f}')
