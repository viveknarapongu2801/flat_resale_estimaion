import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import requests

# ----------------------------
# S3 model configuration
# ----------------------------
MODEL_URL = "https://vivek-practics-ds-bucket.s3.ap-southeast-2.amazonaws.com/random_forest_model.pkl"
MODEL_PATH = "random_forest_model.pkl"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model from AWS S3...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    st.success("Model downloaded successfully!")

# Load the model
model = joblib.load(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title('HDB Resale Price Predictor')
st.write('Enter the details of the flat to get a predicted resale price.')

# Provide categorical options manually or from saved unique values
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH']  # replace with full list
flat_type_options = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
flat_model_options = ['Model A', 'Model B', 'Improved', 'Simplified']  # replace with your actual models

# Input fields
town = st.selectbox('Town', town_options)
flat_type = st.selectbox('Flat Type', flat_type_options)
block = st.text_input('Block')
street_name = st.text_input('Street Name')
storey_mid = st.slider('Storey Level', 1, 50, 10)
floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=1.0)
flat_model = st.selectbox('Flat Model', flat_model_options)
lease_commence_date = st.number_input('Lease Commencement Year', min_value=1960, max_value=2025)
year = st.number_input('Resale Year', min_value=1990, max_value=2025)

# Prediction
if st.button('Predict Resale Price'):
    input_data = pd.DataFrame([[town, flat_type, block, street_name, floor_area_sqm, flat_model, lease_commence_date, year, storey_mid]],
                              columns=['town', 'flat_type', 'block', 'street_name', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'year', 'storey_mid'])

    # Ensure order matches your trained model (replace X_train_columns with actual columns used in training)
    X_train_columns = ['town', 'flat_type', 'block', 'street_name', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'year', 'storey_mid']
    input_data = input_data[X_train_columns]

    predicted_price = model.predict(input_data)
    st.success(f'Predicted Resale Price: ${predicted_price[0]:,.2f}')
