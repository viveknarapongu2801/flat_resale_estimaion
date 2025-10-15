import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model (assuming it's saved as 'random_forest_model.pkl')
# You'll need to save your trained model after the training step
# Example: joblib.dump(model, 'random_forest_model.pkl')
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()

# Assuming 'encoder' was used for categorical features and is available
# You'll need to save and load the encoder as well if you plan to use it for new data
# Example: joblib.dump(encoder, 'label_encoder.pkl')
# try:
#     encoder = joblib.load('label_encoder.pkl')
# except FileNotFoundError:
#     st.error("Label encoder file not found.")
#     st.stop()


st.title('HDB Resale Price Predictor')

st.write('Enter the details of the flat to get a predicted resale price.')

# Get unique values for categorical features from the original dataframe
# This assumes your original dataframe 'df' is available or you have saved the unique values
try:
    town_options = df['town'].unique().tolist()
    flat_type_options = df['flat_type'].unique().tolist()
    flat_model_options = df['flat_model'].unique().tolist()
except NameError:
    st.error("Original dataframe 'df' not found. Please ensure it's loaded or provide unique values for categorical features.")
    st.stop()


# Create input fields for the user
town = st.selectbox('Town', town_options)
flat_type = st.selectbox('Flat Type', flat_type_options)
block = st.text_input('Block')
street_name = st.text_input('Street Name')
storey_mid = st.slider('Storey Level', 1, 50, 10) # Assuming a reasonable range for storey levels
floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=1.0)
flat_model = st.selectbox('Flat Model', flat_model_options)
lease_commence_date = st.number_input('Lease Commencement Year', min_value=1960, max_value=2025)
year = st.number_input('Resale Year', min_value=1990, max_value=2025)


if st.button('Predict Resale Price'):
    # Create a dataframe from user inputs
    input_data = pd.DataFrame([[town, flat_type, block, street_name, floor_area_sqm, flat_model, lease_commence_date, year, storey_mid]],
                               columns=['town', 'flat_type', 'block', 'street_name', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'year', 'storey_mid'])

    # Ensure the order of columns matches the training data
    input_data = input_data[X_train.columns]


    # Make prediction
    predicted_price = model.predict(input_data)

    st.success(f'Predicted Resale Price: ${predicted_price[0]:,.2f}')
