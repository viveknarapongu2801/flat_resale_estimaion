import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import boto3

# ----------------------------
# S3 CONFIGURATION
# ----------------------------
BUCKET_NAME = "vivek-practics-ds-bucket"
MODEL_KEY = "random_forest_model.pkl"
MODEL_PATH = "random_forest_model.pkl"

# Initialize S3 client (works for public buckets)
s3 = boto3.client("s3")

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model from AWS S3...")
    try:
        s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_PATH)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except ValueError:
    st.error("Downloaded model is corrupted or incomplete. Please re-upload to S3 and try again.")
