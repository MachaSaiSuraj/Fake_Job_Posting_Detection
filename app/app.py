import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'fake_job_model.pkl'))
vectorizer_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl'))

st.write("Model path:", model_path)
st.write("Vectorizer path:", vectorizer_path)

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    model = joblib.load(model_path)

if not os.path.exists(vectorizer_path):
    st.error(f"Vectorizer file not found at {vectorizer_path}")
else:
    vectorizer = joblib.load(vectorizer_path)