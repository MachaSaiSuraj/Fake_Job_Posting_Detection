import streamlit as st
import joblib
import os

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'fake_job_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl'))

# App title
st.title("Fake Job Posting Detection")

# User input
user_input = st.text_area("Enter the job description:")

if st.button("Check if it's fake"):
    if user_input.strip() == "":
        st.warning("Please enter some job description text.")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("⚠️ This job posting is likely **FAKE**.")
        else:
            st.success("✅ This job posting appears to be **Genuine**.")
