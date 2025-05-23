import streamlit as st
import os
import joblib
import numpy as np
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.email_processor import load_vocabulary, process_email, get_top_features

# Set paths
dataset_path = os.path.join(project_root, 'data', 'raw', 'emails.csv')
model_path = os.path.join(project_root, 'models', 'svm_spam_model.pkl')

# Load vocabulary and model
vocab = load_vocabulary(dataset_path)
model = joblib.load(model_path)

# Streamlit app
st.title("Spam Email Detector")
st.write("Enter an email text to check if it's spam or ham.")

# User input
email_text = st.text_area("Email Text", placeholder="Type your email here...", value="Hi John, let’s meet tomorrow at 10 AM to discuss the project. Please bring the report. Thanks!")

if st.button("Check"):
    if email_text:
        # Process email into feature vector
        feature_vector = process_email(email_text, vocab)
        
        # Predict
        prediction = model.predict(feature_vector)[0]
        label = "Spam" if prediction == 1 else "Ham"
        
        # Get confidence (decision function score)
        decision_score = model.decision_function(feature_vector)[0]
        confidence = 1 / (1 + np.exp(-decision_score))  # Convert to probability-like score
        
        # Get top contributing words
        top_words = get_top_features(email_text, vocab, top_n=5)
        
        # Display result
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.2%}")
        st.write("Top contributing words (word: frequency):")
        for word, freq in top_words:
            st.write(f"- {word}: {freq}")
    else:
        st.write("Please enter an email text to classify.")