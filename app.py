import streamlit as st
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Load the trained RNN model
model = joblib.load('rnn_sentiment_model.pkl')

# Dummy tokenizer (replace with your actual tokenizer or preprocessing logic)
def preprocess_input(text):
    # This is a placeholder: replace with real preprocessing (e.g., tokenization, padding)
    # For example, if using a vocab dictionary: [vocab.get(word, 0) for word in text.split()]
    numeric_input = [ord(char) for char in text[:100]]  # simple char-to-int (for demo)
    return torch.tensor(numeric_input, dtype=torch.float32)

# Predict sentiment from input text
def predict_sentiment(text):
    model.eval()
    inputs = preprocess_input(text)

    # Ensure tensor shape: (1, sequence_length)
    inputs = inputs.unsqueeze(0)

    with torch.no_grad():
        output = model(inputs)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class

# Streamlit interface
st.title("🧠 Sentiment Analysis App")
st.write("Enter a review below and let the RNN model analyze the sentiment!")

# User text input
review = st.text_area("Your Review")

# Prediction
if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        prediction = predict_sentiment(review)
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        st.success(f"Predicted Sentiment: **{sentiment}**")
