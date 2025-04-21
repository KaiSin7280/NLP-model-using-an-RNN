import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer from sentiment_model.pkl
model_data = joblib.load("sentiment_model.pkl")
model_path = model_data['model_path']
model = load_model(model_path)  # works fine with .keras
tokenizer = model_data['tokenizer']
model = load_model(model_path)

# Constants
MAX_SEQUENCE_LENGTH = 100
labels = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Positive"}

# Streamlit UI
st.title("📊 Sentiment Analysis App")
st.write("Enter a review and let the RNN model tell you the sentiment!")

user_input = st.text_area("📝 Enter your review here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

        # Predict
        prediction = model.predict(padded)
        sentiment = np.argmax(prediction)

        # Display result
        st.subheader("Prediction:")
        st.success(f"{labels[sentiment]} (Confidence: {prediction[0][sentiment]:.2f})")
