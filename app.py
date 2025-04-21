import streamlit as st
import joblib
import torch
import torch.nn as nn
import numpy as np

# --- Define the RNNModel class (same as during training) ---
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure input is 3D: [batch, sequence, feature]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add feature dimension if needed
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

# --- Load the trained model ---
model = joblib.load('rnn_sentiment_model.pkl')
model.eval()

# --- Preprocess the user input text ---
def preprocess_input(text):
    # Convert characters to ASCII values (demo logic – replace with real tokenizer if needed)
    max_len = 100
    char_tensor = [ord(c) for c in text[:max_len]]
    padded = char_tensor + [0] * (max_len - len(char_tensor))  # pad to fixed length
    tensor_input = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)  # [1, seq_len]
    return tensor_input

# --- Predict sentiment ---
def predict_sentiment(text):
    inputs = preprocess_input(text)
    with torch.no_grad():
        output = model(inputs)
        predicted = torch.argmax(output, dim=1).item()
    return predicted

# --- Sentiment mapping ---
sentiment_labels = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# --- Streamlit Interface ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.write("Enter a review below and let the RNN model predict its sentiment.")

# --- Text input ---
user_review = st.text_area("📝 Your Review")

# --- Predict button ---
if st.button("🔍 Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        prediction = predict_sentiment(user_review)
        sentiment = sentiment_labels.get(prediction, "Unknown")
        st.success(f"### ✅ Sentiment: {sentiment}")
