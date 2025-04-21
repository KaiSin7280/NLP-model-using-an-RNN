import streamlit as st
import joblib
import torch
import torch.nn as nn

# --- Define the RNNModel class ---
class RNNModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, output_size=3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

# --- Load the trained model ---
model = joblib.load('rnn_sentiment_model.pkl')
model.eval()

# --- Preprocess the user input text ---
def preprocess_input(text):
    max_len = 100  # sequence length
    input_size = 128  # feature size per character

    vectors = []
    for c in text[:max_len]:
        val = ord(c) / 255.0  # normalize ASCII value
        vec = [val] * input_size  # expand to vector of 128 dims
        vectors.append(vec)

    # Pad to max_len
    while len(vectors) < max_len:
        vectors.append([0.0] * input_size)

    tensor_input = torch.tensor([vectors], dtype=torch.float32)  # shape: [1, max_len, 128]
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

# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.write("Enter a review below and let the RNN model predict its sentiment.")

user_review = st.text_area("📝 Your Review")

if st.button("🔍 Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        prediction = predict_sentiment(user_review)
        sentiment = sentiment_labels.get(prediction, "Unknown")
        st.success(f"### ✅ Sentiment: {sentiment}")
