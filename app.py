import streamlit as st
import torch
import torch.nn as nn

# --- Define the model class (must match the one used in training) ---
class RNNModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, output_size=3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

# --- Load the full model ---
model = torch.load("rnn_sentiment_model.pkl", map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# --- Preprocess input text into tensor format ---
def preprocess_input(text):
    max_len = 100
    input_size = 128

    # Convert each character to a normalized value
    vectors = []
    for c in text[:max_len]:
        val = ord(c) / 255.0  # Normalize ASCII values between 0 and 1
        vec = [val] * input_size  # Use the same value for all 128 features
        vectors.append(vec)

    # Pad the sequence to max_len if needed
    while len(vectors) < max_len:
        vectors.append([0.0] * input_size)

    tensor_input = torch.tensor([vectors], dtype=torch.float32)  # Shape: [1, max_len, 128]
    return tensor_input

# --- Predict sentiment ---
def predict_sentiment(text):
    inputs = preprocess_input(text)
    with torch.no_grad():  # No need to compute gradients during inference
        output = model(inputs)
        predicted = torch.argmax(output, dim=1).item()  # Get the predicted class
    return predicted

# --- Sentiment labels ---
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
