import streamlit as st
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Load your trained model (make sure the model was saved correctly)
model = joblib.load('rnn_sentiment_model.pkl')

# Assuming your model takes a tensor of input text and predicts sentiment
# Modify this function based on how you preprocess and input data
def preprocess_input(review):
    # Example of preprocessing (e.g., tokenization, padding)
    # This should match the preprocessing done during training
    # For instance, if using a tokenizer:
    # tokens = tokenizer.encode(review, truncation=True, padding=True)
    # return tokens
    
    # As an example, let's pretend we're converting the input string to a tensor:
    input_tensor = np.array([ord(c) for c in review])  # Simple conversion (modify as needed)
    return torch.tensor(input_tensor)

def predict_sentiment(review):
    # Preprocess the input review
    processed_input = preprocess_input(review)
    
    # Convert the input to the model's expected format (e.g., tensor)
    input_tensor = processed_input.unsqueeze(0)  # Add batch dimension
    
    # Model prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)  # Assuming classification output
    
    return predicted_class.item()  # Return the predicted class (positive/negative/neutral)

# Create the Streamlit interface
st.title('Sentiment Analysis of Reviews')

# User input for the review
user_review = st.text_area('Enter a review for sentiment analysis:', '')

# Button to trigger the prediction
if st.button('Analyze Sentiment'):
    if user_review:
        # Get the sentiment prediction
        sentiment = predict_sentiment(user_review)
        
        # Display the result
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        st.write(f'Sentiment: {sentiment_map.get(sentiment, "Unknown")}')
    else:
        st.write('Please enter a review to analyze.')
