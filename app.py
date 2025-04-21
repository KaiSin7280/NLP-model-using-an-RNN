import streamlit as st
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model_data = joblib.load('sentiment_model.pkl')
model_path = model_data['model_path']
tokenizer = model_data['tokenizer']
model = load_model(model_path)

# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded)[0]

        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_class = prediction.argmax()
        confidence = prediction[predicted_class] * 100

        st.write(f"**Prediction:** {label_map[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")
