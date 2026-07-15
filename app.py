import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('spam_lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
except Exception as e:
    st.error("Please run train_model.py first to generate the model and tokenizer files!")
    st.stop()

# Streamlit App UI
st.title("🛡️ SMS Spam Detection App")
st.write("This application uses a Deep Learning LSTM model to check if an incoming message is Spam or Ham.")

# Text Area for Input
user_input = st.text_area("Enter the SMS text below:", placeholder="Type or paste your message here...", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        # 1. Preprocess the input exactly like we did during training
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
        
        # 2. Make prediction
        prediction = model.predict(padded)[0][0]
        
        # 3. Display Results
        st.write("---")
        if prediction > 0.5:
            st.error(f"🚨 **Spam Detected!** (Confidence Score: {prediction * 100:.2f}%)")
            st.write("Avoid clicking links or sharing personal info with this sender.")
        else:
            st.success(f"✅ **Ham (Safe)!** (Confidence Score: {(1 - prediction) * 100:.2f}% safe)")