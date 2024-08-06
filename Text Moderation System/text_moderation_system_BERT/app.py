import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

# Load the model and tokenizer
model_save_path = 'model'
tokenizer_save_path = 'tokenizer'

model = TFBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    label = tf.argmax(predictions, axis=1).numpy()[0]
    return label, predictions.numpy()

# Streamlit app
st.title("Text Sentiment Analysis with BERT")

text = st.text_area("Enter the text for sentiment analysis:")

if st.button("Analyze"):
    if text:
        label, predictions = predict(text)
        sentiment = "Negative" if label == 1 else "Positive"
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {predictions[0][label]:.2f}")
    else:
        st.write("Please enter text to analyze.")
