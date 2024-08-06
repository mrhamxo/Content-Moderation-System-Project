# app.py

import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Load the pre-trained model
model = keras.models.load_model("model/save_model.keras")

def prediction_function(image, model):
    classes = ['Safe', 'Violent']
    resize = cv2.resize(image, (224, 224))
    scale = resize / 255.0
    image = scale.reshape(1, 224, 224, 3)
    prediction = model.predict(image)
    threshold = 0.5
    pred = 1 if prediction > threshold else 0
    return classes[pred], image

st.title("Image Moderation System")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to a format suitable for prediction
    image = np.array(image.convert('RGB'))

    # Make prediction
    label, processed_image = prediction_function(image, model)
    
    # Display the result
    st.write(f"Prediction: {label}")
    st.image(processed_image, caption=f"Predicted: {label}", use_column_width=True)
