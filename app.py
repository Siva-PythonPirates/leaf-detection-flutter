import os
import requests
import pandas as pd
import streamlit as st
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import google.generativeai as genai
from google.generativeai import generative_models

# Gemini AI API
api_key = "AIzaSyBFvdQlmO4yqipuNs-l7wd37xWgZ34dVV8"


# Image classification
processor = AutoImageProcessor.from_pretrained("hareeshr/medicinal_plants_image_detection")
model = AutoModelForImageClassification.from_pretrained("hareeshr/medicinal_plants_image_detection")
pipeline = pipeline(task="image-classification", model=model, image_processor=processor)

def predict(image):
    predictions = pipeline(image)
    return predictions[0]

def main():
    st.title("Medicinal Plant Classification and Information")

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        clicked = st.form_submit_button("Predict")

        if clicked:
            result = predict(image)
            label = result['label']
            score = result['score'] * 100
            st.success(f"The predicted image is {label} with {score:.2f}% confidence.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-pro")

            prompt = "Provide detailed information about the medicinal plant 'Tulasi'. Please include its botanical name, common names, medicinal properties, traditional uses, active compounds, potential health benefits, and any known contraindications or side effects. Additionally, discuss who can benefit from its usage, such as individuals with specific health conditions or symptoms, and who should avoid it, such as pregnant or breastfeeding women, individuals with certain medical conditions, or those taking specific medications. Please provide evidence-based information and cite credible sources where applicable."
            # content = [prompt]
            response = model.generate_content(prompt)

            print(response.text)

            st.write(response.text)
if __name__ == "__main__":
    main()