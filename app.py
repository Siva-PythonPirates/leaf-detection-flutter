import pandas as pd
import streamlit as st
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("hareeshr/medicinal_plants_image_detection")
model = AutoModelForImageClassification.from_pretrained("hareeshr/medicinal_plants_image_detection")
pipeline = pipeline(task="image-classification", model=model, image_processor=processor)

def predict(image):
    predictions = pipeline(image)
    return predictions[0]

def main():
    st.title("Image Classification")
    
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

if __name__ == "__main__":
    main()
