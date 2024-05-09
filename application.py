import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/predict/"
SAMPLE_DIRECTORY = "images/"

def load_image(image_id):
    return Image.open(f"{SAMPLE_DIRECTORY}/{image_id}_leftImg8bit.png")

def load_mask(mask_id):
    return Image.open(f"{SAMPLE_DIRECTORY}/{mask_id}_gtFine_labelIds.png")

def get_prediction(image_path):
    files = {'file': ('filename.png', open(image_path, 'rb'), 'image/png')}
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        return np.array(response.json()["predicted_mask"])
    else:
        st.error("Failed to get prediction")
        return None

def display_images(image, real_mask, predicted_mask):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(real_mask, cmap='gray')
    axs[1].set_title("Real Mask")
    axs[1].axis("off")

    axs[2].imshow(predicted_mask, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")
    
    st.pyplot(fig)

def main():
    st.title("Image Segmentation Predictor")
    
    # Affichage des ID d'images disponibles
    image_ids = ["_".join(image.stem.split('_')[:3]) for image in Path('./images/').glob('*.png')]
    selected_id = st.selectbox("Choose an image ID", image_ids)

    if st.button("Predict Mask"):
        image_path = f"{SAMPLE_DIRECTORY}/{selected_id}_leftImg8bit.png"
        #mask_path = f"{SAMPLE_DIRECTORY}/{selected_id}_gtFine_labelIds.png"
        
        original_image = load_image(selected_id)
        real_mask = load_mask(selected_id)
        predicted_mask = get_prediction(image_path)
        
        if predicted_mask is not None:
            display_images(original_image, real_mask, predicted_mask)

if __name__ == "__main__":
    main()
