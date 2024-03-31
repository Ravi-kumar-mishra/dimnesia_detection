# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:14:20 2024

@author: ravik
"""

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = '/content/drive/MyDrive/Datasets/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

# Load the model
model = load_model()

# Title
st.title('Alzheimer\'s Disease Prediction from MRI')

# Upload Image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    st.write(f"Prediction: {class_names[prediction]}")
