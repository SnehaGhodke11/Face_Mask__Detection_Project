import os
import gdown
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Google Drive file ID (mask_final.zip मध्ये तुझं model आहे)
FILE_ID = "1Co4mQYE3Z1bEmoXCKi9R29tzS5RdbNkL"
ZIP_OUTPUT = "mask_final.zip"
MODEL_DIR = "model_dir"

# Download zip if not exists
if not os.path.exists(ZIP_OUTPUT):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_OUTPUT, quiet=False)

# Extract zip if not already extracted
with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zip_ref:
    zip_ref.extractall(MODEL_DIR)

# Auto-detect keras file inside extracted folder
MODEL_PATH = None
for root, dirs, files in os.walk(MODEL_DIR):
    for f in files:
        if f.endswith(".keras"):
            MODEL_PATH = os.path.join(root, f)
            break

if MODEL_PATH is None:
    raise FileNotFoundError("mask_final.keras not found in extracted zip")

# Load model
model = load_model(MODEL_PATH)

st.title("😷 Face Mask Detection")

# Initialize session_state variable
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

uploaded_file = st.file_uploader("📂 Upload an Image:", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file)
    img = Image.open(uploaded_file)
    img = img.resize((128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = prediction[0][0]

    if prob > 0.5:
        st.error("Prediction: Without Mask ❌😷")
    else:
        st.success("Prediction: WITH Mask ✅😷")

col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state.open_camera = True
    
with col2:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera = False

if st.session_state.open_camera:
    camera_image = st.camera_input("Click Photo / Selfie:")
    if camera_image is not None:
        img = Image.open(camera_image)
        img = img.resize((128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = prediction[0][0]

        if confidence > 0.5:
            st.error(f"Without Mask ❌😷 ({confidence:.2%})")
        else:
            st.success(f"With Mask ✅😷 ({(1-confidence):.2%})")

        # Close Camera after capture (optional)
        st.session_state.open_camera = False
