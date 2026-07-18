import streamlit as st 
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


st.title("Face Mask Detection")

model=load_model("mask_final.keras")

if "open_camera" not in st.session_state:
    st.session_state.open_camera = False
    
uploaded_file=st.file_uploader("Upload an Image:",type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file)

    img=Image.open(uploaded_file)

    img=img.resize((128,128))  #resize img to 128,128as this we used while training 

    img_array=image.img_to_array(img) /255.0  #create array of image

    img_array=np.expand_dims(img_array,axis=0)

    prediction=model.predict(img_array)

    prob=prediction[0][0]

    if prob >0.5:
        st.error("Prediction: Without Mask ❌😷")
    else:
        st.success("Prediction: WITH Mask ✅😷")

col1,col2 =st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state.open_camera=True
    
with col2:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera=False

if st.session_state.open_camera:
    camere_image=st.camera_input("Click Photo / Selfie:")
    if camere_image is not None:
        
        img=Image.open(camere_image)
        img=img.resize((128,128))
        img_array=image.img_to_array(img)
        img_array=img_array/255.0
        img_array=np.expand_dims(img_array,axis=0)

        prediction=model.predict(img_array)
        confidence=prediction[0][0]

        if confidence>0.5:
            st.success(f"Without Mask 😷 ({confidence:.2%})")
        else:
            st.success(f"With Mask ✅ ({(1-confidence):.2%})")

        #Close Camera after capture(optional)
        st.session_state.open_camera=False
