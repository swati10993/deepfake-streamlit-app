import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import gdown
import os
from PIL import Image

# ---------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ---------------------------
file_id = "1zIUsFH_gUyfjL_aDUFqEchRkj562OsI_"
output = "deepfake_model.h5"

if not os.path.exists(output):
    with st.spinner("Downloading model... Please wait."):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# Load model
model = load_model(output)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🔍 Deepfake Detection App")
st.write("Upload an image to check if it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.error("🚨 Fake Face Detected!")
        st.write(f"Confidence: {pred:.2f}")
    else:
        st.success("✅ Real Face Detected")
        st.write(f"Confidence: {1 - pred:.2f}")
    

