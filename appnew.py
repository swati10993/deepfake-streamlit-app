import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE (Your File)
# -------------------------------------------------------
file_id = "1zIUsFH_gUyfjL_aDUFqEchRkj562OsI_"   # your file ID
model_path = "xception_epoch_01_manual.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model... Please wait..."):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)
IMAGE_SIZE = (299, 299)  # Xception input

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.set_page_config(page_title="Deepfake Face Detector")
st.title("🧠 Deepfake Face Detection")
st.write("Upload a face image to detect whether it's **Real** or a **Deepfake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize(IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)   # important!
    img_array = np.expand_dims(img_array, axis=0)

    # Predict (sigmoid)
    prediction = model.predict(img_array)[0][0]

    # Class mapping: fake = 0, real = 1
    if prediction < 0.5:
        label = "🔴 Deepfake"
        confidence = 1 - prediction
    else:
        label = "🟢 Real"
        confidence = prediction

    st.markdown("---")
    st.markdown(f"### 🧾 Prediction: **{label}**")
    st.markdown(f"📊 Confidence Score: **{confidence:.2f}**")

    

