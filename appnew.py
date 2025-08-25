import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image

# Load model
model = load_model("xception_epoch_01_manual.h5")
IMAGE_SIZE = (299, 299)  # Xception input size

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
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Adjusted for {'fake': 0, 'real': 1}
    label = "🔴 Deepfake" if prediction < 0.5 else "🟢 Real"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    st.markdown("---")
    st.markdown(f"### 🧾 Prediction: **{label}**")
    st.markdown(f"📊 Confidence Score: **{confidence:.2f}**")

