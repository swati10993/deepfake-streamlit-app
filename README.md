# Deepfake Detection Using Deep Learning

## Project Description
Deepfake Face Detection using Xception (Keras)
This project focuses on detecting AI-generated (deepfake) face images using a transfer learning approach with the Xception model. The system classifies images into Real or Fake using a trained deep learning model. A user-friendly Streamlit interface is included for real-time predictions.


## Project Demo (Streamlit App)

🔗 Live App: Your Streamlit URL here:https://deepfake-app-app-brnezubmad7aanduqucemb.streamlit.app/
🗂 GitHub Repository: https://github.com/swatisah/deepfake-detection


## Features

✔ Deepfake image classification (Real vs Fake)

✔ Xception-based transfer learning

✔ Preprocessing pipeline for face images

✔ High accuracy (≈95%)

✔ Streamlit Web App for live testing

✔ Model loading + prediction support


## PROJECT STRUCTURE

📦 deepfake-detection
 ┣ 📂 dataset/
 ┣ 📂 model/
 ┃ ┗ xception_model.h5
 ┣ 📂 streamlit_app/
 ┃ ┗ app.py
 ┣ 📂 src/
 ┃ ┣ train.py
 ┃ ┣ preprocess.py
 ┃ ┗ predict.py
 ┣ requirements.txt
 ┗ README.md


## MODEL ARCHITECTURE

Base Model: Xception (ImageNet weights)
Layers Added:GlobalAveragePoolin
Dropout (0.5)
Dense (1) + Sigmoid
Loss: Binary Crossentropy
Optimizer: Adam (lr = 0.0001)
Metrics: Accuracy

## Dataset

This project uses the **Deepfake and Real Images Dataset (Version 1 – 1.8 GB)** from Kaggle.  
The dataset is organized into three main splits:

 Description
- **Train** folder is used to train the Xception model  
- **Validation** folder is used to tune model performance during training  
- **Test** folder is used to evaluate accuracy on unseen images  

Dataset Labels
- `Real` → authentic human faces  
- `Fake` → AI-generated / manipulated deepfake faces  
This clear folder structure makes it ideal for a binary classification deep learning model.
The original dataset was **1.8 GB**, containing thousands of real and fake face images.  
Due to storage and training limitations, a **subset of the dataset** was extracted for this project.

I selected a balanced set of images from each folder:

- Train → subset of Real + Fake  
- Validation → subset of Real + Fake  
- Test → subset of Real + Fake  

This reduced dataset allowed:
- Faster training  
- Easier model experimentation  
- Smooth deployment on Streamlit Cloud  

Despite using a smaller subset, the Xception model still achieved strong performance.



## Installation

Follow the steps below to set up and run the Deepfake Detection project:
1. Clone the Repository
   
git clone https://github.com/swati10993/deepfake-streamlit-app

cd deepfake-streamlit-app

2️. Install Dependencies
pip install -r requirements.txt

3️. Download the Xception Model

Because GitHub does not allow files larger than 100MB, download the trained model from Google Drive:

🔗 Download Model (.h5)
https://drive.google.com/file/d/1zIUsFH_gUyfjL_aDUFqEchRkj562OsI_/view?usp=sharing

Place the file inside the project folder:
deepfake-streamlit-app/
 ├── appnew.py
 ├── requirements.txt
 ├── xception_epoch_01_manual.h5   ← place here

 4. Run the App
streamlit run appnew.py


## How to Use

Open Streamlit app
Upload any face image
Model processes and classifies
Output: Real or Fake

## Results

Training Accuracy: ~95%

Validation Accuracy: ~94–96%

Model performs well on unseen deepfake images


## Future Improvements

Video deepfake detection using frame extraction.

Lightweight MobileNet model for mobile deployment.

More datasets like DFDC & Celeb-DF.

Liveness detection


## Technologies Used

>Python
>
>TensorFlow / Keras

>NumPy, OpenCV

>Streamlit
>GitHub
>Xception Architecture


## Author

Swati Kumari

Deep Learning Project — 2025

Patna Women’s College
