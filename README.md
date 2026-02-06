🧠 Brain Tumor Detection with Explainable AI (Streamlit)

An interactive web-based application for brain tumor detection from MRI images using deep learning and Explainable AI (Grad-CAM). The system provides real-time predictions with confidence scores and visually highlights tumor-affected regions to improve interpretability and trust in AI-driven medical diagnosis.

🚀 Features

  - Upload brain MRI images through a simple Streamlit interface
  - Real-time Tumor / No Tumor classification with confidence score
  - Optimized inference using TensorFlow Lite (TFLite)
  - Grad-CAM visual explanations for tumor-positive predictions
  - Side-by-side display of original MRI and highlighted tumor regions
  - Built-in glossary explaining medical and AI terminology

🏗️ System Architecture

  - User uploads MRI image
  - Image preprocessing (resize, normalization)
  - TFLite model performs fast classification
  - Keras (.h5) CNN model generates Grad-CAM heatmap (if tumor detected)
  - Results and explanations displayed in the Streamlit UI

🧠 Model Details

  - Task: Binary Classification (Tumor / No Tumor)
  - Models Used:
      - brain_tumor_classifier.tflite – Fast inference
      - brain_tumor_classifier.h5 – Explainability (Grad-CAM)
  - Decision Threshold: 0.42
  - Explainability: Grad-CAM applied on last convolutional layer

🛠️ Tech Stack

  - Programming Language: Python
  - Web Framework: Streamlit
  - Deep Learning: TensorFlow, Keras
  - Model Optimization: TensorFlow Lite
  - Explainable AI: Grad-CAM
  - Image Processing: OpenCV, PIL
  - Numerical Computing: NumPy

📂 Project Structure
```tree
├── streamlit_app.py
├── brain_tumor_classifier.h5
├── brain_tumor_classifier.tflite
├── requirements.txt
└── README.md
```

⚙️ Installation & Setup
- 1️⃣ Clone the Repository
        git clone https://github.com/your-username/brain-tumor-detection-streamlit.git
        cd brain-tumor-detection-streamlit
- 2️⃣ Create Virtual Environment (Optional but Recommended)
        python -m venv venv
        source venv/bin/activate   # Windows: venv\Scripts\activate
- 3️⃣ Install Dependencies
        pip install -r requirements.txt

Sample requirements.txt:

streamlit
tensorflow
opencv-python
numpy
pillow

▶️ Run the Application

streamlit run streamlit_app.py

Then open the browser at:
http://localhost:8501

📊 Output

  - Prediction: Tumor / No Tumor
  - Confidence Score: Percentage confidence of the model
  - Grad-CAM Visualization: Highlighted tumor regions 

⚠️ Disclaimer

This project is for educational and research purposes only.
It is not intended for clinical or diagnostic use.
