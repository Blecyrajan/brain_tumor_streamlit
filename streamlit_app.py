import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import sqlite3
import hashlib
from database import create_tables, get_connection

create_tables()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    return cur.fetchone()

st.sidebar.title("User Authentication")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    option = st.sidebar.selectbox("Choose", ["Login", "Register"])

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if option == "Register":
        if st.sidebar.button("Register"):
            if register_user(username, password):
                st.success("Registered successfully!")
            else:
                st.error("Username already exists")

    if option == "Login":
        if st.sidebar.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

st.sidebar.success(f"👤 Logged in as {st.session_state.username}")

# Logout button
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.user_id = None
    st.rerun()

# Load models
@st.cache_resource
def load_models():
    tflite_interpreter = tf.lite.Interpreter(model_path="brain_tumor_classifier (1).tflite")
    tflite_interpreter.allocate_tensors()
    h5_model = tf.keras.models.load_model("brain_tumor_classifier.h5")
    return tflite_interpreter, h5_model

interpreter, h5_model = load_models()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# Grad-CAM
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def generate_gradcam(img_array, original_image):
    layer_name = get_last_conv_layer_name(h5_model)
    grad_model = tf.keras.models.Model([h5_model.inputs], [h5_model.get_layer(layer_name).output, h5_model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)
    return overlayed

# Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_resized = image.resize((width, height))
    image_array = np.array(image_resized).astype(np.float32)
    image_array = (image_array / 127.5) - 1.0
    processed = np.expand_dims(image_array, axis=0)
    return image, processed

# Title and file upload
st.title("🧠 Brain Tumor Detection with Grad-CAM")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_bytes = uploaded_file.read()
    original_img, processed_img = preprocess_image(img_bytes)

    # Predict using TFLite
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = float(output[0][0])
    threshold = 0.42
    result = 'Tumor' if prediction_score > threshold else 'No Tumor'

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (user_id, result, confidence) VALUES (?, ?, ?)",
        (st.session_state.user_id, result, prediction_score)
    )
    conn.commit()

    # Prediction display
    st.markdown(f"### 🧪 Prediction: `{result}`")
    st.markdown(f"### 📊 Confidence: `{round(prediction_score * 100, 2)}%`")

    if result == "Tumor":
        # Grad-CAM generation
        processed_for_h5 = (np.array(original_img.resize((width, height))) / 127.5) - 1.0
        processed_for_h5 = np.expand_dims(processed_for_h5, axis=0).astype(np.float32)
        gradcam_img = generate_gradcam(processed_for_h5, original_img)

        # Show both images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original MRI", use_container_width=True)
        with col2:
            st.image(gradcam_img, caption="Grad-CAM: Highlighted tumor regions", use_container_width=True)

    else:
        st.image(original_img, caption="Uploaded Image", use_container_width=True)
        st.info("No tumor detected. Grad-CAM explanation is not required.")

#Prediction history per user
st.subheader("📜 Your Prediction History")

conn = get_connection()
cur = conn.cursor()
cur.execute(
    "SELECT result, confidence, timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC",
    (st.session_state.user_id,)
)

rows = cur.fetchall()

if rows:
    st.table(rows)
else:
    st.info("No prediction history yet.")


# Sidebar glossary
with st.sidebar.expander("📖 Glossary (Click to Expand)"):
    st.markdown("### 🧠 Medical Terms")
    st.markdown("- **MRI (Magnetic Resonance Imaging):** Imaging technique used to visualize internal structures of the body, especially brain tissues.")
    st.markdown("- **Tumor:** An abnormal growth of tissue. It can be benign (non-cancerous) or malignant (cancerous).")
    st.markdown("- **Glioma:** A common type of brain tumor arising from glial cells.")
    st.markdown("- **Meningioma:** A tumor that arises from the meninges, the protective membranes around the brain and spinal cord.")
    st.markdown("- **Pituitary Tumor:** A tumor in the pituitary gland, which controls various hormones.")

    st.markdown("### 🧪 AI Terms")
    st.markdown("- **Grad-CAM (Gradient-weighted Class Activation Mapping):** A technique to highlight important regions in the image influencing the model's decision.")
    st.markdown("- **TensorFlow Lite (TFLite):** A lightweight version of TensorFlow optimized for mobile and edge devices.")
    st.markdown("- **.h5 Model:** A format used by Keras to save full models, including weights and architecture.")
    st.markdown("- **Confidence Score:** Indicates how sure the model is about its prediction, expressed as a percentage.")
    st.markdown("- **Threshold:** The decision boundary to classify an image as ‘Tumor’ or ‘No Tumor’.")


    
