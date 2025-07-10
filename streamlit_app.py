import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os

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

# UI
st.title("🧠 Brain Tumor Detection with Grad-CAM")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_bytes = uploaded_file.read()
    original_img, processed_img = preprocess_image(img_bytes)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = float(output[0][0])
    threshold = 0.42
    result = 'Tumor' if prediction_score > threshold else 'No Tumor'

    # Grad-CAM
    processed_for_h5 = (np.array(original_img.resize((width, height))) / 127.5) - 1.0
    processed_for_h5 = np.expand_dims(processed_for_h5, axis=0).astype(np.float32)
    gradcam_img = generate_gradcam(processed_for_h5, original_img)

    st.markdown(f"### 🧪 Prediction: `{result}`")
    st.markdown(f"### 📊 Confidence: `{round(prediction_score * 100, 2)}%`")
    st.markdown("### 🔍 Explanation (Grad-CAM):")
    st.image(gradcam_img, caption="Grad-CAM Result", use_column_width=True)
