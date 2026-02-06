import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="brain_tumor_classifier (1).tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Keras model for Grad-CAM
model = load_model('brain_tumor_classifier.h5')
last_conv_layer_name = "Conv_1"  # MobileNetV2's final conv layer

# Streamlit title
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection & Explainability")
st.markdown("Upload an MRI brain scan to classify as **Tumor** or **No Tumor** and view Grad-CAM explanations.")

# Image upload
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

# Image size
width, height = 224, 224

# Function to preprocess uploaded image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((width, height))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0  # Normalize for MobileNetV2
    return img, img_array

# Function to generate Grad-CAM
def generate_gradcam(input_image, orig_image):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        loss = predictions[:, 0]  # Binary classification

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (orig_image.size[0], orig_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(orig_image), 0.6, heatmap_color, 0.4, 0)

    return Image.fromarray(superimposed_img)

# If an image is uploaded
if uploaded_file:
    img_bytes = uploaded_file.read()
    original_img, processed_img = preprocess_image(img_bytes)

    # Run TFLite prediction
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = float(output[0][0])
    threshold = 0.42
    result = "Tumor" if prediction_score > threshold else "No Tumor"

    # Display results
    st.markdown(f"### 🧪 Prediction: `{result}`")
    st.markdown(f"### 📊 Confidence: `{round(prediction_score * 100, 2)}%`")

    # Prepare input for Grad-CAM using the Keras model
    processed_for_h5 = (np.array(original_img.resize((width, height))) / 127.5) - 1.0
    processed_for_h5 = np.expand_dims(processed_for_h5, axis=0).astype(np.float32)
    gradcam_img = generate_gradcam(processed_for_h5, original_img)

    # Display original and Grad-CAM image side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original MRI", use_container_width=True)
    with col2:
        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

# Sidebar glossary
with st.sidebar.expander("Glossary (Click to Expand)"):
    st.markdown("### Medical Terms")
    st.markdown("- **MRI (Magnetic Resonance Imaging):** Imaging technique used to visualize internal structures of the body, especially brain tissues.")
    st.markdown("- **Tumor:** An abnormal growth of tissue. It can be benign (non-cancerous) or malignant (cancerous).")
    st.markdown("- **Glioma:** A common type of brain tumor arising from glial cells.")
    st.markdown("- **Meningioma:** A tumor that arises from the meninges, the protective membranes around the brain and spinal cord.")
    st.markdown("- **Pituitary Tumor:** A tumor in the pituitary gland, which controls various hormones.")

    st.markdown("### AI Terms")
    st.markdown("- **Grad-CAM (Gradient-weighted Class Activation Mapping):** A technique to highlight important regions in the image influencing the model's decision.")
    st.markdown("- **TensorFlow Lite (TFLite):** A lightweight version of TensorFlow optimized for mobile and edge devices.")
    st.markdown("- **.h5 Model:** A format used by Keras to save full models, including weights and architecture.")
    st.markdown("- **Confidence Score:** Indicates how sure the model is about its prediction, expressed as a percentage.")
    st.markdown("- **Threshold:** The decision boundary to classify an image as ‘Tumor’ or ‘No Tumor’.")
