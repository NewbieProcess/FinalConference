import streamlit as st
import numpy as np
import cv2 # Keep cv2 for potential future uses, though tf.keras.preprocessing handles core loading now
from PIL import Image # PIL is still used by Streamlit for uploaded files and camera input
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os

# --- Constants ---
# IMPORTANT: Double check this path! Remove the '!' if it's not part of the actual filename.
MODEL_PATH = "DetectModle.keras" # Assuming the '!' was a typo or placeholder based on common practice
class_names = ["Healthy", "Pinguecula", "Pterygium Stage1(Trace-Mild)", "Pterygium Stage2(Moderate-Severe)"]

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        st.stop() # Stop the app if the model can't be loaded
        return None

model = load_trained_model()


# --- Image Preprocessing Function ---
def preprocess_image_for_model(pil_image_or_path):
    temp_file = None
    if isinstance(pil_image_or_path, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image_or_path.save(tmp.name)
            temp_file = tmp.name
        image_to_load = temp_file
    else:
        image_to_load = pil_image_or_path

    try:
        image = tf.keras.preprocessing.image.load_img(image_to_load, target_size=(320, 280))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        st.write(f"Shape after img_to_array (pixels 0-255): {input_arr.shape}, dtype: {input_arr.dtype}")
        st.image(input_arr.astype(np.uint8), caption="Image after tf.keras.preprocessing (Resized, RGB, 0-255)", use_container_width=True)
        processed = np.expand_dims(input_arr, axis=0)

        st.write(f"Final processed shape for model (pixels 0-255): {processed.shape}")
        return processed

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

# --- Prediction Function ---
def predict(pil_image_or_path):
    processed_image = preprocess_image_for_model(pil_image_or_path)
    prediction = model.predict(processed_image)

    st.write("Raw prediction probabilities:", prediction) # Debug output

    predicted_class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return class_names[predicted_class_index], confidence

# --- Streamlit UI ---
st.title("Pinguecula & Pterygium Detection App")
st.subheader("📷 Upload or capture an image to detect eye condition severity")

page = st.sidebar.selectbox("Navigate", ["Home", "Upload / Take Photo"])

if page == "Home":
    st.info("Choose 'Upload / Take Photo' from the sidebar to get started.")

elif page == "Upload / Take Photo":
    st.write("Upload an image file or take a photo with your webcam.")

    img_to_predict = None

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_to_predict = Image.open(uploaded_file).convert("RGB")  # Ensure it's RGB
        st.image(img_to_predict, caption="Uploaded Image", use_container_width=True)

    # Handle Camera Input (only show if no uploaded file)
    if "camera_image_data" not in st.session_state:
        st.session_state["camera_image_data"] = None

    # ✅ แสดงกล้องเฉพาะเมื่อยังไม่มีการอัปโหลด
    if uploaded_file is None:
        camera_input = st.camera_input("Take a photo")
        if camera_input is not None:
            st.session_state["camera_image_data"] = Image.open(camera_input).convert("RGB")
            img_to_predict = st.session_state["camera_image_data"]
            st.image(img_to_predict, caption="Captured Image", use_container_width=True)
        elif st.session_state["camera_image_data"] is not None:
            # ถ้ายังไม่มีการอัปโหลด และกล้องเคยใช้ → แสดงภาพเดิม
            img_to_predict = st.session_state["camera_image_data"]
            st.image(img_to_predict, caption="Previously Captured Image", use_container_width=True)
    else:
        # ✅ ถ้ามีการอัปโหลดแล้ว → ล้างภาพจากกล้อง
        st.session_state["camera_image_data"] = None

    # Prediction Button
    if img_to_predict is not None:
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                label, confidence = predict(img_to_predict)
                st.markdown("### 🧠 Prediction Result")
                st.success(f"**{label}** (Confidence: {confidence*100:.2f}%)")
    else:
        st.info("Please upload an image or take a photo to start prediction.")

    # Clear Inputs Button
    if st.button("Clear Inputs"):
        st.session_state["camera_image_data"] = None
        st.rerun()