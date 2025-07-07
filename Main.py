import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile

# --- Constants ---
MODEL_PATH = "DetectModle.keras"
class_names = ["Healthy", "Pinguecula", "Pterygium Stage1(Trace-Mild)", "Pterygium Stage2(Moderate-Severe)"]

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        st.stop()
        return None

model = load_trained_model()

# --- Image Preprocessing ---
def preprocess_image_for_model(image_np):
    image_resized = cv2.resize(image_np, (280, 320))  # Resize to match model input
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image_rgb.astype("float32"), axis=0)

    st.image(image_rgb, caption="Preprocessed Image", use_container_width=True)
    return image_array

# --- Prediction ---
def predict(image_np):
    processed = preprocess_image_for_model(image_np)
    prediction = model.predict(processed)
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
        # Read image from upload as NumPy array
        bytes_data = uploaded_file.read()
        npimg = np.frombuffer(bytes_data, np.uint8)
        img_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        img_to_predict = img_np

    if "camera_image_data" not in st.session_state:
        st.session_state["camera_image_data"] = None

    if uploaded_file is None:
        camera_input = st.camera_input("Take a photo")
        if camera_input is not None:
            bytes_data = camera_input.read()
            npimg = np.frombuffer(bytes_data, np.uint8)
            img_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            st.session_state["camera_image_data"] = img_np
            st.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), caption="Captured Image", use_container_width=True)
            img_to_predict = img_np
        elif st.session_state["camera_image_data"] is not None:
            img_np = st.session_state["camera_image_data"]
            st.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), caption="Previously Captured Image", use_container_width=True)
            img_to_predict = img_np
    else:
        st.session_state["camera_image_data"] = None

    if img_to_predict is not None:
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                label, confidence = predict(img_to_predict)
                st.markdown("### 🧠 Prediction Result")
                st.success(f"**{label}** (Confidence: {confidence*100:.2f}%)")
    else:
        st.info("Please upload an image or take a photo to start prediction.")

    if st.button("Clear Inputs"):
        st.session_state["camera_image_data"] = None
        st.rerun()
