import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper
from PIL import Image

# --- Constants ---
FIRST_MODEL_PATH = "EyeImageDetect.keras"
FIRST_CLASS_NAMES = ["Eye Detected", "No Eye Detected"]
SEC_MODEL_PATH = "HPP1P2Detect.keras"
SEC_CLASS_NAMES = ["Healthy", "Pinguecula", "Pterygium Stage 1 (Trace-Mild)", "Pterygium Stage 2 (Moderate-Severe)"]

# Thresholds
CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Condition Detector",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Initialize session state for image management ---
if 'img_raw_bytes' not in st.session_state:
    st.session_state.img_raw_bytes = None
if 'img_for_prediction' not in st.session_state:
    st.session_state.img_for_prediction = None
if 'current_input_method' not in st.session_state:
    st.session_state.current_input_method = "none"

# --- Load Models (Cached) ---
@st.cache_resource
def load_first_model():
    with st.spinner("🚀 Loading AI model for eye detection..."):
        try:
            model = load_model(FIRST_MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"❌ Failed to load eye detection model: {e}. Please ensure '{FIRST_MODEL_PATH}' is in the correct directory.")
            st.stop()

@st.cache_resource
def load_sec_model():
    with st.spinner("🧠 Loading AI model for eye condition analysis..."):
        try:
            model = load_model(SEC_MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"❌ Failed to load eye condition model: {e}. Please ensure '{SEC_MODEL_PATH}' is in the correct directory.")
            st.stop()

first_model = load_first_model()
sec_model = load_sec_model()

# --- Preprocessing ---
def preprocess_image(image_np, target_size=(280, 320)):
    """Resizes, converts to RGB, and expands dimensions for model input."""
    image_resized = cv2.resize(image_np, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image_rgb.astype("float32"), axis=0)
    return image_array

# --- Prediction Logic ---
def predict_eye_detection(image_np):
    processed_image = preprocess_image(image_np)
    prediction = first_model.predict(processed_image)[0]
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[predicted_class_index]
    return FIRST_CLASS_NAMES[predicted_class_index], confidence

def predict_eye_condition(image_np):
    processed_image = preprocess_image(image_np)
    prediction = sec_model.predict(processed_image)[0]

    top_2 = np.sort(prediction)[-2:]
    confidence = top_2[-1]
    margin = top_2[-1] - top_2[-2]

    predicted_class_index = np.argmax(prediction)

    if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
        return "Uncertain", confidence
    return SEC_CLASS_NAMES[predicted_class_index], confidence

# --- Helper Function for Display ---
def display_prediction_result(label, confidence, is_eye_detection=False):
    """Displays prediction results with appropriate styling and advice."""
    if is_eye_detection:
        if "No Eye" in label:
            st.error(f"❌ **{label}** ")
            st.info("Please ensure your image clearly shows an eye. The AI couldn't detect one. Try re-uploading or cropping again.")
        else:
            st.success(f"✅ **{label}** ")
    else: # Eye condition prediction
        if label == "Uncertain":
            st.warning("⚠️ **Uncertain Diagnosis**")
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.info("The AI model's confidence is low, or the results are ambiguous. For a definitive diagnosis, please consult a medical professional.")
        elif "Healthy" in label:
            st.balloons()
            st.success(f"🎉 **{label}!**")
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.info("Great news! Your eye appears healthy based on AI analysis. Remember to still consult a healthcare professional for a complete eye examination.")
        else: # Pinguecula or Pterygium stages
            st.warning(f"🚨 **Potential Condition: {label}**")
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.info("This is an AI-based preliminary finding. It suggests a potential eye condition. **Please seek professional medical advice for proper diagnosis and treatment.**")

            # Add specific advice based on the detected condition
            if label == "Pinguecula":
                st.markdown(
                    """
                    **คำแนะนำเพิ่มเติมสำหรับภาวะต้อลม (Pinguecula):**
                    หากเกิดการระคายเคือง แนะนำให้ใช้ยาหยอดตาเพื่อบรรเทาอาการ แต่ยาหยอดตาเหล่านี้ไม่ได้ช่วยรักษาให้ต้อลมหายไปโดยตรง แต่จะช่วยลดการอักเสบและการระคายเคือง และช่วยป้องกันไม่ให้ต้อลมลุกลามหรืออักเสบเพิ่มขึ้น
                    """
                )
            elif label == "Pterygium Stage 1 (Trace-Mild)":
                st.markdown(
                    """
                    **คำแนะนำเพิ่มเติมสำหรับภาวะต้อเนื้อ ระยะที่ 1 (เริ่มแรก-ไม่รุนแรง):**
                    กรณีเป็นระยะแรกเริ่ม ยาหยอดตาที่ใช้จะช่วยบรรเทาอาการตาแดงและระคายเคือง เพื่อลดการอักเสบ และชะลอการลุกลามของต้อเนื้อ อย่างไรก็ตาม ยาหยอดตาเหล่านี้ไม่ได้รักษาให้ต้อเนื้อหายไป จำเป็นต้องปรึกษาจักษุแพทย์เพื่อรับการตรวจและประเมินเพิ่มเติม
                    """
                )
                st.warning("⚠️ **โปรดปรึกษาจักษุแพทย์:** สำหรับการวินิจฉัยและแผนการรักษาที่เหมาะสม.")
            elif label == "Pterygium Stage 2 (Moderate-Severe)":
                st.markdown(
                    """
                    **คำแนะนำเพิ่มเติมสำหรับภาวะต้อเนื้อ ระยะที่ 2 (ปานกลาง-รุนแรง):**
                    ต้อเนื้อในระยะนี้อาจมีความรุนแรงมากขึ้น และอาจส่งผลต่อการมองเห็นได้ จำเป็นอย่างยิ่งที่จะต้องได้รับการประเมินจากจักษุแพทย์โดยเร็วที่สุด เพื่อพิจารณาแนวทางการรักษาที่เหมาะสม ซึ่งอาจรวมถึงการผ่าตัด
                    """
                )
                st.error("🚨 **โปรดไปพบจักษุแพทย์โดยด่วน:** เพื่อการตรวจวินิจฉัยและวางแผนการรักษาที่จำเป็น.")


# --- Streamlit UI ---

# Header Section
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>👁️ Eye scan AI</h1>
         <p>Your intelligent assistant for preliminary eye health checks (Healthy , Pinguecula , Pterygium).</p>
    </div>
    """,
    unsafe_allow_html=True
)

# How it works / Welcome message
st.markdown("---")
st.markdown(
    """
    **Welcome!** Discover the power of AI to get a quick, preliminary assessment for common eye conditions such as **Pinguecula** and **Pterygium** (early and advanced stages), or simply to check for **healthy** signs.

    **Here's how to use EyeScan AI:**
    1.  **📸 Provide an Image:** Upload a clear photo of your eye from your device or capture one using your camera.
    2.  **✂️ Crop Precisely:** Adjust the cropping box to perfectly frame and focus on your eye. This helps our AI analyze accurately.
    3.  **🔬 Get Analysis:** Click the 'Analyze' button to receive an AI-powered prediction on your eye's condition.

    **Important Disclaimer:** EyeScan AI is an **informational tool only** and is **not a substitute for professional medical advice or diagnosis**. Always consult a qualified ophthalmologist or healthcare provider for any health concerns, proper diagnosis, and treatment.
    """
)
st.markdown("---")

st.subheader("📸 Start Your Eye Scan")
st.markdown("Choose how you'd like to interact with the app:")

st.info("💡 **Tip:** For the most accurate results, ensure your eye image is well-lit and clearly visible!")
tab1, tab2, tab3 = st.tabs(["🖼️ Upload Image", "📸 Use Camera", "✍️ Report & Feedback"])

# --- Function to handle image processing and cropping ---
def handle_image_input(uploaded_bytes, method_name, cropper_key):
    # กรณีภาพใหม่ถูกส่งเข้ามาหรือเปลี่ยน input method
    if (uploaded_bytes is not None and st.session_state.img_raw_bytes != uploaded_bytes) or \
       (st.session_state.current_input_method != method_name and uploaded_bytes is not None):
        st.session_state.img_raw_bytes = uploaded_bytes
        st.session_state.img_for_prediction = None  # ล้างภาพครอปเก่า
        st.session_state.current_input_method = method_name
        st.experimental_rerun()  # เรียก rerun เพื่อโหลดภาพใหม่

    # กรณีล้าง input
    elif uploaded_bytes is None and st.session_state.current_input_method == method_name:
        if st.session_state.img_raw_bytes is not None:
            st.session_state.img_raw_bytes = None
            st.session_state.img_for_prediction = None
            st.session_state.current_input_method = "none"
            st.experimental_rerun()

    # หาก method ปัจจุบันกำลัง active และมีภาพ raw อยู่
    if st.session_state.current_input_method == method_name and st.session_state.img_raw_bytes:
        # แปลง bytes เป็น numpy array (BGR)
        img_np_decoded = cv2.imdecode(np.frombuffer(st.session_state.img_raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        # แปลง BGR -> RGB สำหรับ PIL
        img_pil = Image.fromarray(cv2.cvtColor(img_np_decoded, cv2.COLOR_BGR2RGB))

        st.markdown("### ✂️ Step 2: Crop Your Image")
        st.info("**Drag the box** to perfectly frame your eye. A precise crop leads to more accurate analysis.")

        cropped_img = st_cropper(
            img_pil,
            aspect_ratio=(280, 320),
            box_color='#FF4B4B',
            key=cropper_key
        )

        if cropped_img is not None:
            # แปลง cropped image จาก PIL (RGB) -> numpy BGR สำหรับ model
            img_array = np.array(cropped_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            st.session_state.img_for_prediction = img_bgr

            st.markdown("---")
            st.image(cropped_img, caption="✅ Cropped Image Ready for Analysis", use_container_width=True)
            st.markdown("---")
        else:
            st.session_state.img_for_prediction = None
# --- Image Input & Cropping using Tabs ---
with tab1:
    st.markdown("### 🖼️ Upload an Image from Your Device")
    st.markdown("Upload a photo of your eye from your computer or phone. Supported formats: JPG, JPEG, PNG.")
    uploaded_file = st.file_uploader(
        "Drag & Drop or Click to Upload Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an eye for analysis.",
        key="uploader_widget"
    )
    handle_image_input(uploaded_file.getvalue() if uploaded_file else None, "upload", "uploaded_crop")

with tab2:
    st.markdown("### 📸 Use Your Device's Camera")
    st.markdown("Capture a real-time photo of your eye. Ensure good lighting for best results.")
    camera_input = st.camera_input(
        "Take a Photo of Your Eye",
        help="Take a photo of your eye using your device's camera.",
        key="camera_widget"
    )
    handle_image_input(camera_input.getvalue() if camera_input else None, "camera", "camera_crop")

st.divider()

# --- Prediction Button & Results ---
if st.session_state.img_for_prediction is not None:
    st.markdown("### 🔬 Step 3: Get Your Analysis")
    st.info("Once satisfied with your cropped image, click 'Analyze' to see the AI's findings.")
    if st.button("🚀 Analyze Eye Image", type="primary", use_container_width=True):
        st.subheader("📊 Analysis Results")
        with st.spinner("Analyzing image... Please wait. This may take a few moments."):
            # Create columns for side-by-side display on larger screens, stacks on mobile
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Eye Detection Result")
                eye_label, eye_confidence = predict_eye_detection(st.session_state.img_for_prediction)
                display_prediction_result(eye_label, eye_confidence, is_eye_detection=True)

            if "No Eye Detected" in eye_label and eye_confidence > CONFIDENCE_THRESHOLD:
                # If no eye is detected, no need to proceed to the second model
                col2.markdown("#### Eye Condition Result") # Placeholder for clarity
                col2.warning("🚫 Cannot analyze eye condition without an eye detected.")
            else:
                with col2:
                    st.markdown("#### Eye Condition Analysis")
                    condition_label, condition_confidence = predict_eye_condition(st.session_state.img_for_prediction)
                    display_prediction_result(condition_label, condition_confidence)
else:
    st.info("Upload or capture an image in **Step 1** above, then crop it in **Step 2**. The analysis button will appear here once ready!")

st.divider()
