import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper
from PIL import Image

# --- Constants ---
FIRST_MODEL_PATH = "EyeDetect.keras"
FIRST_CLASS_NAMES = ["Eye Detected", "No Eye Detected"]
SEC_MODEL_PATH = "EyeAnalysis.keras"
SEC_CLASS_NAMES = ["Healthy", "Pinguecula", "Pterygium Stage 1 (Trace-Mild)", "Pterygium Stage 2 (Moderate-Severe)"]

# Thresholds
CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10

# --- Translation Data ---
TEXTS = {
    "en": {
        "page_title": "Eye Condition Detector",
        "app_header": "👁️ Eye scan AI",
        "app_subheader": "Your intelligent assistant for preliminary eye health checks (Healthy, Pinguecula, Pterygium).",
        "welcome_title": "Welcome!",
        "welcome_message": "Discover the power of AI to get a quick, preliminary assessment for common eye conditions such as **Pinguecula** and **Pterygium** (early and advanced stages), or simply to check for **healthy** signs.",
        "how_to_use_title": "Here's how to use EyeScan AI:",
        "step1_title": "📸 Provide an Image:",
        "step1_desc": "Upload a clear photo of your eye from your device or capture one using your camera.",
        "step2_title": "✂️ Crop Precisely:",
        "step2_desc": "Adjust the cropping box to perfectly frame and focus on your eye. This helps our AI analyze accurately.",
        "step3_title": "🔬 Get Analysis:",
        "step3_desc": "Click the 'Analyze' button to receive an AI-powered prediction on your eye's condition.",
        "disclaimer_title": "Important Disclaimer:",
        "disclaimer_text": "EyeScan AI is an **informational tool only** and is **not a substitute for professional medical advice or diagnosis**. Always consult a qualified ophthalmologist or healthcare provider for any health concerns, proper diagnosis, and treatment.",
        "start_scan_subheader": "📸 Start Your Eye Scan",
        "choose_interaction": "Choose how you'd like to interact with the app:",
        "tip_info": "💡 **Tip:** For the most accurate results, ensure your eye image is well-lit and clearly visible!",
        "tab_upload_image": "🖼️ Upload Image",
        "tab_use_camera": "📸 Use Camera",
        "upload_section_title": "🖼️ Upload an Image from Your Device",
        "upload_section_desc": "Upload a photo of your eye from your computer or phone. Supported formats: JPG, JPEG, PNG.",
        "uploader_label": "Drag & Drop or Click to Upload Image",
        "uploader_help": "Upload a clear image of an eye for analysis.",
        "camera_section_title": "📸 Use Your Device's Camera",
        "camera_section_desc": "Capture a real-time photo of your eye. Ensure good lighting for best results.",
        "camera_label": "Take a Photo of Your Eye",
        "camera_help": "Take a photo of your eye using your device's camera.",
        "crop_step_title": "✂️ Step 2: Crop Your Image",
        "crop_step_info": "**Drag the box** to perfectly frame your eye. A precise crop leads to more accurate analysis.",
        "cropped_image_caption": "✅ Cropped Image Ready for Analysis",
        "analyze_step_title": "🔬 Step 3: Get Your Analysis",
        "analyze_step_info": "Once satisfied with your cropped image, click 'Analyze' to see the AI's findings.",
        "analyze_button": "🚀 Analyze Eye Image",
        "analysis_results_header": "📊 Analysis Results",
        "eye_detection_result_title": "Eye Detection Result",
        "eye_condition_analysis_title": "Eye Condition Analysis",
        "no_eye_detected_error": "❌ **No Eye Detected** ",
        "no_eye_detected_advice": "Please ensure your image clearly shows an eye. The AI couldn't detect one. Try re-uploading or cropping again.",
        "cannot_analyze_condition": "🚫 Cannot analyze eye condition without an eye detected.",
        "uncertain_diagnosis_warning": "⚠️ **Uncertain Diagnosis**",
        "confidence_label": "Confidence:",
        "uncertain_advice": "The AI model's confidence is low, or the results are ambiguous. For a definitive diagnosis, please consult a medical professional.",
        "healthy_success": "🎉 **Healthy!**",
        "healthy_advice": "Great news! Your eye appears healthy based on AI analysis. Remember to still consult a healthcare professional for a complete eye examination.",
        "potential_condition_warning": "🚨 **Potential Condition: {}**",
        "professional_advice_needed": "This is an AI-based preliminary finding. It suggests a potential eye condition. **Please seek professional medical advice for proper diagnosis and treatment.**",
        "pinguecula_advice": """
        **Additional advice for Pinguecula:**
        If irritation occurs, it is recommended to use eye drops to alleviate symptoms. However, these eye drops do not directly cure pinguecula but help reduce inflammation and irritation and help prevent pinguecula from worsening or becoming more inflamed.
        """,
        "pterygium1_advice": """
        **Additional advice for Pterygium Stage 1 (Trace-Mild):**
        In the early stages, eye drops can help relieve red eyes and irritation, reduce inflammation, and slow the progression of pterygium. However, these eye drops do not cure pterygium. It is necessary to consult an ophthalmologist for further examination and assessment.
        """,
        "pterygium1_consult_doctor": "⚠️ **Please consult an ophthalmologist:** For proper diagnosis and treatment plan.",
        "pterygium2_advice": """
        **Additional advice for Pterygium Stage 2 (Moderate-Severe):**
        Pterygium at this stage may be more severe and can affect vision. It is crucial to be assessed by an ophthalmologist as soon as possible to consider appropriate treatment, which may include surgery.
        """,
        "pterygium2_consult_doctor": "🚨 **Please see an ophthalmologist urgently:** For necessary diagnosis and treatment planning.",
        "initial_message": "Upload or capture an image in **Step 1** above, then crop it in **Step 2**. The analysis button will appear here once ready!",
        "loading_first_model": "🚀 Loading AI model for eye detection...",
        "failed_to_load_first_model": "❌ Failed to load eye detection model: {}. Please ensure '{}}' is in the correct directory.",
        "loading_sec_model": "🧠 Loading AI model for eye condition analysis...",
        "failed_to_load_sec_model": "❌ Failed to load eye condition model: {}. Please ensure '{}}' is in the correct directory.",
        "analyzing_image": "Analyzing image... Please wait. This may take a few moments.",
        "language_selector_label": "Select Language",
        "sidebar_settings_title": "Settings"
    },
    "th": {
        "page_title": "เครื่องมือตรวจสภาพดวงตา",
        "app_header": "👁️ AI สแกนดวงตา",
        "app_subheader": "ผู้ช่วยอัจฉริยะของคุณสำหรับการตรวจสุขภาพตาเบื้องต้น (ตาปกติ, ต้อลม, ต้อเนื้อ).",
        "welcome_title": "ยินดีต้อนรับ!",
        "welcome_message": "ค้นพบพลังของ AI เพื่อรับการประเมินเบื้องต้นอย่างรวดเร็วสำหรับภาวะดวงตาที่พบบ่อย เช่น **ต้อลม (Pinguecula)** และ **ต้อเนื้อ (Pterygium)** (ระยะเริ่มต้นและระยะลุกลาม) หรือเพียงแค่ตรวจสอบสัญญาณของ **สุขภาพตาที่ดี**.",
        "how_to_use_title": "วิธีใช้งาน EyeScan AI:",
        "step1_title": "📸 ใส่รูปภาพ:",
        "step1_desc": "อัปโหลดรูปภาพดวงตาที่ชัดเจนจากอุปกรณ์ของคุณ หรือถ่ายรูปโดยใช้กล้องของคุณ.",
        "step2_title": "✂️ ครอบตัดอย่างแม่นยำ:",
        "step2_desc": "ปรับกรอบการครอบตัดเพื่อให้โฟกัสที่ดวงตาของคุณอย่างสมบูรณ์ สิ่งนี้ช่วยให้ AI ของเราวิเคราะห์ได้อย่างแม่นยำ.",
        "step3_title": "🔬 รับการวิเคราะห์:",
        "step3_desc": "คลิกปุ่ม 'วิเคราะห์' เพื่อรับการคาดการณ์จาก AI เกี่ยวกับสภาพดวงตาของคุณ.",
        "disclaimer_title": "คำเตือนที่สำคัญ:",
        "disclaimer_text": "EyeScan AI เป็น **เครื่องมือให้ข้อมูลเท่านั้น** และ **ไม่สามารถใช้แทนคำแนะนำทางการแพทย์หรือการวินิจฉัยจากผู้เชี่ยวชาญได้** โปรดปรึกษาจักษุแพทย์หรือผู้ให้บริการด้านสุขภาพที่มีคุณสมบัติเหมาะสมสำหรับข้อกังวลด้านสุขภาพ การวินิจฉัยและการรักษาที่ถูกต้องเสมอ.",
        "start_scan_subheader": "📸 เริ่มการสแกนดวงตาของคุณ",
        "choose_interaction": "เลือกวิธีการโต้ตอบกับแอป:",
        "tip_info": "💡 **เคล็ดลับ:** เพื่อผลลัพธ์ที่แม่นยำที่สุด ตรวจสอบให้แน่ใจว่าภาพดวงตาของคุณมีแสงสว่างเพียงพอและมองเห็นได้ชัดเจน!",
        "tab_upload_image": "🖼️ อัปโหลดรูปภาพ",
        "tab_use_camera": "📸 ใช้กล้อง",
        "upload_section_title": "🖼️ อัปโหลดรูปภาพจากอุปกรณ์ของคุณ",
        "upload_section_desc": "อัปโหลดรูปภาพดวงตาจากคอมพิวเตอร์หรือโทรศัพท์ของคุณ รูปแบบที่รองรับ: JPG, JPEG, PNG.",
        "uploader_label": "ลากและวาง หรือคลิกเพื่ออัปโหลดรูปภาพ",
        "uploader_help": "อัปโหลดรูปภาพดวงตาที่ชัดเจนเพื่อทำการวิเคราะห์.",
        "camera_section_title": "📸 ใช้กล้องของอุปกรณ์ของคุณ",
        "camera_section_desc": "ถ่ายรูปดวงตาแบบเรียลไทม์ ตรวจสอบให้แน่ใจว่ามีแสงสว่างเพียงพอเพื่อผลลัพธ์ที่ดีที่สุด.",
        "camera_label": "ถ่ายรูปดวงตาของคุณ",
        "camera_help": "ถ่ายรูปดวงตาของคุณโดยใช้กล้องของอุปกรณ์.",
        "crop_step_title": "✂️ ขั้นตอนที่ 2: ครอบตัดรูปภาพของคุณ",
        "crop_step_info": "**ลากกรอบ** เพื่อจัดวางดวงตาของคุณให้สมบูรณ์ การครอบตัดที่แม่นยำจะนำไปสู่การวิเคราะห์ที่แม่นยำยิ่งขึ้น.",
        "cropped_image_caption": "✅ รูปภาพที่ครอบตัดพร้อมสำหรับการวิเคราะห์",
        "analyze_step_title": "🔬 ขั้นตอนที่ 3: รับการวิเคราะห์ของคุณ",
        "analyze_step_info": "เมื่อพอใจกับรูปภาพที่ครอบตัดแล้ว คลิก 'วิเคราะห์' เพื่อดูผลการค้นหาของ AI.",
        "analyze_button": "🚀 วิเคราะห์รูปภาพดวงตา",
        "analysis_results_header": "📊 ผลการวิเคราะห์",
        "eye_detection_result_title": "ผลการตรวจจับดวงตา",
        "eye_condition_analysis_title": "การวิเคราะห์สภาพดวงตา",
        "no_eye_detected_error": "❌ **ไม่พบดวงตา** ",
        "no_eye_detected_advice": "โปรดตรวจสอบให้แน่ใจว่ารูปภาพของคุณแสดงดวงตาอย่างชัดเจน AI ไม่สามารถตรวจพบดวงตาได้ ลองอัปโหลดใหม่หรือครอบตัดอีกครั้ง.",
        "cannot_analyze_condition": "🚫 ไม่สามารถวิเคราะห์สภาพดวงตาได้หากไม่พบดวงตา.",
        "uncertain_diagnosis_warning": "⚠️ **การวินิจฉัยไม่แน่ชัด**",
        "confidence_label": "ความมั่นใจ:",
        "uncertain_advice": "ความมั่นใจของแบบจำลอง AI ต่ำ หรือผลลัพธ์ไม่ชัดเจน สำหรับการวินิจฉัยที่ชัดเจน โปรดปรึกษาผู้เชี่ยวชาญทางการแพทย์.",
        "healthy_success": "🎉 **สุขภาพดี!**",
        "healthy_advice": "ข่าวดี! ดวงตาของคุณดูมีสุขภาพดีจากการวิเคราะห์ของ AI โปรดจำไว้ว่ายังคงต้องปรึกษาผู้เชี่ยวชาญด้านสุขภาพสำหรับการตรวจตาที่สมบูรณ์.",
        "potential_condition_warning": "🚨 **ภาวะที่อาจเกิดขึ้น: {}**",
        "professional_advice_needed": "นี่เป็นผลการค้นหาเบื้องต้นจาก AI ซึ่งบ่งชี้ถึงภาวะดวงตาที่อาจเกิดขึ้น **โปรดปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการวินิจฉัยและการรักษาที่เหมาะสม**",
        "pinguecula_advice": """
        **คำแนะนำเพิ่มเติมสำหรับภาวะต้อลม (Pinguecula):**
        หากเกิดการระคายเคือง แนะนำให้ใช้ยาหยอดตาเพื่อบรรเทาอาการ แต่ยาหยอดตาเหล่านี้ไม่ได้ช่วยรักษาให้ต้อลมหายไปโดยตรง แต่จะช่วยลดการอักเสบและการระคายเคือง และช่วยป้องกันไม่ให้ต้อลมลุกลามหรืออักเสบเพิ่มขึ้น
        """,
        "pterygium1_advice": """
        **คำแนะนำเพิ่มเติมสำหรับภาวะต้อเนื้อ ระยะที่ 1 (เริ่มแรก-ไม่รุนแรง):**
        กรณีเป็นระยะแรกเริ่ม ยาหยอดตาที่ใช้จะช่วยบรรเทาอาการตาแดงและระคายเคือง เพื่อลดการอักเสบ และชะลอการลุกลามของต้อเนื้อ อย่างไรก็ตาม ยาหยอดตาเหล่านี้ไม่ได้รักษาให้ต้อเนื้อหายไป จำเป็นต้องปรึกษาจักษุแพทย์เพื่อรับการตรวจและประเมินเพิ่มเติม
        """,
        "pterygium1_consult_doctor": "⚠️ **โปรดปรึกษาจักษุแพทย์:** สำหรับการวินิจฉัยและแผนการรักษาที่เหมาะสม.",
        "pterygium2_advice": """
        **คำแนะนำเพิ่มเติมสำหรับภาวะต้อเนื้อ ระยะที่ 2 (ปานกลาง-รุนแรง):**
        ต้อเนื้อในระยะนี้อาจมีความรุนแรงมากขึ้น และอาจส่งผลต่อการมองเห็นได้ จำเป็นอย่างยิ่งที่จะต้องได้รับการประเมินจากจักษุแพทย์โดยเร็วที่สุด เพื่อพิจารณาแนวทางการรักษาที่เหมาะสม ซึ่งอาจรวมถึงการผ่าตัด
        """,
        "pterygium2_consult_doctor": "🚨 **โปรดไปพบจักษุแพทย์โดยด่วน:** เพื่อการตรวจวินิจฉัยและวางแผนการรักษาที่จำเป็น.",
        "initial_message": "อัปโหลดหรือถ่ายรูปภาพใน **ขั้นตอนที่ 1** ด้านบน จากนั้นครอบตัดใน **ขั้นตอนที่ 2** ปุ่มวิเคราะห์จะปรากฏขึ้นที่นี่เมื่อพร้อม!",
        "loading_first_model": "🚀 กำลังโหลดโมเดล AI สำหรับตรวจจับดวงตา...",
        "failed_to_load_first_model": "❌ โหลดโมเดลตรวจจับดวงตาไม่สำเร็จ: {}. โปรดตรวจสอบให้แน่ใจว่า '{}}' อยู่ในไดเรกทอรีที่ถูกต้อง.",
        "loading_sec_model": "🧠 กำลังโหลดโมเดล AI สำหรับการวิเคราะห์สภาพดวงตา...",
        "failed_to_load_sec_model": "❌ โหลดโมเดลสภาพดวงตาไม่สำเร็จ: {}. โปรดตรวจสอบให้แน่ใจว่า '{}}' อยู่ในไดเรกทอรีที่ถูกต้อง.",
        "analyzing_image": "กำลังวิเคราะห์รูปภาพ... โปรดรอสักครู่ อาจใช้เวลาเล็กน้อย.",
        "language_selector_label": "เลือกภาษา",
        "sidebar_settings_title": "การตั้งค่า"
    }
}

# --- Initialize session state for language ---
if 'language' not in st.session_state:
    st.session_state.language = 'en' # Default to English

def get_text(key, *args):
    """Retrieves translated text for a given key in the current language."""
    text = TEXTS[st.session_state.language].get(key, f"Translation Missing: {key}")
    if args:
        return text.format(*args)
    return text

# --- Page Configuration ---
st.set_page_config(
    page_title=get_text("page_title"),
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
    with st.spinner(get_text("loading_first_model")):
        try:
            model = load_model(FIRST_MODEL_PATH)
            return model
        except Exception as e:
            st.error(get_text("failed_to_load_first_model", e, FIRST_MODEL_PATH))
            st.stop()

@st.cache_resource
def load_sec_model():
    with st.spinner(get_text("loading_sec_model")):
        try:
            model = load_model(SEC_MODEL_PATH)
            return model
        except Exception as e:
            st.error(get_text("failed_to_load_sec_model", e, SEC_MODEL_PATH))
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
            st.error(get_text("no_eye_detected_error"))
            st.info(get_text("no_eye_detected_advice"))
        else:
            st.success(f"✅ **{label}** ")
    else: # Eye condition prediction
        if label == "Uncertain":
            st.warning(get_text("uncertain_diagnosis_warning"))
            st.write(f"{get_text('confidence_label')} {confidence * 100:.2f}%")
            st.info(get_text("uncertain_advice"))
        elif "Healthy" in label:
            st.balloons()
            st.success(get_text("healthy_success"))
            st.write(f"{get_text('confidence_label')} {confidence * 100:.2f}%")
            st.info(get_text("healthy_advice"))
        else: # Pinguecula or Pterygium stages
            st.warning(get_text("potential_condition_warning").format(label))
            st.write(f"{get_text('confidence_label')} {confidence * 100:.2f}%")
            st.info(get_text("professional_advice_needed"))

            # Add specific advice based on the detected condition
            if label == "Pinguecula":
                st.markdown(get_text("pinguecula_advice"))
            elif label == "Pterygium Stage 1 (Trace-Mild)":
                st.markdown(get_text("pterygium1_advice"))
                st.warning(get_text("pterygium1_consult_doctor"))
            elif label == "Pterygium Stage 2 (Moderate-Severe)":
                st.markdown(get_text("pterygium2_advice"))
                st.error(get_text("pterygium2_consult_doctor"))


# --- Streamlit UI ---

# Sidebar for language selection
with st.sidebar:
    st.title(get_text("sidebar_settings_title"))
    language_options = {
        "en": "English",
        "th": "ภาษาไทย"
    }
    selected_lang_key = st.selectbox(
        get_text("language_selector_label"),
        options=list(language_options.keys()),
        format_func=lambda x: language_options[x],
        index=list(language_options.keys()).index(st.session_state.language)
    )

    if selected_lang_key != st.session_state.language:
        st.session_state.language = selected_lang_key
        st.rerun() # Rerun the app to apply the new language immediately

# Header Section
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>{get_text("app_header")}</h1>
         <p>{get_text("app_subheader")}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# How it works / Welcome message
st.markdown("---")
st.markdown(
    f"""
    **{get_text("welcome_title")}** {get_text("welcome_message")}

    **{get_text("how_to_use_title")}**
    1.  **{get_text("step1_title")}** {get_text("step1_desc")}
    2.  **{get_text("step2_title")}** {get_text("step2_desc")}
    3.  **{get_text("step3_title")}** {get_text("step3_desc")}

    **{get_text("disclaimer_title")}** {get_text("disclaimer_text")}
    """
)
st.markdown("---")

st.subheader(get_text("start_scan_subheader"))
st.markdown(get_text("choose_interaction"))

st.info(get_text("tip_info"))
tab1, tab2= st.tabs([get_text("tab_upload_image"), get_text("tab_use_camera")])

# --- Function to handle image processing and cropping ---
def handle_image_input(uploaded_bytes, method_name, cropper_key):
    # Case 1: A new raw image is provided OR the input method has switched
    if (uploaded_bytes is not None and st.session_state.img_raw_bytes != uploaded_bytes) or \
       (st.session_state.current_input_method != method_name and uploaded_bytes is not None):
        st.session_state.img_raw_bytes = uploaded_bytes
        st.session_state.img_for_prediction = None  # Clear previously cropped image
        st.session_state.current_input_method = method_name
        st.rerun() # Trigger a rerun to clear old display elements and re-render with new raw image for cropper

    # Case 2: The 'x' button was clicked, or camera input was cleared (uploaded_bytes is None)
    # and the current method matches. This means the user explicitly cleared the input.
    elif uploaded_bytes is None and st.session_state.current_input_method == method_name:
        if st.session_state.img_raw_bytes is not None: # Only clear if there was an image to begin with
            st.session_state.img_raw_bytes = None
            st.session_state.img_for_prediction = None
            st.session_state.current_input_method = "none" # Reset active method
            st.rerun() # Trigger a rerun to clear the display

    # If the current input method is active and we have raw image bytes
    if st.session_state.current_input_method == method_name and st.session_state.img_raw_bytes:
        # Decode bytes to numpy array using OpenCV
        img_np_decoded = cv2.imdecode(np.frombuffer(st.session_state.img_raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        # Convert OpenCV's BGR to PIL's RGB
        img_pil = Image.fromarray(cv2.cvtColor(img_np_decoded, cv2.COLOR_BGR2RGB))

        st.markdown(f"### {get_text('crop_step_title')}")
        st.info(get_text("crop_step_info"))
        cropped_img = st_cropper(
            img_pil,
            aspect_ratio=(280, 320),
            box_color='#FF4B4B', # A distinct color for the crop box
            key=cropper_key
        )
        if cropped_img:
            # Update the image for prediction ONLY if the cropper provides a valid output
            st.session_state.img_for_prediction = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2RGB) # Ensure RGB for further processing
            st.markdown("---")
            st.image(cropped_img, caption=get_text("cropped_image_caption"), use_container_width=True)
            st.markdown("---")
        else:
            # If cropped_img is None (e.g., first render of cropper after new upload), ensure img_for_prediction is cleared
            st.session_state.img_for_prediction = None


# --- Image Input & Cropping using Tabs ---
with tab1:
    st.markdown(f"### {get_text('upload_section_title')}")
    st.markdown(get_text("upload_section_desc"))
    uploaded_file = st.file_uploader(
        get_text("uploader_label"),
        type=["jpg", "jpeg", "png"],
        help=get_text("uploader_help"),
        key="uploader_widget"
    )
    handle_image_input(uploaded_file.getvalue() if uploaded_file else None, "upload", "uploaded_crop")

with tab2:
    st.markdown(f"### {get_text('camera_section_title')}")
    st.markdown(get_text("camera_section_desc"))
    camera_input = st.camera_input(
        get_text("camera_label"),
        help=get_text("camera_help"),
        key="camera_widget"
    )
    handle_image_input(camera_input.getvalue() if camera_input else None, "camera", "camera_crop")

st.divider()

# --- Prediction Button & Results ---
if st.session_state.img_for_prediction is not None:
    st.markdown(f"### {get_text('analyze_step_title')}")
    st.info(get_text("analyze_step_info"))
    if st.button(get_text("analyze_button"), type="primary", use_container_width=True):
        st.subheader(get_text("analysis_results_header"))
        with st.spinner(get_text("analyzing_image")):
            # Create columns for side-by-side display on larger screens, stacks on mobile
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {get_text('eye_detection_result_title')}")
                eye_label, eye_confidence = predict_eye_detection(st.session_state.img_for_prediction)
                display_prediction_result(eye_label, eye_confidence, is_eye_detection=True)

            if "No Eye Detected" in eye_label and eye_confidence > CONFIDENCE_THRESHOLD:
                # If no eye is detected, no need to proceed to the second model
                col2.markdown(f"#### {get_text('eye_condition_analysis_title')}") # Placeholder for clarity
                col2.warning(get_text("cannot_analyze_condition"))
            else:
                with col2:
                    st.markdown(f"#### {get_text('eye_condition_analysis_title')}")
                    condition_label, condition_confidence = predict_eye_condition(st.session_state.img_for_prediction)
                    display_prediction_result(condition_label, condition_confidence)
else:
    st.info(get_text("initial_message"))

st.divider()
