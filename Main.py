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
SEC_CLASS_NAMES = ["Healthy", "Pinguecula", "Pterygium Stage 1 (Trace-Mild)", "Pterygium Stage 2 (Moderate-Severe)", "Red Eye(Conjunctivitis)"]

# Thresholds
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.10

# --- Translation Data ---
TEXTS = {
    "en": {
        "page_title": "Ocular scan ",
        "app_header": "OcuScanAI",
        "app_subheader": "Your intelligent assistant for preliminary eye health checks (Healthy, Pinguecula, Pterygium, Red Eye).",
        "welcome_title": "Welcome!",
        "welcome_message": "Let AI help you quickly screen for common eye conditions like Pinguecula, Pterygium (both early and advanced stages), Red Eye, or just check if your eyes appear healthy.",
        "how_to_use_title": "How to use",
        "step1_title": "📸 Input an Image",
        "step1_desc": "Take or upload a clear photo of your eye (just make sure we can see your full eye like 👁️) so we can help check it better!",
        "step2_title": "✂️ Crop your image",
        "step2_desc": "Drag the box to perfectly frame your eye. A precise crop helps our AI analyze it more accurately.",
        "step3_title": "🔬 Get the result",
        "step3_desc": "Click the 'Analyze' button to receive an AI-powered prediction on your eye's condition.",
        "disclaimer_title": "Important Disclaimer:",
        "disclaimer_text": "EyeScan AI is an **informational tool only** and is **not a substitute for professional medical advice or diagnosis**. Always consult a qualified ophthalmologist or healthcare provider for any health concerns, proper diagnosis, and treatment.",
        "start_scan_subheader": "📸 Start Your Eye Scan",
        "choose_interaction": "Choose how you'd like to use the app:",
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
        "crop_step_info": "Drag the box to perfectly frame your eye. A precise crop leads to more accurate analysis.",
        "cropped_image_caption": "✅ Cropped Image Ready for Analysis",
        "analyze_step_title": "🔬 Step 3: Get Your Analysis",
        "analyze_step_info": "Once satisfied with your cropped image, click 'Analyze' to see the AI's findings.",
        "analyze_button": "🚀 Analyze Eye Image",
        "analysis_results_header": "📊 Analysis Results",
        "eye_detection_result_title": "Eye Detection Result",
        "eye_condition_analysis_title": "Eye Condition Analysis",
        "no_eye_detected_error": "❌ **No Eye Detected**",
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
        Pterygium at this stage may be more severe and can affect vision , as it is approaching or nearly covering the pupil. It is crucial to be assessed by an ophthalmologist as soon as possible to consider appropriate treatment, which may include surgery.
        """,
        "pterygium2_consult_doctor": "🚨 **Please see an ophthalmologist urgently:** For necessary diagnosis and treatment planning.",
        "red_eye_advice": """
        **Additional advice for Red Eye:**
        Redness in the eye can be caused by many factors, including irritation, allergies, infection, or other underlying conditions. While often harmless, persistent or severe redness, especially with pain, discharge, or vision changes, warrants medical attention.
        """,
        "red_eye_consult_doctor": "⚠️ **Please consult a healthcare professional or ophthalmologist:** To determine the cause of the redness and receive appropriate treatment.",
        "initial_message": "Upload or capture an image in **Step 1** above, then crop it in **Step 2**. The analysis button will appear here once ready!",
        "loading_first_model": "🚀 Loading AI model for eye detection...",
        "loading_sec_model": "🧠 Loading AI model for eye condition analysis...",
        "analyzing_image": "Analyzing image... Please wait. This may take a few moments.",
        "language_selector_label": "Select Language",
        "sidebar_settings_title": "Settings"
    },
  "th": {
    "page_title": "เครื่องมือตรวจสภาพดวงตา",
    "app_header": "OcuScanAI",
    "app_subheader": "ผู้ช่วยตรวจสุขภาพตาด้วยตัวเอง (เช็คตาปกติ ต้อลม ต้อเนื้อ ตาแดง).",
    "welcome_title": "ยินดีต้อนรับครับ!",
    "welcome_message": "ให้ AI ช่วยตรวจเบื้องต้นว่าตาของคุณเป็นต้อลม ต้อเนื้อ (ตั้งแต่ระยะเริ่มต้นจนถึงระยะรุนแรง) ตาแดง หรือแค่เช็คว่าตาดูปกติดีอยู่ไหมแบบรวดเร็วและง่ายครับ",
    "how_to_use_title": "วิธีการใช้งาน",
    "step1_title": "📸 ขั้นตอนที่ 1: ใส่รูปภาพ",
    "step1_desc": "อัปโหลดรูปถ่ายดวงตาที่ชัดหรือจะถ่ายด้วยกล้อง (แต่ต้องเห็นดวงตาทั้งดวงแบบชัดๆนะ 👁️) เพื่อให้ AI วิเคราะห์ได้แม่นยำขึ้น",
    "step2_title": "✂️ ขั้นตอนที่ 2: ครอบตัดรูป",
    "step2_desc": "ลากกรอบครอบตัดให้พอดีกับดวงตา",
    "step3_title": "🔬 ขั้นตอนที่ 3: ดูผลวิเคราะห์",
    "step3_desc": "กดปุ่ม 'วิเคราะห์' เพื่อดูผลการวินิจฉัยเบื้องต้นจาก AI ครับ",
    "disclaimer_title": "ข้อควรทราบ:",
    "disclaimer_text": "OcuScanAI เป็นแค่เครื่องมือช่วยดูข้อมูลเบื้องต้นเท่านั้น ไม่ใช่คำแนะนำหรือการวินิจฉัยจากแพทย์ หากมีอาการหรือข้อสงสัย ควรไปพบจักษุแพทย์เพื่อรับคำแนะนำที่ถูกต้องครับ",
    "start_scan_subheader": "📸 เริ่มสแกนดวงตาของคุณได้เลยครับ",
    "choose_interaction": "เลือกวิธีใช้แอปได้เลยครับ:",
    "tip_info": "💡 **เคล็ดลับ:** ใช้รูปถ่ายที่มีแสงสว่างเพียงพอ และเห็นดวงตาชัด ๆ เพื่อผลลัพธ์ที่แม่นยำที่สุดครับ!",
    "tab_upload_image": "🖼️ อัปโหลดรูป",
    "tab_use_camera": "📸 ใช้กล้อง",
    "upload_section_title": "🖼️ อัปโหลดรูปจากเครื่องของคุณครับ",
    "upload_section_desc": "เลือกอัปโหลดรูปดวงตาจากคอมพิวเตอร์หรือมือถือรองรับเฉพาะไฟล์ **JPG, JPEG, PNG**",
    "uploader_label": "ลากรูปมาวางหรือคลิกเพื่อเลือกไฟล์",
    "uploader_help": "อัปโหลดรูปถ่ายดวงตาที่ชัดเจนเพื่อให้ AI วิเคราะห์ครับ",
    "camera_section_title": "📸 อัพรูปจากกล้อง",
    "camera_section_desc": "ถ่ายรูปดวงตาควรตรวจสอบให้มีแสงสว่างพอเหมาะเพื่อภาพที่ชัดเจนครับ",
    "camera_label": "ถ่ายรูปดวงตาของคุณครับ",
    "camera_help": "ถ่ายรูปดวงตาด้วยกล้องอุปกรณ์ของคุณครับ",
    "crop_step_title": "✂️ ขั้นตอนที่ 2: ครอบตัดรูปของคุณ",
    "crop_step_info": "ลากกรอบครอบให้พอดีกับดวงตา",
    "cropped_image_caption": "✅ รูปที่ครอบตัดพร้อมสำหรับวิเคราะห์",
    "analyze_step_title": "🔬 ขั้นตอนที่ 3: ผลวิเคราะห์",
    "analyze_step_info": "เมื่อพอใจกับรูปที่ครอบแล้วสามารถกดปุ่ม 'วิเคราะห์' เพื่อดูผลได้ครับ",
    "analyze_button": "🚀 วิเคราะห์รูปดวงตา",
    "analysis_results_header": "📊 ผลวิเคราะห์",
    "eye_detection_result_title": "ผลตรวจจับรูปดวงตา",
    "eye_condition_analysis_title": "ผลวิเคราะห์สภาพดวงตาครับ",
    "no_eye_detected_error": "❌ **ไม่พบดวงตา**",
    "no_eye_detected_advice": "ตอนนี้ AI ยังตรวจสอบดวงตาของคุณไม่ได้ ลองอัพรูปหรือครอปรูปใหม่อีกทีดูนะครับ",
    "cannot_analyze_condition": "🚫 ไม่สามารถวิเคราะห์ได้ ไม่พบดวงตาในรูป",
    "uncertain_diagnosis_warning": "⚠️ **ผลไม่แน่ชัด**",
    "confidence_label": "ความมั่นใจ:",
    "uncertain_advice": "AI ยังไม่มั่นใจในผลนี้ครับ",
    "healthy_success": "🎉 **ตาดูปกติดีครับ!**",
    "healthy_advice": "ดีมากครับ! ดวงตาของคุณดูปกติดี แต่ควรไปตรวจตากับแพทย์เป็นประจำด้วยนะครับ",
    "potential_condition_warning": "🚨 **พบภาวะที่อาจเป็น: {} ครับ**",
    "professional_advice_needed": "นี่เป็นแค่การวิเคราะห์เบื้องต้นจากAIเท่านั้น ควรไปพบแพทย์เพื่อวินิจฉัยและรักษาอย่างถูกต้องครับ",
    "pinguecula_advice": "**คำแนะนำเพิ่มเติมสำหรับต้อลมครับ:** ถ้าตาเริ่มระคายเคือง อาจใช้ยาหยอดตาช่วยบรรเทาอาการได้ แต่ยาหยอดตาไม่ได้รักษาต้อลมให้หายไปโดยตรงนะครับ ช่วยลดอาการอักเสบและระคายเคือง และป้องกันไม่ให้ต้อลมลุกลามครับ",
    "pterygium1_advice": "**คำแนะนำสำหรับต้อเนื้อ ระยะที่ 1 (เริ่มต้น) :** ระยะแรกสามารถใช้ยาหยอดตาเพื่อลดตาแดงและระคายเคือง ช่วยลดการอักเสบและชะลอการลุกลามแต่ยาหยอดตาไม่สามารถรักษาต้อเนื้อให้หายได้ ควรไปพบจักษุแพทย์เพื่อตรวจเพิ่มเติม",
    "pterygium1_consult_doctor": "⚠️ **โปรดพบจักษุแพทย์ครับ:** เพื่อวินิจฉัยและวางแผนรักษาที่เหมาะสม",
    "pterygium2_advice": "**คำแนะนำสำหรับต้อเนื้อ ระยะที่ 2 (รุนแรง) ครับ:** ต้อเนื้อระยะนี้อาจมีผลต่อการมองเห็นเพราะใกล้เข้าสู้รูม่านตามากๆหรือเข้าสู่รูม่านตาแล้ว ควรไปพบแพทย์โดยเร็วเพื่อประเมินและพิจารณาการรักษา ซึ่งอาจรวมถึงการผ่าตัด",
    "pterygium2_consult_doctor": "🚨 **โปรดไปพบจักษุแพทย์ด่วนครับ:** เพื่อรับคำวินิจฉัยและรักษา",
    "red_eye_advice": """**คำแนะนำเพิ่มเติมสำหรับตาแดงครับ:**
    ตาแดงอาจเกิดได้จากหลายสาเหตุ เช่น การระคายเคือง, ภูมิแพ้, การติดเชื้อ หรือภาวะทางการแพทย์อื่น ๆ แม้ว่ามักจะไม่เป็นอันตราย แต่หากตาแดงมีอาการต่อเนื่องหรือรุนแรง โดยเฉพาะอย่างยิ่งมีอาการปวด, มีขี้ตา, หรือการมองเห็นเปลี่ยนแปลงไป ควรปรึกษาแพทย์""",
    "red_eye_consult_doctor": "⚠️ **โปรดปรึกษาแพทย์หรือจักษุแพทย์:** เพื่อหาสาเหตุของตาแดงและรับการรักษาที่เหมาะสมครับ",
    "initial_message": "อัปโหลดหรือถ่ายรูปใน **ขั้นตอนที่ 1** แล้วครอบตัดใน **ขั้นตอนที่ 2** ปุ่มวิเคราะห์จะโผล่มาเมื่อพร้อมใช้งานครับ!",
    "loading_first_model": "🚀 กำลังโหลดโมเดล AI สำหรับตรวจจับดวงตา...",
    "loading_sec_model": "🧠 กำลังโหลดโมเดล AI สำหรับวิเคราะห์สภาพตา...",
    "analyzing_image": "กำลังวิเคราะห์รูปภาพ... กรุณารอสักครู่ครับ",
    "language_selector_label": "เลือกภาษา",
    "sidebar_settings_title": "ตั้งค่า"
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

# --- Apply Custom CSS for a better look and feel ---
st.markdown("""
<style>
/* Center the main header and add a professional look */
h1 {
    text-align: center;
    color: var(--primary-color);
    font-size: 3em;
    font-weight: 700;
}
p {
    text-align: center;
    font-size: 1.1em;
}
/* This makes the divider a bit more visible in both themes */
hr {
    border-top: 1px solid var(--primary-color);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: nowrap;
    border-radius: 4px;
    background-color: var(--secondary-background-color);
    gap: 5px;
    padding-top: 10px;
    padding-bottom: 10px;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--secondary-background-color);
    color: var(--primary-color) !important;
    border-bottom: 2px solid var(--primary-color) !important;
}
/* Style for the "Analyze" button */
.stButton>button {
    background-color: var(--primary-color);
    color: var(--text-color);
    font-size: 1.2em;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    width: 100%;
}
.stButton>button:hover {
    background-color: var(--secondary-background-color);
    color: var(--primary-color);
}
/* Custom styling for the "How to Use" steps */
.step-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 20px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.step {
    text-align: center;
    padding: 15px;
    border: 1px solid var(--secondary-background-color);
    border-radius: 10px;
    background-color: lightblue;
    flex: 1;
    min-width: 250px;
}
.step h3 {
    color: var(--primary-color);
    font-size: 1.2em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

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
def preprocess_image(image_np, target_size=(320, 280)):
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
    else:
        if label == "Uncertain":
            st.warning(get_text("uncertain_diagnosis_warning"))
            st.write(f"{get_text('confidence_label')} {confidence * 100:.2f}%")
            st.info(get_text("uncertain_advice"))
        elif "Healthy" in label:
            st.balloons()
            st.success(get_text("healthy_success"))
            st.write(f"{get_text('confidence_label')} {confidence * 100:.2f}%")
            st.info(get_text("healthy_advice"))
        else:
            st.warning(get_text("potential_condition_warning").format(label))
            st.write(f"{get_text('confidence_label')} {confidence * 100:.2f}%")
            st.info(get_text("professional_advice_needed"))
            if label == "Pinguecula":
                st.markdown(get_text("pinguecula_advice"))
            elif label == "Pterygium Stage 1 (Trace-Mild)":
                st.markdown(get_text("pterygium1_advice"))
                st.warning(get_text("pterygium1_consult_doctor"))
            elif label == "Pterygium Stage 2 (Moderate-Severe)":
                st.markdown(get_text("pterygium2_advice"))
                st.error(get_text("pterygium2_consult_doctor"))
            elif label == "Red Eye(Conjunctivitis)":
                st.markdown(get_text("red_eye_advice"))
                st.info(get_text("red_eye_consult_doctor"))

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
        st.rerun()

# Header Section
st.markdown(f"<h1>👀 {get_text('app_header')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p>{get_text('app_subheader')}</p>", unsafe_allow_html=True)
st.markdown("---")

# Welcome and "How to use" Section
st.markdown(f"**{get_text('welcome_title')}** {get_text('welcome_message')}")
st.divider()

st.header(get_text("how_to_use_title"))
st.markdown(f"""
<div class="step-container">
    <div class="step">
        <h3>{get_text("step1_title")}</h3>
        <p>{get_text("step1_desc")}</p>
    </div>
    <div class="step">
        <h3>{get_text("step2_title")}</h3>
        <p>{get_text("step2_desc")}</p>
    </div>
    <div class="step">
        <h3>{get_text("step3_title")}</h3>
        <p>{get_text("step3_desc")}</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.info(f"**{get_text('disclaimer_title')}** {get_text('disclaimer_text')}")
st.divider()

st.subheader(get_text("start_scan_subheader"))
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

        st.markdown("### ✂️ Step 2: Crop Your Image")
        st.info("**Drag the box** to perfectly frame your eye. A precise crop leads to more accurate analysis.")
        cropped_img = st_cropper(
            img_pil,
            aspect_ratio=(320, 280),
            box_color='#FF4B4B', # A distinct color for the crop box
            key=cropper_key
        )
        if cropped_img:
            # Update the image for prediction ONLY if the cropper provides a valid output
            st.session_state.img_for_prediction = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2RGB) # Ensure RGB for further processing
            st.markdown("---")
            st.image(cropped_img, caption="✅ Cropped Image Ready for Analysis", use_container_width=True)
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
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {get_text('eye_detection_result_title')}")
                eye_label, eye_confidence = predict_eye_detection(st.session_state.img_for_prediction)
                display_prediction_result(eye_label, eye_confidence, is_eye_detection=True)
            if "No Eye Detected" in eye_label and eye_confidence > CONFIDENCE_THRESHOLD:
                col2.markdown(f"#### {get_text('eye_condition_analysis_title')}")
                col2.warning(get_text("cannot_analyze_condition"))
            else:
                with col2:
                    st.markdown(f"#### {get_text('eye_condition_analysis_title')}")
                    condition_label, condition_confidence = predict_eye_condition(st.session_state.img_for_prediction)
                    display_prediction_result(condition_label, condition_confidence)
else:
    st.info(get_text("initial_message"))

st.divider()
