import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper
from PIL import Image
import os

# --- Environment Setup (No GPU support for Streamlit) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Constants ---
FIRST_MODEL_PATH = "EyeDetect.keras"
FIRST_CLASS_NAMES = ["Eye Detected", "No Eye Detected"]
SEC_MODEL_PATH = "EyeAnalysis.keras"
SEC_CLASS_NAMES = ["Healthy", "Pinguecula", "Pterygium Stage 1 (Trace-Mild)", "Pterygium Stage 2 (Moderate-Severe)", "Red Eye(Conjunctivitis)"]
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.10

# --- Translation Data ---
TEXTS = {
    "en": {
        "page_title": "Ocular scan ",
        "app_header": "👀 OcuScanAI",
        "app_subheader": "Your intelligent assistant for preliminary eye health checks (Healthy, Pinguecula, Pterygium, Red Eye).",
        "welcome_title": "Welcome!",
        "welcome_message": "Let AI help you quickly screen for common eye conditions like Pinguecula, Pterygium (both early and advanced stages), Red Eye, or just check if your eyes appear healthy.",
        "how_to_use_title": "How to use",
        "step1_title": "📸 Input an Image:",
        "step1_desc": "Take or upload a clear photo of your eye (just make sure we can see your full eye like 👁️) so we can help check it better!",
        "step2_title": "✂️ Crop your image:",
        "step2_desc": "Drag the box to perfectly frame your eye. A precise crop helps our AI analyze it more accurately.",
        "step3_title": "🔬 Get the result:",
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
        "failed_to_load_first_model": "System error occurred.(I)",
        "loading_sec_model": "🧠 Loading AI model for eye condition analysis...",
        "failed_to_load_sec_model": "System error occurred.(II)",
        "analyzing_image": "Analyzing image... Please wait. This may take a few moments.",
        "language_selector_label": "Select Language",
        "sidebar_settings_title": "Settings"
    },
    "th": {
        "page_title": "เครื่องมือตรวจสภาพดวงตา",
        "app_header": "👀 OcuScanAI",
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
        "crop_step_info": "**ลากกรอบ**ครอบให้พอดีกับดวงตา",
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
        ตาแดงอาจเกิดได้จากหลายสาเหตุ เช่น การระคายเคือง, ภูมิแพ้, การติดเชื้อ หรือภาวะทางการแพทย์อื่น ๆ แม้ว่ามักจะไม่เป็นอันตราย แต่หากตาแดงมีอาการ
