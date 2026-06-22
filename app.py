import streamlit as st
import os

from modules.ocr_engine import ClinicalOCR
from modules.llm_extraction import (
    extract_symptoms_llm,
    extract_structured_data,
    extract_labs_from_ocr_llm
)
from modules.normalization import normalize_lab_keys
from modules.llm_chat import generate_chat_response
from main import run_clinical_engine
from modules.pdf_processor import PDFProcessor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Clinical AI Assistant",
    layout="wide",
    page_icon=""
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main { background-color: #0b0f14; }
.block-container { padding-top: 1.5rem; }

.card {
    background: linear-gradient(145deg, #161b22, #0e1117);
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #58a6ff;
}

.severity-high { color: #ff4d4d; font-weight: bold; }
.severity-medium { color: #ffa500; font-weight: bold; }
.severity-low { color: #4CAF50; font-weight: bold; }

input {
    background-color: #0e1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
input:focus {
    border: 1px solid #4CAF50 !important;
    box-shadow: 0 0 5px #4CAF50 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div style="margin-bottom:15px;">
    <h2 style="margin-bottom:0;">Clinical AI Assistant</h2>
    <p style="color:#8b949e; margin-top:4px;">
        AI-powered clinical decision support system
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# INIT
# =========================
ocr = ClinicalOCR()
pdf_processor = PDFProcessor()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "results" not in st.session_state:
    st.session_state.results = None

if "structured_data" not in st.session_state:
    st.session_state.structured_data = None

# =========================
# TOP CONTROL PANEL (ONLY INPUT)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Report",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed"
    )

with col2:
    symptom_text = st.text_input(
        "Describe symptoms",
        placeholder="Enter symptoms OR full clinical text...",
        label_visibility="collapsed"
    )

with col3:
    analyze_btn = st.button("Analyze", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# STATUS BAR
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1: st.success("OCR Engine Ready")
with s2: st.success("LLM Connected")
with s3: st.success("Clinical Model Loaded")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN LOGIC (TOP BUTTON)
# =========================
if analyze_btn:

    labs_display = {}
    labs_numeric = {}
    symptoms = []

    # FILE INPUT
    if uploaded_file:

        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_path.endswith(".pdf"):
            text = pdf_processor.extract_text(file_path)

            if not text.strip():
                images = pdf_processor.convert_to_images(file_path)
                text = ""
                for img in images:
                    text += ocr.extract(img) + "\n"
        else:
            text = ocr.extract(file_path)

        labs_raw = extract_labs_from_ocr_llm(text)
        labs_numeric, labs_display = normalize_lab_keys(labs_raw)

        if symptom_text:
            symptoms = extract_symptoms_llm(symptom_text)
        else:
            st.warning("Please enter symptoms with report")
            st.stop()

    # TEXT ONLY
    elif symptom_text:

        structured = extract_structured_data(symptom_text)

        if not structured:
            st.error("Could not extract data from text")
            st.stop()

        labs_display = structured.get("labs", {})
        labs_numeric = structured.get("labs_numeric", {})
        symptoms = structured.get("symptoms", [])

    else:
        st.warning("Please upload a file or enter text")
        st.stop()

    st.session_state.structured_data = {
        "labs": labs_display,
        "labs_numeric": labs_numeric,
        "symptoms": symptoms
    }

    st.session_state.results = run_clinical_engine(st.session_state.structured_data)

# =========================
# RESULTS
# =========================
if st.session_state.results:

    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Clinical Analysis</div>', unsafe_allow_html=True)

        for disease, data in st.session_state.results.items():

            st.markdown(f"### {disease.upper()}")

            if "probabilities" in data:
                st.write("**Detected Types:**", data["predicted_labels"])
            else:
                st.write("**Diagnosis:**", data.get("prediction"))

            severity = data.get("severity")

            if severity == "Severe":
                st.markdown(f'<div class="severity-high">Severity: {severity}</div>', unsafe_allow_html=True)
            elif severity == "Moderate":
                st.markdown(f'<div class="severity-medium">Severity: {severity}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="severity-low">Severity: {severity}</div>', unsafe_allow_html=True)

            st.write("**Recommended Action:**")
            st.write(data.get("recommended_action"))

            st.info(data.get("explanation"))

            shap_values = data.get("shap_values")

            st.divider()

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Assistant</div>', unsafe_allow_html=True)

        user_query = st.text_input("Ask about your report...")

        if user_query:
            response = generate_chat_response(
                user_query,
                st.session_state.results,
                st.session_state.chat_history
            )

            st.session_state.chat_history.append(("User", user_query))
            st.session_state.chat_history.append(("Bot", response))

        for role, msg in st.session_state.chat_history[::-1]:
            st.chat_message("user" if role == "User" else "assistant").write(msg)

        st.markdown('</div>', unsafe_allow_html=True)

