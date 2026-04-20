import streamlit as st
from PIL import Image
import io
import time

from utils.model_utils import load_model, get_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Design System ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #09121a, #0d2b27, #153e34, #0b1a13);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    /* Make typography legible against dark bg natively without enforcing all spans */
    .stApp {
        color: #e2e8f0;
    }
    h1, h2, h3, p {
        color: #e2e8f0 !important;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #42d392, #647eff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 300;
        color: #94a3b8 !important;
        margin-bottom: 2rem;
    }

    /* Style the main layout columns natively as Glass Containers */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
    }
    [data-testid="column"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .result-card {
        text-align: center;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #42d392, #3bb17b) !important;
        border-radius: 10px;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #42d392 0%, #3bb17b 100%);
        color: #0b1a13;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 1.2rem 1.2rem !important;
        margin-top: 1rem !important;
        height: auto !important;
        border-radius: 12px;
        border: none;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(66, 211, 146, 0.5);
        background: linear-gradient(135deg, #4bf1a7 0%, #42d392 100%);
        color: #000000;
    }
    
    /* Ensure the file uploader is dark enough for white text */
    [data-testid="stFileUploader"] {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
         border: 1px dashed #42d392;
         background: rgba(0, 0, 0, 0.5);
    }
    
    /* Make text readable inside dropzone */
    .stFileUploaderDropzoneInstructions > div > span {
        color: #ffffff !important;
    }
    .stFileUploaderDropzoneInstructions > div > small {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- App Backend Initialization ---
@st.cache_resource
def initialize_ai_v2():
    model_path = "model/best_model_hybrid.pt"
    model, class_names = load_model(model_path)
    return model, class_names

model, class_names = initialize_ai_v2()

# --- Main Layout ---
st.markdown('<div class="hero-title">Smart Plant Disease Diagnosis </div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Advanced Artificial Intelligence for Precision Flora Diagnostics</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ AI Module Offline: Failed to load the deep learning environment. Check model folder.")
    st.stop()

# Let Streamlit handle the columns structurally; our CSS will style them natively as Glass Cards!
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<h3 style='margin-bottom: 0.2rem;'>UPLOAD IMAGE</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1rem;'>Provide a clear, well-lit image of the affected plant leaf for optimal AI accuracy.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Display image natively without inner columns preventing multiple nested CSS bugs
        st.image(image, use_container_width=True)

with col_right:
    st.markdown("<h3 style='margin-bottom: 0.2rem;'> ANALYSIS ENGINE</h3>", unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Waiting for telemetry data...\n\nUpload a leaf image on the left module to initiate scanning.")
    else:
        st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>Data loaded. Ready for forward pass tensor inference.</p>", unsafe_allow_html=True)
        analyze_button = st.button(" Initialize Diagnostics ")
        
        if analyze_button:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            
            progress_text = "Extracting texture features..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.015) 
                if percent_complete == 50:
                    my_bar.progress(percent_complete + 1, text="Synthesizing probabilistic tensors...")
                else:
                    my_bar.progress(percent_complete + 1)
            my_bar.empty()
            
            with st.spinner('Compiling final diagnostic report...'):
                try:
                    prediction, confidence = get_prediction(model, class_names, file_bytes)
                    
                    if prediction:
                        pred_clean = prediction.replace("_", " ").title()
                        
                        is_healthy = "Healthy" in pred_clean or "Background" in pred_clean
                        color = "#42d392" if is_healthy else "#ff4b4b"
                        bg_tint = "rgba(66, 211, 146, 0.08)" if is_healthy else "rgba(255, 75, 75, 0.08)"
                        border_tint = "rgba(66, 211, 146, 0.2)" if is_healthy else "rgba(255, 75, 75, 0.2)"
                        
                        st.markdown(f"""
<div class="result-card" style="background: {bg_tint}; border: 1px solid {border_tint};">
<h4 style="display: flex; justify-content: center; color: {color}; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 2px; font-size: 0.85rem; font-weight: 600;">Diagnostic Verdict</h4>
<h2 style="display: flex; justify-content: center; font-size: 2.2rem; font-weight: 700; color: #ffffff; margin-bottom: 1.5rem; line-height: 1.2;">{pred_clean}</h2>
<div style="display: flex; justify-content: center; margin-bottom: 0.4rem; font-size: 0.95rem; color: #cbd5e1; font-weight: 500;">
<span>Confidence: {confidence * 100:.2f}% </span>
</div>
</div>
""", unsafe_allow_html=True)
                        
                        st.progress(float(confidence))
                    else:
                        st.error("Neuro-inference failed. Retrain or try another image.")
                except Exception as e:
                    st.error(f"Inference Exception: {e}")
