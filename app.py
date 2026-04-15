import streamlit as st
from PIL import Image
import io

# We import from the internal utils folder rather than making web HTTP requests
from utils.model_utils import load_model, get_prediction

@st.cache_resource
def initialize_ai():
    """
    Using st.cache_resource prevents Streamlit from reloading 
    the heavy PyTorch dictionary on every single user interaction.
    """
    model_path = "model/best_model_hybrid.pt"
    # Load model and class names locally
    model, class_names = load_model(model_path)
    return model, class_names

st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #f7f9fc; }
    .stProgress .st-bo { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Smart Plant Disease Diagnosis")
st.markdown("**Upload a leaf image** to determine its health and potential diseases.")

# Execute PyTorch Cache Initializer
model, class_names = initialize_ai()

if model is None:
    st.error("Failed to boot the deep learning environment. Is the .pt file in the model folder?")
    st.stop()

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    
    st.markdown("---")
    
    if st.button("Diagnosis Image"):
        # Reset file pointer to the beginning before reading for payload
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
            
        with st.spinner('AI is analyzing the cellular structure...'):
            try:
                # Bypass FastAPI completely and perform CPU/GPU Matrix Inference locally!
                prediction, confidence = get_prediction(model, class_names, file_bytes)
                
                if prediction:
                    st.success("Diagnosis Complete!")
                    st.subheader("Result: " + prediction.replace("_", " ").title())
                    
                    # Show confidence visually
                    st.write(f"Confidence: **{confidence * 100:.2f}%**")
                    st.progress(float(confidence))
                else:
                    st.error("The model failed to predict the class.")
            except Exception as e:
                st.error(f"Internal PyTorch Inference Error: {e}")
