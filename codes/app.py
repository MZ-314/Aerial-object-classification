import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- 1. Configuration and Model Loading ---

# Updated path to point to models directory (relative to project root)
MODEL_PATH = os.path.join('..', 'models', 'best_mobilenetv2_model.keras')
CLASS_NAMES = ['bird', 'drone']
IMG_SIZE = (224, 224) 

@st.cache_resource
def load_model():
    """Loads the model from the specified path."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure '{MODEL_PATH}' exists.")
        return None

# --- 2. Preprocessing Function ---
def preprocess_image(image):
    """Resizes and preprocesses the PIL image for MobileNetV2."""
    
    # 2a. Resize
    image = image.resize(IMG_SIZE)
    
    # 2b. Convert to NumPy array and add batch dimension (1, H, W, C)
    image_array = np.array(image, dtype=np.float32)[np.newaxis, ...]
    
    # 2c. Apply MobileNetV2 preprocessing (scales pixels from [0, 255] to [-1, 1])
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    
    return processed_image

# --- 3. Prediction Function ---
def predict_image(model, processed_image):
    """Runs prediction on the processed image, returning all probabilities."""
    
    # model.predict returns probabilities
    predictions = model.predict(processed_image)
    
    # Returns the full prediction array and the top class info
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])
    
    return predictions, predicted_class, confidence

# --- 4. Main Streamlit Application ---

st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🦅",
    layout="centered"
)

st.title("🦅 Aerial Object Classification")
st.markdown("### Drone vs. Bird Detection System")
st.markdown("Upload an aerial image to classify it as either a **Drone** or a **Bird** using a deep learning model.")

# Load the model
model = load_model()

if model is not None:
    # Sidebar with project info
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Architecture:** MobileNetV2  
        **Input Size:** 224x224  
        **Classes:** Bird, Drone  
        **Framework:** TensorFlow/Keras
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload an aerial image (JPG, JPEG, or PNG)
        2. Click 'Classify Image' button
        3. View prediction and confidence scores
        """)
    
    # Create the file uploader widget
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            # Add a button to trigger prediction
            st.write("")
            st.write("")
            if st.button('🔍 Classify Image', use_container_width=True):
                with st.spinner('Analyzing image...'):
                    
                    # 4a. Preprocess
                    processed_image = preprocess_image(image)
                    
                    # 4b. Predict
                    predictions, predicted_class, confidence = predict_image(model, processed_image)
                    
                    # 4c. Display Results
                    if predicted_class.lower() == 'drone':
                        icon = "🚁"
                    else:
                        icon = "🐦"
                    
                    st.success(f"{icon} **Prediction: {predicted_class.upper()}**")
                    st.metric("Confidence", f"{confidence:.2%}")
        
        # Show detailed probabilities after prediction
        if st.session_state.get('show_details', False) or (uploaded_file and 'predictions' in locals()):
            st.markdown("---")
            st.markdown("#### 📈 Detailed Probabilities")
            
            if 'predictions' in locals():
                # Create columns for each class
                cols = st.columns(len(CLASS_NAMES))
                
                for idx, (col, class_name) in enumerate(zip(cols, CLASS_NAMES)):
                    with col:
                        prob = predictions[0][idx]
                        st.metric(
                            label=class_name.upper(),
                            value=f"{prob:.2%}",
                            delta=None
                        )

else:
    st.error("❌ Cannot run classification - model failed to load. Please check the model path.")
    st.info("Expected model location: `models/best_mobilenetv2_model.keras`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Aerial Object Classification System | Powered by TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True
)