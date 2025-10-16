import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="🌸 Flower Classifier",
    page_icon="🌺",
    layout="centered"
)

# ----------------------------
# Hide Streamlit Warnings
# ----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow info logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for Render (no CUDA drivers)

# ----------------------------
# Load Model (once)
# ----------------------------
@st.cache_resource
def load_model():
    # Replace 'flower_model.h5' with your actual model path
    model = tf.keras.models.load_model("my_model.keras")
    return model

model = load_model()

# ----------------------------
# Class Labels (adjust to your dataset)
# ----------------------------
CLASS_NAMES = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

# ----------------------------
# Streamlit Title & Description
# ----------------------------
st.title("🌼 Flower Image Classifier")
st.markdown(
    "Upload a photo of a flower, and this app will predict which type it is using a TensorFlow model."
)

# ----------------------------
# File Uploader (with unique key)
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"],
    key="flower_upload"  # ✅ unique key avoids duplicate ID errors
)

# ----------------------------
# Prediction Logic
# ----------------------------
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((180, 180))  # Adjust to your model's input shape
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display result
    st.subheader("🌸 Prediction Result:")
    st.write(f"**Predicted flower:** {CLASS_NAMES[np.argmax(score)]}")
    st.write(f"**Confidence:** {100 * np.max(score):.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Powered by Streamlit + TensorFlow • © 2025 Flower Vision AI")


