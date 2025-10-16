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
# Suppress TensorFlow warnings
# ----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----------------------------
# Load Model (once)
# ----------------------------
@st.cache_resource
def load_model():
    # Replace 'flower_model.h5' with your actual model
    model = tf.keras.models.load_model("my_model.keras")
    return model

model = load_model()

# ----------------------------
# Class labels
# ----------------------------
CLASS_NAMES = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"],
    key="flower_upload"
)

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(img, model):
    """
    Predict a flower class from a PIL image
    without changing its original size.
    """
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension

    # Check if model expects flat input
    input_shape = model.input_shape
    if len(input_shape) == 2:  # Dense expects flat vector
        img_array = tf.reshape(img_array, [1, -1])

    # Predict
    predictions = model.predict(img_array)
    return tf.nn.softmax(predictions[0])

# ----------------------------
# Run prediction
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    score = predict_image(image, model)

    st.subheader("🌸 Prediction Result:")
    st.write(f"**Predicted flower:** {CLASS_NAMES[np.argmax(score)]}")
    st.write(f"**Confidence:** {100 * np.max(score):.2f}%")

st.markdown("---")
st.caption("Powered by Streamlit + TensorFlow • © 2025 Flower Vision AI")

