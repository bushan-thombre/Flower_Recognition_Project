import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(page_title="🌼 Flower Classification", layout="centered")
st.title("🌸 Flower Recognition App")
st.write("Upload a flower image and let the model predict the type!")

# -------------------------------
# 2. Load Trained Model
# -------------------------------
@st.cache_resource
def load_flower_model():
    model = load_model("import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(page_title="🌼 Flower Classification", layout="centered")
st.title("🌸 Flower Recognition App")
st.write("Upload a flower image and let the model predict the type!")

# -------------------------------
# 2. Load Trained Model
# -------------------------------
@st.cache_resource
def load_flower_model():
    model = load_model("import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(page_title="🌼 Flower Classification", layout="centered")
st.title("🌸 Flower Recognition App")
st.write("Upload a flower image and let the model predict the type!")

# -------------------------------
# 2. Load Trained Model
# -------------------------------
@st.cache_resource
def load_flower_model():
    model = load_model("my_model.keras")  # your saved Keras model
    return model

model = load_flower_model()

# -------------------------------
# 3. Define Class Names
# -------------------------------
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# -------------------------------
# 4. Prediction Function
# -------------------------------
def predict_image(image_file, model):
    # Load the image with correct target size
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # <-- resize to match your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# -------------------------------
# 5. Streamlit File Uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    
    # Predict
    predicted_class, confidence = predict_image(uploaded_file, model)
    
    # Display Results
    st.success(f"🌼 **Predicted Flower:** {predicted_class}")
    st.info(f"💪 **Confidence:** {confidence:.2f}%")
")  # your saved Keras model
    return model

model = load_flower_model()

# -------------------------------
# 3. Define Class Names
# -------------------------------
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# -------------------------------
# 4. Prediction Function
# -------------------------------
def predict_image(image_file, model):
    # Load the image with correct target size
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # <-- resize to match your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# -------------------------------
# 5. Streamlit File Uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    
    # Predict
    predicted_class, confidence = predict_image(uploaded_file, model)
    
    # Display Results
    st.success(f"🌼 **Predicted Flower:** {predicted_class}")
    st.info(f"💪 **Confidence:** {confidence:.2f}%")
")  # your saved Keras model
    return model

model = load_flower_model()

# -------------------------------
# 3. Define Class Names
# -------------------------------
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# -------------------------------
# 4. Prediction Function
# -------------------------------
def predict_image(image_file, model):
    # Load the image with correct target size
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # <-- resize to match your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# -------------------------------
# 5. Streamlit File Uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    
    # Predict
    predicted_class, confidence = predict_image(uploaded_file, model)
    
    # Display Results
    st.success(f"🌼 **Predicted Flower:** {predicted_class}")
    st.info(f"💪 **Confidence:** {confidence:.2f}%")
