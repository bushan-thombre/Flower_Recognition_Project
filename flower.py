import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load your saved Keras model
model = load_model("my_model.keras")

# Example class names — replace these with your actual flower categories
class_names = list_ = ['Daisy','Danelion','Rose','sunflower', 'tulip']
# Streamlit page config
st.set_page_config(page_title="🌸 Flower Recognition System", page_icon="🌼")

st.title("🌸 Flower Recognition System")
st.write("Upload an image of a flower ")

# File uploader
uploaded_file = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img = image.resize((224, 224))  # match your model input size
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict using the model
    preds = model.predict(x)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show result
    st.success(f"🌼 Predicted: **{predicted_class.capitalize()}** ({confidence:.2f}% confidence)")

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load your saved Keras model
model = load_model("my_model.keras")

# Example class names — replace these with your actual flower categories
class_names = list_ = ['Daisy','Danelion','Rose','sunflower', 'tulip']
# Streamlit page config
st.set_page_config(page_title="🌸 Flower Recognition System", page_icon="🌼")

st.title("🌸 Flower Recognition System")
st.write("Upload an image of a flower ")

# File uploader
uploaded_file = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img = image.resize((224, 224))  # match your model input size
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict using the model
    preds = model.predict(x)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show result
    st.success(f"🌼 Predicted: **{predicted_class.capitalize()}** ({confidence:.2f}% confidence)")


