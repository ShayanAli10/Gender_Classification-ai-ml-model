
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Gender Classification",
    page_icon="👤",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"G:\Gender_Classification\models\gender_classification_mobilenetv2.keras"
    )

model = load_model()

# Application title
st.title("Gender Classification")
st.write("Upload an image to predict Male or Female.")

# Image upload
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        # Preprocess image
        image = image.resize((224, 224))

        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        # Make prediction
        prediction = model.predict(image_array, verbose=0)[0][0]

        if prediction > 0.5:
            label = "Male"
            confidence = prediction * 100
        else:
            label = "Female"
            confidence = (1 - prediction) * 100

        # Display result
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(float(confidence / 100))
