import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

MODEL_PATH = "brain_tumor_cnn.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE = (150, 150)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_artifacts()

st.title("Brain Tumor MRI Classification")
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    img_resized = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)
    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    st.subheader(f"Prediction: {class_names[pred_idx]}")
    st.write(f"Confidence: **{float(probs[pred_idx]) * 100:.2f}%**")
