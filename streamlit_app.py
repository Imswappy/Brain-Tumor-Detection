import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import io, os
import kagglehub  

# using Streamlit's cache so the model loads only once
@st.cache_resource
def load_model_cached():
    # ✅ Download the trained model from KaggleHub
    path = kagglehub.model_download("noorsaeed/mri_brain_tumor_model/keras/default")
    model_path = os.path.join(path, "model.h5")   # KaggleHub returns folder, model is inside
    return load_model(model_path)

# keep the label order consistent with training
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

def preprocess_image(pil_img: Image.Image, image_size: int = 128):
    pil_img = pil_img.convert('RGB').resize((image_size, image_size))
    arr = img_to_array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(model, pil_img: Image.Image):
    batch = preprocess_image(pil_img, image_size=128)
    preds = model.predict(batch)
    idx = int(np.argmax(preds, axis=1)[0])
    conf = float(np.max(preds, axis=1)[0])
    label = CLASS_LABELS[idx]
    readable = "No Tumor" if label == 'notumor' else f"Tumor: {label}"
    return readable, conf, preds[0].tolist()

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")

# ✅ Load model automatically via KaggleHub
model = load_model_cached()

st.subheader("Upload an MRI image")
uploaded_img = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_img is not None:
    pil_img = Image.open(io.BytesIO(uploaded_img.read()))
    st.image(pil_img, caption="Uploaded MRI", use_column_width=True)

    if st.button("Predict"):
        label, conf, raw = predict(model, pil_img)
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")
        with st.expander("Raw model outputs"):
            st.json({"class_order": CLASS_LABELS, "scores": raw})


st.caption('I keep preprocessing consistent with training (resize to 128×128, scale to [0,1]).')
