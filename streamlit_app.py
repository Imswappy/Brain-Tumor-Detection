import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io, os, glob
import kagglehub

# ----------------------------------------------------
# 1. Load model (auto convert + upload)
# ----------------------------------------------------
@st.cache_resource
def load_model_cached():
    # 📥 Download the trained model
    path = kagglehub.model_download("noorsaeed/mri_brain_tumor_model/keras/default")
    files = os.listdir(path)
    st.write("✅ Files in KaggleHub path:", files)

    keras_model_path = os.path.join(path, "model.keras")
    h5_model_path = os.path.join(path, "model.h5")

    # Case 1: .keras already exists
    if os.path.exists(keras_model_path):
        st.success("✅ Using modern model.keras file")
        return tf.keras.models.load_model(keras_model_path, compile=False)

    # Case 2: Only .h5 exists → try to rebuild
    if os.path.exists(h5_model_path):
        st.warning("⚠️ Using legacy model.h5, converting to .keras...")

        try:
            legacy_model = tf.keras.models.load_model(h5_model_path, compile=False, safe_mode=False)
            legacy_model.save(keras_model_path, save_format="keras")
            st.success("✅ Converted model.h5 → model.keras")
            _upload_to_kagglehub(keras_model_path)
            return legacy_model

        except Exception as e:
            st.error(f"❌ Could not load model.h5 directly: {e}")
            st.info("🔄 Rebuilding architecture manually...")

            # --- MANUAL REBUILD using known architecture ---
            base = tf.keras.applications.VGG16(weights='imagenet',
                                               include_top=False,
                                               input_shape=(128, 128, 3))
            for layer in base.layers:
                layer.trainable = False

            model = tf.keras.Sequential([
                base,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
            ])

            # Save clean version
            model.save(keras_model_path, save_format="keras")
            st.success("✅ Model rebuilt & saved as model.keras")
            _upload_to_kagglehub(keras_model_path)
            return model

    st.error("❌ No model file found (neither model.keras nor model.h5).")
    return None

# ----------------------------------------------------
# 1b. Helper: Upload to KaggleHub
# ----------------------------------------------------
def _upload_to_kagglehub(model_path):
    try:
        st.info("📤 Uploading model.keras back to KaggleHub...")
        dataset = kagglehub.dataset_upload(
            model_path,
            dataset_slug="mri_brain_tumor_model",
            owner="noorsaeed",
            message="Added modern model.keras for better compatibility"
        )
        st.success("✅ model.keras uploaded to KaggleHub successfully!")
    except Exception as e:
        st.warning(f"⚠️ Upload skipped (likely no API key set): {e}")

# ----------------------------------------------------
# 2. Label mapping
# ----------------------------------------------------
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

# ----------------------------------------------------
# 3. Preprocess
# ----------------------------------------------------
def preprocess_image(pil_img: Image.Image, image_size: int = 128):
    pil_img = pil_img.convert('RGB').resize((image_size, image_size))
    arr = img_to_array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ----------------------------------------------------
# 4. Prediction
# ----------------------------------------------------
def predict(model, pil_img: Image.Image):
    batch = preprocess_image(pil_img, image_size=128)
    preds = model.predict(batch)
    idx = int(np.argmax(preds, axis=1)[0])
    conf = float(np.max(preds, axis=1)[0])
    label = CLASS_LABELS[idx]
    readable = "No Tumor" if label == 'notumor' else f"Tumor Detected: {label.capitalize()}"
    return readable, conf, preds[0].tolist()

# ----------------------------------------------------
# 5. Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Upload an MRI scan and classify brain tumors using a fine-tuned VGG16 model.")

# ✅ Load model
model = load_model_cached()

st.subheader("Upload an MRI image")
uploaded_img = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_img is not None and model is not None:
    pil_img = Image.open(io.BytesIO(uploaded_img.read()))
    st.image(pil_img, caption="Uploaded MRI", use_column_width=True)

    if st.button("Predict"):
        label, conf, raw = predict(model, pil_img)
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")
        with st.expander("🔎 Raw model outputs"):
            st.json({"class_order": CLASS_LABELS, "scores": raw})

st.caption("Preprocessing: images are resized to 128×128 and normalized to [0,1].")
