# 🧠 NeuroVision: Brain Tumor MRI Classification with Deep Learning

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Keras](https://img.shields.io/badge/Framework-Keras-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Python](https://img.shields.io/badge/Python-3.10+-green)

---

## 📌 Project Overview

This project focuses on **automated brain tumor detection** using **MRI images**.  
We apply **transfer learning with VGG16**, a convolutional neural network (CNN) pre-trained on the ImageNet dataset, and fine-tune it to classify brain MRIs into:

- **Meningioma**
- **Glioma**
- **Pituitary Tumor**
- **No Tumor**

The trained model is deployed via a **Streamlit app**, where users can upload MRI images and get predictions with confidence scores.

---

## 📂 Project Structure

```
brain-tumor-mri/
│
├── brain_tumour_detection.ipynb   # Training & evaluation (KaggleHub dataset)
├── streamlit_app.py                       # Deployment UI with Streamlit
├── requirements.txt                       # Dependencies
└── README.md                              # Documentation
```

---

## 📊 Dataset

Dataset is fetched from KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

Structure:

```
Training/
   ├── glioma/
   ├── meningioma/
   ├── pituitary/
   └── notumor/
Testing/
   ├── glioma/
   ├── meningioma/
   ├── pituitary/
   └── notumor/
```

- **Training samples:** ~2870  
- **Testing samples:** ~394  
- Each subdirectory corresponds to a tumor class.

---

## 🏗️ Model Architecture (Transfer Learning with VGG16)

We use **VGG16**, a pre-trained CNN on ImageNet, as a **feature extractor** and fine-tune its top layers.  

### 🔹 Step 1: Base Model
```python
from tensorflow.keras.applications import VGG16

base_model = VGG16(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)
```
- **Input size:** 128 × 128 × 3 (resized MRI images).  
- `include_top=False`: removes VGG16’s fully connected (FC) head.  
- `weights="imagenet"`: initializes weights from ImageNet (~1.4M images, 1000 classes).  

Mathematically, each convolutional layer applies:
<img width="491" height="139" alt="image" src="https://github.com/user-attachments/assets/4b3d794f-99f0-4c6c-9ba4-eda7f9457edc" />

where:
- W^{(k)}: convolutional kernel for feature map k  
- x: input patch  
- σ: activation (ReLU in VGG16)  

### 🔹 Step 2: Freezing and Fine-Tuning
```python
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze top 3 layers
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True
```
- Lower layers retain **general features** (edges, textures).  
- Top 3 layers fine-tuned to capture **domain-specific features** of MRI tumors.  


### 🔹 Step 3: Custom Classification Head
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense

model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(unique_labels), activation='softmax')
])
```

- **Flatten:** Reshapes VGG16 feature maps (4 × 4 × 512) → (8192,).  
- **Dense(128, relu):** Fully connected layer learns non-linear combinations of features.  
 <img width="270" height="52" alt="image" src="https://github.com/user-attachments/assets/2cf5d1ab-c611-4e93-8514-b6d4f7310f0d" />

- **Dropout(0.3, 0.2):** Randomly zeroes units during training, reducing overfitting.  
- **Output Layer:** Softmax classifier for 4 classes:  
<img width="365" height="120" alt="image" src="https://github.com/user-attachments/assets/97e12f86-7221-4f25-a15f-b5f7f0147a2a" />
 

---

## 📒 Notebook (`brain_tumour_detection_vs_code.ipynb`)

Sections:
1. **Dataset loading** with KaggleHub  
2. **EDA**: Class distributions, sample MRI visualization
   <img width="1484" height="678" alt="image" src="https://github.com/user-attachments/assets/7c03c569-4821-4e08-a5a9-69ce111462e6" />
3. **Preprocessing**: Resize, normalize, split  
4. **Model**: VGG16 base + custom classifier  
5. **Training**: Adam optimizer, categorical crossentropy loss  
<img width="340" height="107" alt="image" src="https://github.com/user-attachments/assets/dcca8f71-eb06-4c47-a664-eabf2bd20f71" />
<img width="820" height="402" alt="image" src="https://github.com/user-attachments/assets/361f3639-a715-4571-828f-d570024132d2" />


6. **Evaluation**: Accuracy, confusion matrix, classification report
   <img width="662" height="556" alt="image" src="https://github.com/user-attachments/assets/880a05e0-ead1-45e7-a58c-15fb24a928dd" />
   <img width="854" height="710" alt="image" src="https://github.com/user-attachments/assets/db0363a0-ffbd-4ea4-8cf3-684bc7572cd8" />


8. **Saving model** in `.h5` format  

---

## 🧮 Training Statistics

- **Optimizer:** Adam (β₁=0.9, β₂=0.999)  
- **Learning Rate:** 1e-4  
- **Batch Size:** 32  
- **Loss:** Categorical Crossentropy  
- **Regularization:** Dropout (0.3 & 0.2)  

**Performance:**
- Training Accuracy: ~95%  
- Test Accuracy: ~92%  
- Balanced F1 scores across all classes  

---

## 🎨 Streamlit App (`streamlit_app.py`)

Interactive deployment UI:
- Upload MRI image (`png/jpg/jpeg`)  
- Preprocessing: resize 128 × 128, normalize [0,1]  
- Prediction from VGG16-based model  
- Displays:
  - Label
  - Confidence
  - Uploaded image
  - Raw probabilities

Run:
```bash
streamlit run streamlit_app.py
```

---

## ⚙️ Installation (VS Code)

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# 2. Install deps
pip install -r requirements.txt

# 3. Run app
streamlit run streamlit_app.py
```

---

## 🚀 Deployment

- Local: http://localhost:8501  
- LAN: http://<your-ip>:8501  
- Global: deploy on **Streamlit Cloud**
- <img width="464" height="927" alt="image" src="https://github.com/user-attachments/assets/794eaa0d-eb8b-49ae-9ef6-356c2daf7f62" /> <img width="457" height="929" alt="image" src="https://github.com/user-attachments/assets/16e4bfcb-66cb-4756-9237-a3e325e55509" />
<img width="464" height="935" alt="image" src="https://github.com/user-attachments/assets/87bb2bbc-a614-4fc6-885e-5961376ae2b3" /> <img width="470" height="930" alt="image" src="https://github.com/user-attachments/assets/31f3d85e-4915-402e-9a65-696660237b81" />






---

## 🙌 Acknowledgements

- **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- **Pre-trained Model**: [MRI Brain Tumor Model](https://www.kaggle.com/models/noorsaeed/mri_brain_tumor_model)  
- **Base Architecture**: [VGG16](https://arxiv.org/abs/1409.1556)  

---
