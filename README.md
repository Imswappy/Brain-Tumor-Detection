# 🧠 Brain Tumor MRI Classification using Deep Learning

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
\$
h_{i,j}^{(k)} = \sigma \left( \sum_{m,n} W_{m,n}^{(k)} \cdot x_{i+m, j+n} + b^{(k)} \right)
\$

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
  \$
  h = ReLU(Wx + b)
  \$
- **Dropout(0.3, 0.2):** Randomly zeroes units during training, reducing overfitting.  
- **Output Layer:** Softmax classifier for 4 classes:  
  \$
  P(y=i|x) = \frac{\exp(z_i)}{\sum_{j=1}^{4} \exp(z_j)}
  \$  

---

## 📒 Notebook (`brain_tumour_detection_vs_code.ipynb`)

Sections:
1. **Dataset loading** with KaggleHub  
2. **EDA**: Class distributions, sample MRI visualization  
3. **Preprocessing**: Resize, normalize, split  
4. **Model**: VGG16 base + custom classifier  
5. **Training**: Adam optimizer, categorical crossentropy loss  
   \$
   L = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})
   \$
6. **Evaluation**: Accuracy, confusion matrix, classification report  
7. **Saving model** in `.h5` format  

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

---

## 🙌 Acknowledgements

- **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- **Pre-trained Model**: [MRI Brain Tumor Model](https://www.kaggle.com/models/noorsaeed/mri_brain_tumor_model)  
- **Base Architecture**: [VGG16](https://arxiv.org/abs/1409.1556)  

---
