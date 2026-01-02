# Deepfake vs Real Image Detection using Vision Transformer (ViT)

## 📌 Project Overview

This project delivers a **complete, end-to-end deepfake image detection system** built using a **fine-tuned Vision Transformer (ViT)**. It covers the full ML lifecycle — from data preparation and model training to evaluation and **production-grade deployment** as a web application.

The system is designed to be:
- **Highly accurate**
- **Scalable**
- **Reproducible**
- **User-friendly**

A live demo is publicly hosted, allowing users to upload images and instantly verify whether they are **Real or Deepfake**.

---

## 🎯 Core Objectives

- **Build a High-Accuracy Deepfake Detector**  
  Train a robust deep learning model capable of distinguishing real vs deepfake images with exceptional precision and recall.

- **Handle Dataset Imbalance**  
  Address class imbalance using **manual random oversampling** to ensure fair learning.

- **Enable Scalable Deployment**  
  Deploy the trained model using **Docker + Flask**, hosted on **Hugging Face Spaces**.

- **Design an Intuitive UI**  
  Provide a clean, responsive interface with animated predictions and confidence scores.

- **Ensure Full Reproducibility**  
  Maintain a clean project structure and host the trained model publicly.

---

## 📂 Dataset & Preprocessing

- **Dataset Source:**  
  Kaggle – https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

- **Original Size:**  
  1,000 images (500 Real, 500 Fake)

- **After Oversampling:**  
  **76,161 images**
  - 38,080 Real
  - 38,081 Fake

### Preprocessing Details
- Image size normalized to **224 × 224**
- **Training Augmentation:**
  - Random rotations
  - Sharpness variations
- **Test Data:**
  - Only resizing + normalization (no augmentation bias)

---

## 🧠 Methodology

### 1️⃣ Data Pipeline
- Dataset loaded into a Pandas DataFrame
- Manual oversampling applied
- Labels encoded:
  - `0 → Real`
  - `1 → Fake`
- **Stratified split:**  
  - 60% Training  
  - 40% Testing
- Lazy transformations applied using `ViTImageProcessor`

---

### 2️⃣ Model Architecture & Training

- **Base Model:** Vision Transformer (ViT)
- **Pretrained Weights:**  
  `dima806/deepfake_vs_real_image_detection`
- **Framework:** Hugging Face `transformers`
- **Training Setup:**
  - Epochs: 2
  - Batch size: 32
  - Learning rate: `1e-6`
  - Weight decay: `0.02`
- **Training Time:** ~2 hours 6 minutes

Transfer learning enables the model to reuse learned visual representations and specialize in deepfake detection efficiently.

---

### 3️⃣ Evaluation Results

Evaluated on **76,161 test images**.

| Metric | Real | Fake | Overall |
|------|------|------|--------|
| **Precision** | 99.25% | 99.22% | 99.24% |
| **Recall** | 99.22% | 99.25% | 99.24% |
| **F1-Score** | 99.24% | 99.24% | 99.24% |
| **Accuracy** | — | — | **99.24%** |

- **Loss:** 0.0229  
- **Misclassifications:** ~582 images  
- **Trainable Parameters:** ~85K  

---

## 🚀 Deployment

### Model Artifacts
- `model.safetensors`
- `config.json`
- `preprocessor_config.json`

### Deployment Stack
- **Backend:** Flask REST API
- **Frontend:** HTML, CSS, JavaScript
- **Containerization:** Docker
- **Web Server:** Gunicorn
- **Hosting:** Hugging Face Spaces (Port 7860)
- **Model Hosting:** Hugging Face Hub

Predictions are returned in **5–10 seconds**.

---

## 🎨 User Interface & Experience

### Features
- Dark gradient modern UI
- Drag-and-drop image upload
- Live image preview with hover zoom
- Animated loading spinner
- Smooth fade-in prediction results
- Confidence score display
- Fully responsive (mobile + desktop)

---

## 🏗 System Architecture

### Model Architecture
- Vision Transformer (ViT)
- Input: 224 × 224 RGB image
- Output: Binary classification (Real / Fake)
- Fine-tuned via transfer learning

### Application Architecture
- Flask REST API
- Frontend served via Flask
- Dockerized environment
- Deployed on Hugging Face Spaces

---

## 🧾 Technology Stack

- **Language:** Python 3.10
- **ML:** PyTorch, Hugging Face Transformers
- **Web:** Flask
- **Container:** Docker
- **Hosting:** Hugging Face Spaces & Hub
- **Data:** Pandas, NumPy
- **Images:** Pillow
- **Tools:** Git, VS Code, Jupyter Notebook

---

## ✅ Conclusion

This project showcases how **Vision Transformers combined with transfer learning** can achieve state-of-the-art performance in deepfake detection. With **99.24% accuracy**, public deployment, and clean engineering practices, it stands as a strong **full-stack AI project** suitable for research, internships, and production demos.

---

## 🙏 Acknowledgments

- Kaggle for the dataset  
- Hugging Face for `transformers`, Hub, and Spaces  
- Open-source contributors behind Flask, Docker, and PyTorch


