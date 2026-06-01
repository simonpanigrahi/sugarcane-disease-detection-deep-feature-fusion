# Sugarcane Leaf Disease Detection using Deep Feature Fusion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![Accuracy](https://img.shields.io/badge/Accuracy-92.36%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A hybrid deep learning pipeline that fuses MobileNetV2 deep embeddings with handcrafted GLCM texture and HSV colour features, achieving **92.36% accuracy** across 12 disease classes on a 12,000-image dataset.

---

## Pipeline Architecture

```
Input Image ──► MobileNetV2 (ImageNet weights)
                     │
                     ▼
             1,280-dim deep embeddings
                     │
                     ├──────────────────────────────┐
                     │                              │
Input Image ──► GLCM Texture (8) +         Feature Concatenation
                HSV Histogram (3)          (1,280 + 11 = 1,291-dim)
                     │                              │
              11-dim handcrafted                    ▼
               feature vector             XGBoost Classifier
                                                    │
                                                    ▼
                                         Predicted Disease Class
```

The core insight is that deep CNN features capture high-level semantic patterns while GLCM and HSV features encode local texture and colour distribution — both of which are diagnostically relevant for plant pathology.

---

## Dataset

| Attribute        | Detail                                   |
|------------------|------------------------------------------|
| Total Images     | ~12,000                                  |
| Disease Classes  | 12 (balanced)                            |
| Image Source     | PlantVillage / field-collected dataset   |
| Preprocessing    | Resize to 224×224, normalise to [0,1]    |
| Augmentation     | Rotation, flip, zoom, brightness shift   |

---

## Results

| Metric                   | Score     |
|--------------------------|-----------|
| Overall Accuracy         | **92.36%** |
| Macro-avg Precision      | ~0.93     |
| Macro-avg Recall         | ~0.92     |
| Macro-avg F1-Score       | ~0.92     |

Full per-class precision, recall, F1, and confusion matrix are available in `notebooks/04_model_training_evaluation.ipynb`.

---

## Repository Structure

```
sugarcane-disease-detection-deep-feature-fusion/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb        # Dataset loading, class distribution, train/val/test split
│   ├── 02_augmentation.ipynb            # Image augmentation pipeline and visualisation
│   ├── 03_feature_extraction.ipynb      # MobileNetV2 embeddings + GLCM/HSV feature extraction
│   └── 04_model_training_evaluation.ipynb  # XGBoost training, evaluation metrics, confusion matrix
│
├── src/
│   ├── feature_extraction.py            # GLCM texture + HSV colour histogram extraction
│   ├── model.py                         # MobileNetV2 backbone + XGBoost training logic
│   └── utils.py                         # Data loading, preprocessing, visualisation helpers
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/simonpanigrahi/sugarcane-disease-detection-deep-feature-fusion.git
cd sugarcane-disease-detection-deep-feature-fusion
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### 2. Feature Extraction
```bash
jupyter notebook notebooks/03_feature_extraction.ipynb
```
This produces a `.npy` feature matrix by fusing MobileNetV2 embeddings (1,280-dim) with GLCM+HSV features (11-dim).

### 3. Model Training & Evaluation
```bash
jupyter notebook notebooks/04_model_training_evaluation.ipynb
```
Trains an XGBoost classifier on the 1,291-dim fused feature vectors and outputs accuracy, per-class F1, and confusion matrix.

---

## Tech Stack

| Layer              | Tools / Libraries                                  |
|--------------------|----------------------------------------------------|
| Deep Feature Model | TensorFlow/Keras, MobileNetV2 (ImageNet pre-trained) |
| Handcrafted Features | OpenCV (GLCM, HSV histograms)                   |
| Classifier         | XGBoost                                            |
| Data Handling      | NumPy, Pandas                                      |
| Visualisation      | Matplotlib, Seaborn                                |
| Deployment (demo)  | Streamlit                                          |
| Environment        | Google Colab, Jupyter Notebook                     |

---

## Research Context

Plant disease causes significant yield losses in sugarcane cultivation across tropical and subtropical regions. Automated detection from leaf images enables early intervention without requiring domain expertise. This work explores whether fusing pre-trained CNN representations with domain-relevant handcrafted features (texture, colour) improves classification robustness over CNN-only or handcrafted-only approaches.

---

## Author

**Simon Kenny Panigrahi**
B.Tech Information Technology, VSSUT Burla (2022–2026)
Former ML Intern, C-DOT — Centre for Development of Telematics, Government of India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-simonkp-blue?logo=linkedin)](https://linkedin.com/in/simonkp)
[![GitHub](https://img.shields.io/badge/GitHub-simonpanigrahi-black?logo=github)](https://github.com/simonpanigrahi)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
