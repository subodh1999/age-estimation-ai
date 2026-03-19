# Real-Time Age Estimation AI with Fairness Auditing & Explainability

A deep learning system that classifies facial images into 5 age brackets in real-time, with cross-model fairness auditing, EU AI Act compliance documentation, and Grad-CAM/ViT explainability — deployed as a live Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)
![Coverage](https://img.shields.io/badge/Test_Coverage-99%25-brightgreen)

**Live Demo:** [age-estimation-ai.streamlit.app](https://age-estimation-ai.streamlit.app/)

---

## Problem Statement

Large, multi-story retail environments face navigation friction — customers struggle to find relevant sections (Infants, Kids, Seniors) across floors. This system estimates a customer's age bracket via store cameras so staff can proactively assist, classifying faces into **5 functional age brackets**:

| Bracket | Age Range |
|---------|-----------|
| Infant | 0 – 7 |
| Young | 8 – 17 |
| Adult | 18 – 34 |
| Middle-aged | 35 – 59 |
| Senior | 60+ |

**Key constraint:** The system must remain **gender-neutral** and **race-agnostic** — it must not use demographic features as classification proxies.

---

## Results

### Model Comparison (Validation Set: 21,398 samples)

| Model | Best Val Accuracy | Final Train Acc | Val Loss | Overfitting Gap | Parameters |
|-------|------------------|----------------|----------|-----------------|------------|
| CNN (Baseline) | 83.39% | — | — | — | Custom |
| VGG16 | ~85% | — | — | — | ~138M |
| ResNet-50 (unfrozen) | 98.16% | 99.67% | 0.1445 | 1.58% | ~25.6M |
| ViT-B/16 (unfrozen) | 97.45% | 97.88% | 0.1118 | 0.43% | ~86.6M |
| **MobileNetV2** | **98.29%** | **99.79%** | **0.1075** | **1.50%** | **~3.5M** |

MobileNetV2 was selected for deployment: highest accuracy, lowest loss, smallest fairness gap, and fewest parameters — optimized for real-time inference on edge devices.

### Cross-Model Fairness Audit

| Model | Race 0 (White) | Race 1 (Black) | Race 2 (Asian) | Race 3 (Indian) | Race 4 (Other) | Max Gap | Chi² | P-Value |
|-------|---------------|----------------|----------------|-----------------|----------------|---------|------|---------|
| ResNet-50 | 97.43% | 94.79% | 98.07% | 97.43% | 97.68% | 3.28% | 49.79 | < 0.001 |
| ViT-B/16 | 96.68% | 92.04% | 97.46% | 95.94% | 97.25% | 5.42% | 105.56 | < 0.001 |
| **MobileNetV2** | **97.45%** | **95.49%** | **98.17%** | **97.56%** | **98.54%** | **3.05%** | **39.24** | **< 0.001** |

The racial accuracy gap was reduced from **8.44%** (Week 2 CNN) to **3.05%** (MobileNetV2). Race 1 (Black) consistently shows the lowest accuracy across all models, indicating a persistent representation challenge.

### Explainability (XAI)

- **Grad-CAM** heatmaps on ResNet-50 and MobileNetV2 confirm models focus on age-relevant facial features (wrinkles, bone structure, hair) rather than racial proxies
- **ViT attention maps** show global patch-level reasoning, with the CLS token attending to structurally informative face regions
- Both satisfy **EU AI Act Article 13 (Transparency)** documentation requirements

---

## Project Structure

```
age-estimation-ai/
├── README.md
├── requirements.txt
├── Setup_Guide.md                          # Data download & path config
├── app.py                                  # Streamlit webcam app
├── packages.txt
├── best_mobilenetv2.pth                    # Trained MobileNetV2 weights
├── Consolidated_Project_Notebook.ipynb     # Full training, analysis & report
├── .gitignore
```

---

## Development Phases

### Week 1 — Data Analysis & Regulatory Groundwork
- Audited FairFace dataset (124,461 images); only 5,458 (4.4%) had usable labels
- Identified demographic gaps: only 3 racial categories represented, age skewed to 28–42 range
- Selected UTKFace (~67K images) and MORPH-2 (~40K images) as supplementary training data
- Classified project as **EU AI Act High-Risk** (biometric categorization in commercial setting)
- Mapped regulatory requirements: EU AI Act, GDPR Art. 9, Ethics Art. 52, IEEE 7001

### Week 2 — Baseline Models & Fairness Auditing
- **CNN baseline:** 4 conv blocks (32→64→128→256), 20 epochs, achieved **83.39% test accuracy**
- **VGG16:** Transfer learning from ImageNet, 10 epochs, ~85% accuracy
- **Fairness audit:** Chi-squared test yielded χ²=55.67, p≈8.17×10⁻¹³ confirming racial skew
- **Accuracy gap:** 8.44% between highest (Race 2: 91.38%) and lowest (Race 1: 82.94%) groups
- **Finding:** Fairness through Blindness alone is insufficient — facial features act as racial proxies
- Documented System Transparency Assessment per IEEE 7001-2021

### Week 3 — Advanced Models, XAI & Testing
- Trained ResNet-50, ViT-B/16, and MobileNetV2 with differential learning rates
- All models exceeded **97% validation accuracy** with near-perfect Infant/Senior F1 (≥0.99)
- Reduced racial accuracy gap from 8.44% → **3.05%** (MobileNetV2)
- Implemented Grad-CAM and ViT attention map visualizations
- Built **41 automated tests** achieving **99% code coverage** (355 statements, 2 missed)

---

## Live Demo

The deployed Streamlit app uses MobileNetV2 with OpenCV Haar Cascade face detection and WebRTC for real-time webcam inference.

**Try it:** [age-estimation-ai.streamlit.app](https://age-estimation-ai.streamlit.app/)

### Run Locally

```bash
git clone https://github.com/subodh1999/age-estimation-ai.git
cd age-estimation-ai
pip install -r requirements.txt
streamlit run app.py
```

Requires: Python 3.8+, webcam, and `best_mobilenetv2.pth` in the same directory.

> **Note:** If `best_mobilenetv2.pth` exceeds GitHub's 100MB limit, upload it to Google Drive or Hugging Face and add a download link here.

---

## Testing

| Component | Tests | Result |
|-----------|-------|--------|
| Data Pipeline (age mapping, dataset, transforms) | 12 | Passed |
| Model Architectures (ResNet-50, ViT-B/16, MobileNetV2) | 8 | Passed |
| Training Logic (loss, optimizer, scheduler, save/load) | 5 | Passed |
| Fairness Metrics (Chi-Squared, accuracy gap, bias detection) | 5 | Passed |
| Grad-CAM / XAI | 5 | Passed |
| Robustness & Edge Cases | 6 | Passed |

**41/41 tests passed · 99% code coverage · 29.18s runtime**

```bash
pip install pytest pytest-cov
pytest --cov
```

Tests use synthetic `DummyAgeDataset` instances — no real image data required, CI/CD ready.

---

## Datasets

| Dataset | Images | Used For |
|---------|--------|----------|
| [FairFace](https://www.kaggle.com/datasets/abdulwasay551/fairface-race) | 124,461 | Initial demographic audit (Week 1) |
| [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) | ~67,000 | Training: CNN (Week 2), all models (Week 3) |
| [MORPH-2](https://www.kaggle.com/datasets/chiragsaipanuganti/morph) | ~40,000 | Training + test evaluation (Week 2 & 3) |

See [Setup_Guide.md](Setup_Guide.md) for data download links, folder structure, and path configuration.

---

## Regulatory Compliance

| Regulation | Article | Status | Evidence |
|-----------|---------|--------|----------|
| EU AI Act | Art. 10 (Data Governance) | ✅ Documented | Cross-model fairness audit with Chi-Squared tests |
| EU AI Act | Art. 13 (Transparency) | ✅ Documented | Grad-CAM and ViT attention visualizations |
| EU AI Act | Art. 14 (Human Oversight) | ⚠️ Required | Manual override recommended for borderline cases |
| EU AI Act | Art. 15 (Robustness) | ✅ Pass | 41/41 tests, 99% coverage, all models > 97% |
| GDPR | Art. 5 (Data Minimization) | ✅ Pass | Age bucketing limits biometric precision |
| GDPR | Art. 9 (Special Category Data) | ✅ Pass | Aligned crops only; no individual identification |
| GDPR | Art. 22 (Right to Explanation) | ✅ Improved | XAI provides model-level explainability |
| IEEE 7001 | Transparency Standard | ✅ Documented | Full STA covering all 5 transparency dimensions |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchvision, MobileNetV2, ResNet-50, ViT-B/16, VGG16, CNN |
| Computer Vision | OpenCV, Haar Cascades, PIL |
| Explainability | Grad-CAM, ViT Attention Maps |
| Deployment | Streamlit, streamlit-webrtc, WebRTC |
| Fairness | Chi-squared tests, cross-model demographic accuracy audits |
| Testing | pytest, pytest-cov (99% coverage, 41 tests) |
| Data | pandas, NumPy, Matplotlib, Seaborn, scikit-learn |

---

## Author

**Subodh Nadkar**
M.Sc. Applied Data Science & Analytics — SRH University Heidelberg

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/subodh-nadkar)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/subodh1999)

---
