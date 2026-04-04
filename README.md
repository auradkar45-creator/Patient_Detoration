# SepsisGuard AI: Early Warning System for Patient Deterioration

[![Python 3+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)


An advanced clinical decision support system designed to predict patient deterioration (Sepsis) several hours before clinical onset. By leveraging ensemble machine learning (LightGBM + XGBoost) and the PhysioNet/CinC Challenge 2019 dataset, this system identifies subtle biochemical signals of decline with high precision.

## 🚀 Key Features

* **Ensemble Modeling:** Hybrid architecture combining **LightGBM** (60%) and **XGBoost** (40%) to maximize predictive stability.
* **Explainable AI (XAI):** Full integration with **SHAP** (SHapley Additive Explanations) to provide clinicians with the "why" behind every risk score.
* **Imbalance Management:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) and cost-sensitive learning to handle the high class-imbalance (8% sepsis rate).
* **Real-time Dashboard:** A Streamlit-based UI for patient triage, risk visualization, and dynamic threshold adjustment.
* **Bayesian Optimization:** Automated hyperparameter tuning via **Optuna** to achieve a state-of-the-art ROC-AUC.

## 📊 Performance Summary

| Metrics | Value |
| :--- | :--- |
| **ROC-AUC** | **0.9253** |
| **Accuracy** | 95.60% |
| **Precision** | 74.36% |
| **Recall (Sensitivity)** | 60.00% |
| **Optimal Threshold** | 0.35 |

## 🛠️ Project Structure

* `train.py`: The core engineering pipeline—includes feature engineering, SMOTE application, Optuna tuning, and ensemble training.
* `app.py`: The frontend application script; handles real-time risk scoring, patient prioritization, and SHAP visualizations.
* `plots.py`: Evaluation suite that generates ROC curves, Confusion Matrices, Precision-Recall curves, and SHAP summary plots.
* `report.pdf`: IEEE formatted comprehensive technical documentation and clinical analysis.

  ## 👥 Contributors

- [**Burela Bhavan**](https://github.com/BurelaBhavan)
- [**Harsha P**](https://github.com/harshapremesh12-bit)
- [**Aditya R Auradkar**](https://github.com/auradkar45-creator)
- **Surya Prakash**

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/auradkar45-creator/Patient_Detoration.git](https://github.com/auradkar45-creator/Patient_Detoration.git)
cd Patient_Detoration
