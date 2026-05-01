# 📱 Mobile Price Range Classification Using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Birmingham City University](https://img.shields.io/badge/Institution-Birmingham%20City%20University-purple.svg)](https://www.bcu.ac.uk/)

> A comprehensive machine learning pipeline to classify mobile phones into four price ranges using technical specifications, with Explainable AI (SHAP) analysis.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology Overview](#methodology-overview)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [SHAP Explainability](#shap-explainability)
- [Installation and Usage](#installation-and-usage)
- [Technologies Used](#technologies-used)
- [Academic Context](#academic-context)
- [License](#license)
- [Author](#author)

---

## 🔍 About the Project

The smartphone market is globally competitive, and understanding how technical specifications influence pricing is critical for manufacturers, retailers, and consumers. This project builds and compares machine learning classification models to predict which of four price ranges a mobile phone belongs to, based on 20 technical features.

**What makes this project unique:**
- 🔧 **8 engineered features** derived from domain knowledge (pixel density, battery efficiency, processing power, etc.)
- 🧪 **Consensus-based feature selection** combining Correlation Analysis, ANOVA F-test, and Mutual Information
- 📊 **Three-tier evaluation strategy** (training, cross-validation, held-out test set)
- 🔍 **Full SHAP explainability** with global and individual prediction explanations
- 📈 **Residual analysis** examining prediction confidence vs classification errors

---

## 🏆 Key Results

| Metric | Logistic Regression | XGBoost |
|--------|--------------------|---------| 
| **Test Accuracy** | **97.75%** ✅ | 94.00% |
| **Training Accuracy** | 97.06% | 100.00% |
| **Cross-Validation Accuracy** | **95.94% ±0.68%** ✅ | 90.44% ±2.81% |
| **Test Log Loss** | 0.1829 | 0.1532 |
| **Training-Test Gap** | **-0.69%** ✅ | 6.00% |
| **Misclassifications (400 test)** | **9** ✅ | 24 |

> **Winner: Logistic Regression** — Higher test accuracy, better generalization, no overfitting, and fully interpretable.

---

## 📊 Dataset

**Source:** [Kaggle - Mobile Price Prediction](https://www.kaggle.com/datasets/fhabibimoghaddam/mobile-price-prediction) by Habibimoghaddam (2023)

- **2,000 mobile phones** with 20 technical features and 1 target variable
- **Perfectly balanced classes:** 25% per price range (500 samples each)
- **No missing values** or duplicate records
- **Target variable:** price_range (0=Low Cost, 1=Medium Cost, 2=High Cost, 3=Very High Cost)

> ⚠️ The dataset is not included in this repository due to Kaggle terms of service. Please download it from the link above and place it in the data/ folder. See data/README.md for instructions.

---

## 📁 Project Structure

mobile-price-range-classification/
│
├── README.md # This file
├── LICENSE # MIT License
├── .gitignore # Python gitignore
├── requirements.txt # Python dependencies
│
├── notebook/
│ └── mobile_price_classification.ipynb # Complete ML pipeline notebook
│
├── data/
│ └── README.md # Dataset download instructions
│
└── report/
└── Mobile_Price_Range_Classification_Report.pdf # Full academic report

---

## 🔬 Methodology Overview

The project follows a complete end-to-end machine learning pipeline:

Raw Data (2000 samples, 20 features)
↓
Exploratory Data Analysis (EDA)
↓
Outlier Detection and Treatment (IQR + Winsorization at 1st/99th percentile)
↓
Feature Engineering (8 new derived features → 28 total)
↓
Consensus Feature Selection (Correlation + ANOVA F-test + Mutual Information)
↓
11 Final Features (60.7% dimensionality reduction)
↓
Train/Test Split (80/20 | 1600 training, 400 test samples)
↓
Feature Scaling (StandardScaler for Logistic Regression)
↓
Model Training (Logistic Regression | XGBoost)
↓
Three-Tier Evaluation (Training | 5-Fold CV | Test Set)
↓
SHAP Explainability Analysis
↓
Residual Analysis and Learning Curves

---

## ⚙️ Feature Engineering

Eight domain-knowledge-driven features were engineered from the original 20 features:

| Number | Feature | Formula | Description |
|--------|---------|---------|-------------|
| 1 | Total_Pixels | Pixel_H x Pixel_W | Overall display resolution |
| 2 | Screen_Size | sqrt(Screen_H squared + Screen_W squared) | Diagonal screen measurement |
| 3 | Pixel_Density | Total_Pixels / (Screen_H x Screen_W) | Pixels per square cm |
| 4 | Total_Camera_MP | FC + PC | Combined camera capability |
| 5 | Aspect_Ratio | Mobile_D / Mobile_W | Device form factor |
| 6 | Battery_Efficiency | Talk_Time / (Battery_Power / 1000) | Usage per battery unit |
| 7 | Connectivity_Score | Four_G + Three_G + Bluetooth + WiFi | Overall connectivity |
| 8 | Processing_Power | Cores x Clock_Speed | Computational capability |

---

## 📈 Model Performance

### Logistic Regression Configuration

- Solver: L-BFGS (Limited-memory BFGS)
- Max Iterations: 1000
- Multi-class Strategy: Multinomial
- Regularization: L2 (Ridge)

### XGBoost Configuration

- Estimators: 400
- Learning Rate: 0.05
- Max Depth: 6
- Min Child Weight: 3
- Subsample: 0.85
- Column Subsample: 0.85
- L2 Lambda: 1.2
- Gamma: 0.2

### Per-Class Results (Logistic Regression)

| Price Range | Correct/Total | Notes |
|-------------|---------------|-------|
| Range 0 (Low Cost) | 100/100 | Perfect classification |
| Range 1 (Medium Cost) | 97/100 | 3 misclassified as Range 2 |
| Range 2 (High Cost) | 97/100 | Most challenging category |
| Range 3 (Very High Cost) | 97/100 | Strong performance |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) analysis was applied to the Logistic Regression model to provide transparent, theoretically-grounded feature importance explanations based on cooperative game theory.

### Global Feature Importance (Mean Absolute SHAP Values)

| Rank | Feature | SHAP Value | Interpretation |
|------|---------|-----------|----------------|
| 1 | RAM | 6.16 | Dominant predictor, ~4x more important than any other feature |
| 2 | Battery_Power | 1.60 | Secondary differentiator between price tiers |
| 3 | Pixel_W | 0.83 | Display quality indicator |
| 4 | Pixel_H | 0.50 | Combined with Pixel_W for resolution impact |
| 5 | Total_Pixels | 0.38 | Engineered feature validating display importance |
| 6 | Mobile_W | 0.28 | Physical dimension with minor but real effect |
| 7 | Screen_Size | 0.13 | Lower importance than expected |
| 8 to 11 | Pixel_Density, Aspect_Ratio, Screen_W, Battery_Efficiency | less than 0.05 | Minimal individual contribution |

### Individual Prediction Analysis

Three representative predictions were explained using SHAP force plots:

- **High Confidence Correct (97.8% confidence):** High RAM (3912 MB) and moderate battery strongly pushed prediction toward Range 3. All features consistently aligned with the forecast.

- **Low Confidence Correct (56.7% confidence):** RAM (2177 MB) placed the device between Range 1 and Range 2. Competing forces between battery and display features caused uncertainty.

- **Incorrect Prediction (65.4% confidence):** RAM (1482 MB) caused the model to predict Range 1 when true label was Range 2. A boundary case where feature values overlapped between adjacent price categories.

### Key Insight

RAM with a SHAP value of 6.16 is by far the most important predictor, nearly 4 times more influential than Battery_Power. This aligns with industry knowledge that memory capacity is the primary differentiator between budget and premium devices. Modern mobile operating systems and applications increasingly demand more RAM, making it the most critical specification in manufacturer pricing tiers.

---

## 🚀 Installation and Usage

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Kaggle account (for dataset download)

### Step 1: Clone the Repository

git clone https://github.com/Utu8848/mobile-price-range-classification.git
cd mobile-price-range-classification

### Step 2: Install Dependencies

pip install -r requirements.txt

### Step 3: Download the Dataset

1. Go to https://www.kaggle.com/datasets/fhabibimoghaddam/mobile-price-prediction
2. Download the CSV file
3. Place it in the data/ folder
4. See data/README.md for detailed instructions

### Step 4: Run the Notebook

jupyter notebook notebook/mobile_price_classification.ipynb

### Open in Google Colab

The original implementation is available at:
https://gist.github.com/Utu8848/6219ed47050dcc002283ad12629e0625

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.8+ |
| ML Framework | Scikit-learn |
| Gradient Boosting | XGBoost |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Development Environment | Jupyter Notebook / Google Colab |
| Feature Selection | Correlation Analysis, ANOVA F-test, Mutual Information |
| Outlier Treatment | Winsorization using IQR method |
| Validation Strategy | Stratified 5-Fold Cross-Validation |

---

## 🎓 Academic Context

| Detail | Information |
|--------|-------------|
| Author | Utsav Rai |
| Student ID | 25123857 |
| Course | BSc (Hons) Computer Science with Artificial Intelligence |
| Institution | Birmingham City University |
| Date | January 26, 2026 |

### Research Objectives

1. Estimate mobile phone price ranges using technical specifications
2. Compare Logistic Regression vs XGBoost for multi-class classification
3. Identify key price-driving features using SHAP explainability
4. Provide actionable recommendations for mobile pricing strategies
5. Demonstrate a complete and reproducible machine learning pipeline

### Key Contributions

- Consensus-based feature selection reducing dimensionality by 60.7%
- 8 engineered features capturing domain-specific device characteristics
- Full explainability layer with global and per-prediction SHAP analysis
- Demonstration that interpretable linear models can outperform complex ensembles on structured datasets

### Research Findings Summary

- RAM is the single most important pricing predictor (SHAP value 6.16)
- Battery power is the secondary differentiator (SHAP value 1.60)
- Display specifications collectively influence pricing through pixel dimensions
- Connectivity features (WiFi, Bluetooth, 4G) show surprisingly low importance, suggesting standardization across price tiers
- Physical form factor has minimal pricing influence; performance specifications dominate
- Logistic Regression outperformed XGBoost due to quality feature engineering creating near-linear decision boundaries, appropriate dataset size favoring simpler models, and effective L2 regularization

### Limitations

- Dataset limited to 2,000 samples
- No brand, operating system, or release date information
- No regional or temporal market context
- Four-class classification limits pricing granularity

### Future Work

- Incorporate brand and OS-specific features
- Add temporal market trend data
- Explore deep learning for complex feature interactions
- Apply SHAP analysis to ensemble models
- Develop market-specific regional models
- Build regression models for exact price prediction

---

## 📚 Key References

Lundberg, S.M. and Lee, S.-I. (2017) A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems.

Rudin, C. (2019) Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence.

Chandrashekar, G. and Sahin, F. (2014) A survey on feature selection methods. Computers and Electrical Engineering, 40(1): 16-28.

Habibimoghaddam, F. (2023) Mobile Price Prediction. Kaggle Dataset. Available at: https://www.kaggle.com/datasets/fhabibimoghaddam/mobile-price-prediction

Pramanik, R. et al. (2021) Comparative Analysis of Mobile Price Classification Using Feature Engineering Techniques. ISCON 2021.

> Full references with DOIs are available in the academic report in the report/ folder.

---

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for full details.

MIT License - Copyright (c) 2026 Utsav Rai

You are free to use, modify, and distribute this code for any purpose with proper attribution.

---

## 👤 Author

Utsav Rai
BSc (Hons) Computer Science with Artificial Intelligence
Birmingham City University

GitHub: https://github.com/Utu8848
Implementation Gist: https://gist.github.com/Utu8848/6219ed47050dcc002283ad12629e0625

---

Built with dedication for academic research | Birmingham City University 2026
