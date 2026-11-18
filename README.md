# 🧬 MediCore AI – Breast Cancer Prediction System

MediCore AI is an educational Streamlit dashboard for exploring breast-tumour classification with machine learning. It combines an interactive prediction form, model-performance analytics, and a transparent preprocessing workflow in one polished interface.

> ⚠️ **Educational use only:** This project is a machine-learning demonstration and is **not a clinical diagnostic tool**. Its outputs must not be used for medical decisions, screening, diagnosis, or treatment.

## ✨ Project Overview

The application trains a Logistic Regression classifier on scikit-learn’s Breast Cancer Wisconsin Diagnostic dataset and presents the result through a three-page Streamlit dashboard:

- **Overview** — summarizes the dataset, input dimensions, model, and test performance.
- **AI Prediction** — accepts 30 numeric tumour measurements and returns benign/malignant class probabilities.
- **Model Analytics** — displays evaluation metrics and a visual metric comparison chart.

### Objectives

1. Demonstrate a complete binary-classification workflow in Python.
2. Show how outlier capping and feature standardization can be applied before training.
3. Make model predictions and evaluation results easy to inspect through an interactive UI.
4. Provide a clear, reproducible starting point for learning about applied machine learning.

## 🖥️ Dashboard Features

| Area | What it provides |
|---|---|
| Overview | Dataset size, feature count, test accuracy, model name, workflow summary, and performance snapshot |
| AI Prediction | 30 numeric inputs grouped into mean, standard-error, and worst-value measurements |
| Input defaults | Each field starts with that feature’s dataset median to make exploration easier |
| Prediction output | Benign and malignant probabilities plus a 50% classification threshold |
| Model Analytics | Accuracy, precision, recall, F1-score, and a bar chart of metric scores |
| Interface | Wide layout, expanded sidebar navigation, custom dark theme, responsive metric cards, forms, expanders, and progress bars |
| Runtime behavior | The trained model is cached with `st.cache_resource` for reuse during the Streamlit session |

## 🧰 Technologies and Libraries

| Technology | Role |
|---|---|
| Python | Application and machine-learning implementation |
| Streamlit | Interactive dashboard and user interface |
| pandas | Tabular data handling and feature construction |
| NumPy | Numerical Python dependency imported by the application |
| scikit-learn | Dataset loading, splitting, scaling, model training, and evaluation |
| Logistic Regression | Binary classification model |
| HTML/CSS | Custom dashboard styling embedded through Streamlit markdown |

## 🗂️ Dataset

The application loads the dataset directly from scikit-learn with:

```python
load_breast_cancer(as_frame=True)
```

The dataset used by the application contains:

| Property | Value |
|---|---:|
| Records | 569 |
| Numeric input features | 30 |
| Target | Binary tumour class |
| Target mapping in this project | `1 = malignant`, `0 = benign` |
| Input groups | Mean, standard error, and worst-value measurements |

The original scikit-learn target is remapped in the source code with `target == 0`, so that malignant cases become the positive class used for evaluation and prediction probabilities.

## 🤖 Machine Learning Workflow

The model is trained when the application starts, inside the cached `load_model()` function.

### 1. Data loading and target encoding

The Breast Cancer Wisconsin Diagnostic dataset is loaded as a pandas-compatible frame. All 30 named measurement columns are selected as `X`. The target is converted to the project’s convention:

```python
y = (data.frame["target"] == 0).astype(int)
```

### 2. Stratified train-test split

The data is divided into training and held-out test sets using an 80:20 split:

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

This produces 455 training records and 114 test records for the dataset used by the app.

### 3. IQR-based outlier handling

The first and third quartiles are calculated from the training features. The interquartile range is:

```text
IQR = Q3 − Q1
```

Each feature is capped to the interval:

```text
[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]
```

The same training-derived lower and upper bounds are applied to both the training and test sets. Prediction inputs are capped with those same bounds as well.

### 4. Standardization

`StandardScaler` is fitted on the capped training data and then used to transform the capped training data, held-out test data, and user-entered prediction row.

### 5. Logistic Regression

The classifier is created and trained as follows:

```python
LogisticRegression(max_iter=1000, random_state=42)
```

The fitted model provides both class predictions and probability estimates through `predict_proba`.

## 📊 Model Evaluation

The values below are calculated by the application on the held-out test set using scikit-learn’s `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`.

| Metric | Score | Percentage |
|---|---:|---:|
| Accuracy | 0.9649122807 | 96.49% |
| Precision | 0.9750000000 | 97.50% |
| Recall | 0.9285714286 | 92.86% |
| F1-score | 0.9512195122 | 95.12% |

These results correspond to the source-defined split (`random_state=42`, stratification enabled), preprocessing sequence, and Logistic Regression configuration. They should be treated as demonstration results rather than evidence of clinical performance.

## 📁 Project Structure

Within the project scope represented by the supplied app, the structure is:

```text
Downloads/
├── cancer_app_medicore.py   # Streamlit dashboard and ML pipeline
└── README.md                # Project documentation
```

No separate model file, dataset file, configuration file, or requirements file is used by the application. The dataset is loaded at runtime from scikit-learn.

## ⚙️ Installation

Open **Windows PowerShell** in the project folder and run:

```powershell
cd C:\Users\yninj\Downloads

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn
```

If PowerShell blocks local script activation, run this once for the current user according to your local security policy, then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## ▶️ Run the Application

With the virtual environment activated:

```powershell
streamlit run .\cancer_app_medicore.py
```

Streamlit will provide a local browser URL. Use the sidebar to switch between **Overview**, **AI Prediction**, and **Model Analytics**.

To stop the app, return to PowerShell and press `Ctrl+C`.

## 🔍 How Prediction Works

1. Enter or review the 30 numeric tumour measurements on the **AI Prediction** page.
2. The form groups measurements into mean, standard-error, and worst-value features.
3. Submitted values are placed into a one-row pandas DataFrame using the model’s feature order.
4. The values are capped with the training-derived IQR bounds.
5. The cached `StandardScaler` transforms the row.
6. Logistic Regression calculates class probabilities.
7. The dashboard displays benign and malignant probabilities.
8. A probability of at least 50% for the malignant class is labeled **Malignant pattern**; otherwise, it is labeled **Benign pattern**.

The displayed probabilities are model outputs and do not represent a medical diagnosis.

## 🔭 Future Improvements

Potential extensions for future versions include:

- Add a dedicated `requirements.txt` or `pyproject.toml` for reproducible dependency management.
- Separate data preparation, training, inference, and UI code into maintainable modules.
- Add automated tests for preprocessing, target encoding, prediction shape, and metric calculations.
- Add confusion matrix, ROC curve, PR curve, and per-class performance views.
- Add model versioning and an explicit training-data provenance record.
- Add input validation, accessibility improvements, and clearer feature descriptions.
- Compare additional baseline classifiers with cross-validation.
- Add explainability views such as feature coefficients or model-agnostic explanations.
- Add privacy, security, auditability, and clinical validation controls before considering any real-world healthcare use.

## ⚠️ Disclaimer

MediCore AI is an educational machine-learning project built for demonstration and learning purposes. It is **not** a medical device, clinical decision-support system, screening service, or substitute for a qualified healthcare professional. Do not use its predictions to diagnose, treat, or make decisions about any person’s health. Always consult an appropriately licensed medical professional for medical advice and evaluation.

---

<p align="center">Built as an educational demonstration of interactive machine learning with Streamlit.</p>
