# 🧠 Student Health Monitoring and Early Detection Using Machine Learning

A final-year Machine Learning project that predicts the **likelihood of
depression risk among students** from their demographic, academic, lifestyle,
and psychological attributes. The trained **Support Vector Machine (SVM)** model
is served through a clean, fully responsive **Streamlit** web application.

---

## 📌 Overview

| Item | Detail |
|------|--------|
| **Problem** | Early detection of depression risk in students |
| **Dataset** | Student Depression Dataset (27,901 records, 18 raw columns) |
| **Target** | `Depression` (0 = No, 1 = Yes) |
| **Final model** | Support Vector Machine (RBF kernel) |
| **Why SVM?** | It achieved the best overall performance (highest F1 & accuracy) compared to the Random Forest baseline |
| **Frontend** | Streamlit (responsive: desktop, tablet, mobile) |

---

## 📊 Model Performance (held-out 20% test set)

| Metric | SVM (deployed) | Random Forest (baseline) |
|--------|:--------------:|:------------------------:|
| Accuracy  | **0.8412** | 0.8380 |
| Precision | 0.8511 | 0.8516 |
| Recall    | **0.8834** | 0.8761 |
| F1 Score  | **0.8670** | 0.8637 |

> SVM wins on Accuracy, Recall and F1 — confirming it as the deployment model.

---

## 🗂️ Project Structure

```
machine_learning/
│
├── Student_Depression_Dataset.csv   # raw dataset
├── train_model.py                   # Phase 1 (analysis) + Phase 2 (final model)
├── train_model.ipynb                # notebook version of the training pipeline
├── app.py                           # Streamlit application
│
├── model.pkl                        # trained SVM (generated)
├── scaler.pkl                       # fitted StandardScaler (generated)
├── encoders.pkl                     # dict of LabelEncoders (generated)
├── feature_metadata.json            # auto-generated input definitions
│
├── requirements.txt                 # pinned dependencies
└── README.md                        # this file
```

---

## 🔬 Machine Learning Pipeline

1. **Load** the dataset.
2. **Inspect** structure, data types, and class balance.
3. **Detect & handle missing values**
   - numeric columns → median imputation
   - categorical columns → mode imputation
4. **Drop non-predictive / dirty columns** (see *Assumptions* below).
5. **Label-encode** all categorical columns (encoders saved for the app).
6. **Split** into features `X` and target `y`; `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
7. **Scale** features with `StandardScaler` (fit on training data only).
8. **Train** the SVM (RBF kernel, `probability=True`).
9. **Evaluate** with Accuracy, Precision, Recall, F1, and a Confusion Matrix
   (Random Forest trained alongside for comparison).
10. **Phase 2:** retrain the final SVM on the **full cleaned dataset** and save
    all artifacts used by the app.

---

## 🧹 Assumptions Made From the Dataset Structure

These data-quality decisions are encoded in `train_model.py` (`DROP_COLS`):

| Dropped column | Reason |
|----------------|--------|
| `id` | Unique row identifier — no predictive value. |
| `City` | 52 levels, including dirty entries (`'3.0'`, `'City'`, person names). |
| `Profession` | 99.9% of rows are `"Student"` — effectively constant. |
| `Work Pressure` | 99.99% zeros — effectively constant for students. |
| `Job Satisfaction` | 99.97% zeros — effectively constant for students. |

The remaining **12 features** are clean, meaningful, and used for both training
and the app form:

`Gender`, `Age`, `Academic Pressure`, `CGPA`, `Study Satisfaction`,
`Sleep Duration`, `Dietary Habits`, `Degree`,
`Have you ever had suicidal thoughts ?`, `Work/Study Hours`,
`Financial Stress`, `Family History of Mental Illness`.

> The app reads `feature_metadata.json`, so its input widgets are generated
> **automatically** from these columns — change the dataset/columns, re-run
> `train_model.py`, and the form updates itself.

---

## 🚀 Getting Started (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Re)generate the model artifacts
```bash
python train_model.py
```
This prints the full Phase 1 analysis and produces `model.pkl`, `scaler.pkl`,
`encoders.pkl`, and `feature_metadata.json`.

### 3. Run the app
```bash
streamlit run app.py
```
Open the URL shown in the terminal (default `http://localhost:8501`).

---

## ☁️ Deploying to Streamlit Cloud

1. Push this project to a **public GitHub repository** (include the generated
   `model.pkl`, `scaler.pkl`, `encoders.pkl`, and `feature_metadata.json`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **“New app”**, then select:
   - **Repository:** your repo
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.
5. Your app goes live at `https://<your-app-name>.streamlit.app`.

> **Tip:** The dependency versions in `requirements.txt` are pinned to match the
> versions used when the model was pickled — this guarantees the artifacts load
> correctly on the cloud. If you retrain with newer libraries, update the pins.

---

## 📱 Responsive Design

The UI is built mobile-first and verified for phones, tablets, and desktops:

- `layout="wide"` with a width-constrained, centered content container.
- Two-column input groups that **stack vertically on small screens** (no
  horizontal scrolling).
- Custom CSS media queries scale the header/spacing on narrow viewports.
- Touch-friendly sliders and dropdowns; a full-width primary action button.
- Tables rendered with `use_container_width=True` so they never overflow.

---

## ⚠️ Disclaimer

This application is an **educational early-screening tool**, not a medical
device. Its output is a statistical estimate and must **not** be used as a
clinical diagnosis. Anyone with mental-health concerns should consult a
qualified professional.
