"""
================================================================================
Student Health Monitoring and Early Detection Using Machine Learning Techniques
--------------------------------------------------------------------------------
train_model.py

This single script covers PHASE 1 (data analysis & evaluation) and PHASE 2
(final model training & artifact saving) of the project.

What it does
------------
PHASE 1 - DATA ANALYSIS
    1.  Load the dataset
    2.  Display dataset information
    3.  Detect missing values
    4.  Handle missing values appropriately
    5.  Identify categorical columns
    6.  Apply Label Encoding
    7.  Separate X (features) and y (Depression target)
    8.  Train/test split (test_size=0.2, random_state=42)
    9.  Apply StandardScaler
    10. Train an SVM model
    11. Evaluate performance (Accuracy, Precision, Recall, F1, Confusion Matrix)
        - A Random Forest baseline is also trained to justify why SVM is the
          chosen deployment model.

PHASE 2 - FINAL MODEL
    -   Re-train the final SVM on the FULL cleaned dataset.
    -   Persist every object the Streamlit app needs:
            model.pkl              -> trained SVC (with probability=True)
            scaler.pkl             -> fitted StandardScaler
            encoders.pkl           -> dict {column_name: fitted LabelEncoder}
            feature_metadata.json  -> describes each input so the app can build
                                      its form automatically (column order,
                                      type, options, ranges, defaults).

Run:
    python train_model.py
================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Student_Depression_Dataset.csv"

TARGET = "Depression"

# Columns intentionally removed before modelling.  Rationale (see README):
#   id              -> a unique row identifier with no predictive value.
#   City            -> 52 levels containing dirty data ('3.0', 'City', names).
#   Profession      -> 99.9% "Student"; effectively constant.
#   Work Pressure   -> 99.99% zeros; effectively constant for students.
#   Job Satisfaction-> 99.97% zeros; effectively constant for students.
DROP_COLS = ["id", "City", "Profession", "Work Pressure", "Job Satisfaction"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Output artifact paths
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "encoders.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def banner(text: str) -> None:
    """Print a clearly visible section header."""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-predictive / dirty columns and handle missing values."""
    # 1) Drop the columns that add noise or carry no signal.
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2) Handle missing values.
    #    - Numeric columns  -> median (robust to outliers).
    #    - Categorical cols -> most frequent value (mode).
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode every categorical (object) column except the target.

    Returns the encoded DataFrame and a dict of fitted encoders so the exact
    same mapping can be reproduced inside the Streamlit app.
    """
    encoders: dict[str, LabelEncoder] = {}
    categorical_cols = [
        c for c in df.select_dtypes(include="object").columns if c != TARGET
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def build_feature_metadata(
    raw_df: pd.DataFrame,
    feature_order: list[str],
    encoders: dict[str, LabelEncoder],
) -> list[dict]:
    """Describe each feature so the app can auto-generate its input widgets.

    For categorical features we expose the original string options (taken from
    the fitted LabelEncoder).  For numeric features we expose min/max/median and
    whether the field should behave as an integer.
    """
    metadata: list[dict] = []
    for col in feature_order:
        if col in encoders:
            # Categorical feature -> dropdown of original labels.
            options = list(encoders[col].classes_)
            metadata.append(
                {
                    "name": col,
                    "type": "categorical",
                    "options": options,
                    "default": options[0],
                }
            )
        else:
            # Numeric feature -> slider / number input.
            series = raw_df[col].dropna()
            col_min = float(series.min())
            col_max = float(series.max())
            col_median = float(series.median())
            # Treat as an integer field when every value is a whole number.
            is_integer = bool(np.all(np.equal(np.mod(series.values, 1), 0)))
            metadata.append(
                {
                    "name": col,
                    "type": "numeric",
                    "min": col_min,
                    "max": col_max,
                    "default": col_median,
                    "is_integer": is_integer,
                    "step": 1.0 if is_integer else 0.01,
                }
            )
    return metadata


def evaluate(name: str, y_true, y_pred) -> dict:
    """Compute and print the standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    banner(f"EVALUATION - {name}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("\nConfusion Matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    return metrics


# --------------------------------------------------------------------------- #
# PHASE 1 - DATA ANALYSIS & EVALUATION
# --------------------------------------------------------------------------- #
def phase_1(raw_df: pd.DataFrame):
    banner("PHASE 1 - DATA ANALYSIS")

    # Step 2 - Display dataset information
    print(f"Dataset shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns\n")
    print("Column data types:")
    print(raw_df.dtypes)

    # Step 3 - Detect missing values
    banner("MISSING VALUES (raw)")
    missing = raw_df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values detected.")

    # Steps 4-6 - Handle missing values, identify categoricals, label encode
    df = clean_dataframe(raw_df.copy())
    banner("CATEGORICAL COLUMNS IDENTIFIED")
    categorical_cols = [
        c for c in df.select_dtypes(include="object").columns if c != TARGET
    ]
    print(categorical_cols)

    df, encoders = encode_categoricals(df)

    # Step 7 - Separate features (X) and target (y)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_order = list(X.columns)

    # Step 8 - Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 9 - Feature scaling (fit on train only to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 10 - Train the SVM model
    banner("TRAINING MODELS (this can take ~1 minute for SVM)")
    print("Training Support Vector Machine (RBF kernel)...")
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)

    print("Training Random Forest baseline (for comparison)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # Step 11 - Evaluate performance
    svm_metrics = evaluate("Support Vector Machine (SVM)", y_test, svm.predict(X_test_scaled))
    rf_metrics = evaluate("Random Forest (baseline)", y_test, rf.predict(X_test_scaled))

    banner("MODEL COMPARISON SUMMARY")
    comparison = pd.DataFrame(
        {"SVM": svm_metrics, "Random Forest": rf_metrics}
    ).T.round(4)
    print(comparison)
    winner = "SVM" if svm_metrics["f1"] >= rf_metrics["f1"] else "Random Forest"
    print(f"\nBest model by F1 score: {winner}")
    print("--> SVM is selected as the final deployment model (per project spec).")

    return feature_order, encoders


# --------------------------------------------------------------------------- #
# PHASE 2 - FINAL MODEL (train on ALL data, then persist artifacts)
# --------------------------------------------------------------------------- #
def phase_2(raw_df: pd.DataFrame, feature_order: list[str]):
    banner("PHASE 2 - FINAL MODEL TRAINING (full dataset)")

    # Re-run the cleaning + encoding on the full dataset so the saved encoders
    # are fitted on every available category.
    df = clean_dataframe(raw_df.copy())
    df, encoders = encode_categoricals(df)

    X = df[feature_order]          # enforce a consistent column order
    y = df[TARGET]

    # Scale on the full dataset for the final, deployed pipeline.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Final SVM trained on 100% of the cleaned data.
    final_svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    final_svm.fit(X_scaled, y)
    print("Final SVM trained on the complete cleaned dataset.")

    # Build the metadata used by the app to render inputs automatically.
    metadata = build_feature_metadata(raw_df, feature_order, encoders)

    # ----------------------------- Persist ----------------------------- #
    joblib.dump(final_svm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {"target": TARGET, "feature_order": feature_order, "features": metadata},
            fh,
            indent=2,
        )

    banner("ARTIFACTS SAVED")
    for path in (MODEL_PATH, SCALER_PATH, ENCODERS_PATH, METADATA_PATH):
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name:<22} ({size_kb:,.1f} KB)")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    raw_df = load_dataset(DATA_PATH)
    feature_order, _ = phase_1(raw_df)
    phase_2(raw_df, feature_order)
    banner("DONE - the Streamlit app (app.py) is now ready to run")


if __name__ == "__main__":
    main()
