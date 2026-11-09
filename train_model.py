"""
Wine Quality Prediction System
Dataset: UCI Wine Quality — Red Wine CSV
Target rule: quality >= 6 => GOOD QUALITY (1); quality < 6 => NEEDS IMPROVEMENT (0)

Run:
    python train_model.py

This script downloads the red wine dataset, inspects it, trains models, evaluates them,
runs cross-validation, creates a confusion-matrix plot and feature-importance plot,
and saves the selected Random Forest pipeline as wine_model.pkl.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DATA_FILE = "wine_quality.csv"
MODEL_FILE = "wine_model.pkl"
RANDOM_STATE = 42


def load_data():
    """Download data if not available locally; return dataframe."""
    if Path(DATA_FILE).exists():
        df = pd.read_csv(DATA_FILE)
    else:
        print("Downloading UCI red wine dataset...")
        df = pd.read_csv(DATA_URL, sep=";")
        df.to_csv(DATA_FILE, index=False)

    print("\\n--- Dataset inspection ---")
    print("Shape:", df.shape)
    print("\\nFirst five rows:\\n", df.head())
    print("\\nData types:\\n", df.dtypes)
    print("\\nMissing values:\\n", df.isnull().sum())
    print("\\nDuplicate rows:", df.duplicated().sum())
    print("\\nOriginal quality distribution:\\n", df["quality"].value_counts().sort_index())
    return df


def build_pipeline(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def main():
    df = load_data()
    df = df.drop_duplicates().copy()

    # Binary target derived from quality; original quality is excluded from predictors.
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        ),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)
        ),
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE)
        ),
    }

    print("\\n--- Holdout model comparison (same train/test split) ---")
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (Good)": precision_score(y_test, pred, zero_division=0),
            "Recall (Good)": recall_score(y_test, pred, zero_division=0),
            "F1 (Good)": f1_score(y_test, pred, zero_division=0),
        })
        print(f"\\n{name}")
        print(classification_report(
            y_test, pred, target_names=["Needs improvement", "Good quality"],
            zero_division=0
        ))

    comparison = pd.DataFrame(rows).sort_values("F1 (Good)", ascending=False)
    comparison.to_csv("model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # Cross-validation using training split only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\\n--- 5-fold cross-validation (training split) ---")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        print(f"{name}: F1 mean={scores.mean():.3f}, std={scores.std():.3f}")

    # Select Random Forest as the lab's main model and save it.
    rf_model = models["Random Forest"]
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print("\\n--- Random Forest final evaluation ---")
    print("Accuracy:", round(accuracy_score(y_test, rf_pred), 3))
    print("Confusion matrix:\\n", confusion_matrix(y_test, rf_pred))
    print(classification_report(
        y_test, rf_pred, target_names=["Needs improvement", "Good quality"],
        zero_division=0
    ))

    # Confusion matrix plot.
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Needs improvement", "Good quality"],
        yticklabels=["Needs improvement", "Good quality"]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=160)
    plt.close()

    # Random Forest feature importance.
    classifier = rf_model.named_steps["classifier"]
    importances = pd.Series(classifier.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=160)
    plt.close()
    importances.sort_values(ascending=False).to_csv("feature_importance.csv", header=["Importance"])

    # Save trained model and feature names for app use.
    joblib.dump(rf_model, MODEL_FILE)
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "target_rule": "quality >= 6 => GOOD QUALITY; quality < 6 => NEEDS IMPROVEMENT",
            "features": list(X.columns),
            "selected_model": "Random Forest",
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
        }, f, indent=2)

    print(f"\\nSaved {MODEL_FILE}, evaluation plots, and report CSV files.")


if __name__ == "__main__":
    main()
