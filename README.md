# Wine Quality Prediction System

A classroom prototype using the UCI Red Wine Quality dataset to classify samples as:
- **GOOD QUALITY**: original `quality >= 6`
- **NEEDS IMPROVEMENT**: original `quality < 6`

The original `quality` column is used only to construct the target and is removed from model inputs to prevent target leakage.

## Contents

- `train_model.py` — downloads/inspects the dataset, preprocesses data, trains and compares Random Forest, Logistic Regression, and Decision Tree; performs 5-fold cross-validation; creates confusion matrix and feature importance artifacts.
- `app.py` — Streamlit app with CSV upload, prediction, probability threshold slider, and result download.
- `requirements.txt` — dependencies.
- `SAD_and_reflection.md` — short SAD documentation and reflection responses.

## Requirements

Python 3.10 or later.

## Install and run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script downloads the red-wine CSV from the UCI repository if `wine_quality.csv` is not present. Internet access is required for the initial download.

## Dataset

UCI Machine Learning Repository — Wine Quality:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Model and evaluation

The main classifier is Random Forest. The script uses an 80/20 stratified train/test split with `random_state=42`. It prints accuracy, precision, recall, F1-score, and a classification report. It also compares models using the same split and reports 5-fold cross-validation F1 scores on the training set.

Generated outputs include:
- `wine_quality.csv`
- `wine_model.pkl`
- `model_comparison.csv`
- `confusion_matrix.png`
- `feature_importance.png`
- `feature_importance.csv`
- `model_metadata.json`

## GitHub publishing

Create a repository named `wine-quality-prediction`. Then:

```bash
git init
git add train_model.py app.py requirements.txt README.md SAD_and_reflection.md
git commit -m "Build wine quality prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Never commit passwords, API keys, or access tokens.

## Streamlit Community Cloud

1. Push the source files to GitHub.
2. Open https://share.streamlit.io/
3. Select the repository and `app.py` as the entry point.
4. Set the Python version if prompted and deploy.

For cloud deployment, make sure `wine_quality.csv` and `wine_model.pkl` are available in the repository or modify the app to train/load artifacts through an approved deployment workflow. Do not upload private credentials.

## Limitations and responsible use

This model was trained on a specific red-wine dataset. Performance may differ for other wine types, producers, regions, or testing conditions. Use it only as preliminary screening. Predictions must not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control judgment.
