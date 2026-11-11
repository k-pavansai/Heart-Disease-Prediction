# Heart Disease Prediction

## Dataset
`data/heart_disease_prediction_2026.csv` (100,000 rows, 26 columns).

## Objective
Predict whether a record belongs to the **No Disease (0)** or **Disease (1)** class using the `heart_disease` target column.

## Models
- Logistic Regression: interpretable baseline for binary classification.
- Random Forest: captures nonlinear patterns and feature interactions.

## Preprocessing
Missing numeric values are median-imputed and scaled. Categorical values are most-frequent-imputed and one-hot encoded. The data is split 80:20 using stratification.

## Metrics
Accuracy, macro precision, macro recall, macro F1-score, classification report and confusion matrix.

## Run
```bash
pip install -r requirements.txt
python heart_disease_prediction.py
```

This is an educational machine-learning exercise, not a medical diagnostic tool.
