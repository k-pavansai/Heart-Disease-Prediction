import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease_prediction_2026.csv")
df = pd.read_csv(DATA_PATH)
target = "heart_disease"

if target not in df.columns:
    raise ValueError("Target column 'heart_disease' is missing.")

df = df.dropna(subset=[target]).copy()
X = df.drop(columns=[target])
y = df[target].astype(int)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
}

results = []
best_name, best_f1, best_model = None, -1, None

for name, classifier in models.items():
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0)
    }
    results.append(scores)
    print("\n", name)
    print(classification_report(y_test, pred, target_names=["No Disease", "Disease"]))
    if scores["Macro F1"] > best_f1:
        best_name, best_f1, best_model = name, scores["Macro F1"], model
    if name == "Random Forest":
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["No Disease", "Disease"], cmap="Blues"
        )
        plt.title("Heart Disease - Random Forest Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=200)
        plt.show()

results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))
print("\nBest model based on Macro F1:", best_name)
print("Target classes: 0 = No Disease, 1 = Disease")
