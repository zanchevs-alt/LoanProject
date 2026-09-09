"""Train the loan approval model and write its metadata."""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE_PATH = BASE_DIR / "svm_pipeline_model.pkl"
METADATA_FILE_PATH = BASE_DIR / "model_metadata.json"
df = pd.read_csv(BASE_DIR / "train.csv")
numeric_median_features = ["CoapplicantIncome", "ApplicantIncome", "LoanAmount"]
numeric_mode_features = ["Credit_History", "Loan_Amount_Term"]
categorical_features = ["Self_Employed", "Education"]
features = numeric_median_features + numeric_mode_features + categorical_features
target = "Loan_Status"
X, y = df[features].copy(), df[target].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_median_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
numeric_mode_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="most_frequent")), ("scaler", StandardScaler())]
)
categorical_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]
)
preprocessor = ColumnTransformer(
    [("num_median", numeric_median_transformer, numeric_median_features),
     ("num_mode", numeric_mode_transformer, numeric_mode_features),
     ("cat", categorical_transformer, categorical_features)]
)
pipeline = Pipeline(
    [("preprocessor", preprocessor), ("model", SVC(kernel="rbf", random_state=42))]
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
svc_model = pipeline.named_steps["model"]
joblib.dump(pipeline, MODEL_FILE_PATH)
sample_rows = (df[features + [target]].head(10).astype(object)
               .where(pd.notnull(df[features + [target]].head(10)), None)
               .to_dict(orient="records"))
metadata = {
    "features": {"numeric_median_fill": numeric_median_features, "numeric_mode_fill": numeric_mode_features, "categorical": categorical_features, "all": features},
    "target": target,
    "classes": svc_model.classes_.tolist(),
    "model_info": {
        "algorithm": "SVC (Support Vector Classifier)", "kernel": svc_model.kernel, "C": svc_model.C,
        "gamma": str(svc_model.gamma), "decision_function_shape": svc_model.decision_function_shape,
        "total_support_vectors": int(sum(svc_model.n_support_)),
        "support_vectors_per_class": {str(cls): int(count) for cls, count in zip(svc_model.classes_, svc_model.n_support_)},
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"), "train_rows": len(X_train), "test_rows": len(X_test),
    },
    "metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "confusion_matrix_labels": svc_model.classes_.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    },
    "margins_on_test_set": pipeline.decision_function(X_test).tolist(), "samples": sample_rows,
    "model_file": {"name": MODEL_FILE_PATH.name, "size_kb": round(os.path.getsize(MODEL_FILE_PATH) / 1024, 2)},
}
with METADATA_FILE_PATH.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)
print(f"Accuracy: {metadata['metrics']['accuracy']:.4f}")
print(f"Model saved to {MODEL_FILE_PATH.name}")
print(f"Metadata saved to {METADATA_FILE_PATH.name}")