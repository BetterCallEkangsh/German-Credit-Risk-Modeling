"""Train the credit-risk model and save it for the Streamlit app.

Usage: python train.py path/to/german_credit_data.csv
SMOTE lives inside the pipeline, so it is fit on training folds only (no CV leakage).
"""
import json
import sys

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
NUM = ["Age", "Job", "Credit amount", "Duration"]

path = sys.argv[1] if len(sys.argv) > 1 else "german_credit_data.csv"
df = pd.read_csv(path, index_col=0)

# Missing account info is treated as its own category rather than mode-imputed
for c in ["Saving accounts", "Checking account"]:
    df[c] = df[c].fillna("no info")

X = df[CAT + NUM]
y = df["Risk"].map({"good": 0, "bad": 1})
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", StandardScaler(), NUM),
])
pipe = Pipeline([
    ("pre", pre),
    ("smote", SMOTE(random_state=42)),
    ("clf", AdaBoostClassifier(n_estimators=300, learning_rate=1.0, random_state=42)),
])

cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()

pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
prob = pipe.predict_proba(X_te)[:, 1]
metrics = {
    "cv_auc": round(float(cv_auc), 3),
    "test_auc": round(float(roc_auc_score(y_te, prob)), 3),
    "test_recall_bad": round(float(recall_score(y_te, pred)), 3),
    "test_f1_bad": round(float(f1_score(y_te, pred)), 3),
    "test_accuracy": round(float(accuracy_score(y_te, pred)), 3),
}
print(metrics)

joblib.dump(pipe, "model.joblib")
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
