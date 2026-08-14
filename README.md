# German Credit Risk Modeling

A machine learning project to predict credit risk (good/bad) for bank loan applicants using the German Credit Dataset. Built as a portfolio project targeting roles in **Model Risk Management**, **Enterprise Risk Analytics**, and **Model Validation**.

---

## Problem Statement

Banks need to assess whether a loan applicant is likely to default. This is a binary classification problem:
- **0 → Good** — applicant is likely to repay
- **1 → Bad** — applicant is likely to default

Recall on the "bad" class is prioritized over accuracy — missing a defaulter costs far more than a false alarm.

---

## Dataset

- **Source:** [German Credit Data — Kaggle](https://www.kaggle.com/datasets/uciml/german-credit)
- **Size:** 1000 rows, 10 features
- **Class distribution:** 700 good, 300 bad (imbalanced)

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Applicant age |
| Sex | Categorical | male / female |
| Job | Numeric | Job skill level (0–3) |
| Housing | Categorical | own / free / rent |
| Saving accounts | Categorical | little / moderate / quite rich / rich |
| Checking account | Categorical | little / moderate / rich |
| Credit amount | Numeric | Loan amount in DM |
| Duration | Numeric | Loan duration in months |
| Purpose | Categorical | car / furniture / radio/TV / etc. |

---

## Project Pipeline

```
Raw Data
   ↓
Null Imputation (mode for categorical)
   ↓
Train-Test Split (80/20, stratified)
   ↓
Encoding (LabelEncoder for Sex, OneHotEncoder for categorical cols)
   ↓
SMOTE (on train set only — no leakage)
   ↓
Model Training (5 models)
   ↓
Evaluation (F1, Recall, AUC, Confusion Matrix)
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Final Model: AdaBoost + XGBoost
```

---

## Key Design Decisions

**Why SMOTE after train-test split?**
Applying SMOTE before splitting would allow synthetic samples to leak into the test set, inflating evaluation metrics. SMOTE is applied only on training data to preserve the integrity of the test set — a critical requirement in model validation.

**Why Recall over Accuracy?**
A model with 70% accuracy that never predicts "bad" is useless in credit risk. Recall on the bad class measures how many actual defaulters the model catches — the metric that matters to a bank.

**Why `predict_proba` for AUC?**
AUC must be computed using predicted probabilities, not binary class labels. Using `predict` instead of `predict_proba` gives a misleading AUC score.

---

## Results

| Model | Accuracy | F1 | Recall (bad) | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.575 | 0.422 | 0.517 | 0.569 |
| KNN | 0.660 | 0.209 | 0.150 | 0.543 |
| Decision Tree | 0.625 | 0.370 | 0.367 | 0.587 |
| Random Forest | 0.585 | 0.394 | 0.450 | 0.596 |
| **AdaBoost** | 0.565 | **0.453** | **0.600** | **0.593** |

**Winner: AdaBoost** — highest recall (0.60) meaning it correctly identifies 60% of actual defaulters.

> KNN had the highest accuracy (0.66) but the worst recall (0.15) — a classic example of why accuracy is a misleading metric on imbalanced datasets.

---

## Tech Stack

```
Python 3.x
pandas, numpy
scikit-learn
imbalanced-learn (SMOTE)
xgboost
matplotlib, seaborn
```

---

## Installation

```bash
git clone https://github.com/yourusername/german-credit-risk
cd german-credit-risk
pip install -r requirements.txt
jupyter notebook German_Credit_Risk.ipynb
```

**requirements.txt**
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
jupyter
```

---

## Files

```
├── German_Credit_Risk.ipynb   # Main notebook
├── german_credit_data.csv     # Dataset
├── smote_comparison.png       # Class distribution visualization
├── requirements.txt
└── README.md
```

---

## Concepts Demonstrated

- Handling class imbalance with SMOTE (correctly after train-test split)
- Feature encoding — LabelEncoder vs OneHotEncoder
- Model comparison across 5 classifiers
- Evaluation metrics beyond accuracy: F1, Recall, AUC
- Hyperparameter tuning with GridSearchCV
- Avoiding data leakage in cross-validation
- Interpreting confusion matrices for credit risk context

---

## Author

Built as a portfolio project for Enterprise Risk Analytics / Model Validation roles in banking.
