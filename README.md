# German Credit Risk Predictor

A binary classification project that predicts whether a loan applicant is a **good** or **bad** credit risk, using the German Credit dataset (1,000 applicants, 9 features). The project covers data cleaning, exploratory analysis, class-imbalance handling with SMOTE, and a comparison of six tuned classifiers.

## Problem Statement

Lenders lose more by approving a defaulter than by rejecting a good applicant. The goal is to build a model that flags high-risk applicants, so **recall on the "bad" class** is treated as the key metric alongside ROC-AUC.

## Dataset

| Feature | Description |
|---|---|
| Age | Applicant age (19–75) |
| Sex | male / female |
| Job | 0 = unskilled non-resident, 1 = unskilled resident, 2 = skilled, 3 = highly skilled |
| Housing | own / rent / free |
| Saving accounts | little / moderate / quite rich / rich (18% missing) |
| Checking account | little / moderate / rich (39% missing) |
| Credit amount | Loan amount |
| Duration | Loan duration in months |
| Purpose | car, radio/TV, furniture, business, education, etc. |
| **Risk** | Target: good (70%) / bad (30%) |

## Workflow

1. **Cleaning**: imputed missing `Saving accounts` and `Checking account` values with the mode.
2. **EDA**: distributions of age, job level, loan purpose, credit amount and duration; correlation heatmap; outlier check with box plots.
3. **Preprocessing**: stratified 80/20 train-test split, then a `ColumnTransformer` (one-hot encoding for categoricals, standard scaling for numerics).
4. **Imbalance handling**: applied **SMOTE** to the training set only (560 good / 240 bad → 560 / 560).
5. **Modelling**: Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost and XGBoost, first with defaults, then tuned with `RandomizedSearchCV` (5-fold stratified CV, ROC-AUC scoring).
6. **Evaluation**: accuracy, F1, recall, confusion matrix and ROC-AUC on the untouched test set.

## Key EDA Findings

- Applicants aged roughly 25–35 take the most loans.
- Skilled workers (Job = 2) make up the largest share of borrowers; highly skilled workers take loans with the longest durations.
- Car purchases are the most common loan purpose.
- The dataset is imbalanced 70:30, which makes plain accuracy misleading.

## Results (Test Set, 200 samples)

| Model | CV AUC | Test AUC | Test Recall (bad) | Test F1 (bad) | Accuracy |
|---|---|---|---|---|---|
| AdaBoost | 0.799 | **0.668** | **0.533** | **0.504** | 0.685 |
| XGBoost | 0.849 | 0.658 | 0.383 | 0.414 | 0.675 |
| Random Forest | 0.864 | 0.652 | 0.417 | 0.455 | 0.700 |
| Logistic Regression | 0.685 | 0.627 | 0.517 | 0.431 | 0.590 |
| Decision Tree | 0.755 | 0.589 | 0.400 | 0.397 | 0.635 |
| KNN | 0.842 | 0.573 | 0.467 | 0.412 | 0.600 |

**Best model: AdaBoost** (300 estimators, learning rate 1.0), which gave the best test AUC and the best recall on risky applicants.

## Limitations and Honest Takeaways

- **CV vs test gap.** Cross-validation AUCs (up to 0.86) are much higher than test AUCs (~0.57–0.67). SMOTE was applied to the full training set *before* cross-validation, so synthetic neighbours of a sample can land in the validation fold and inflate CV scores. The test-set numbers are the trustworthy ones.
- **Overfitting.** Untuned Random Forest and XGBoost reached ~100% training accuracy but only ~69–71% on the test set.
- **Small data.** With 1,000 rows and a 200-row test set (only 60 "bad" cases), metrics are noisy and a single split can shift them noticeably.
- **Imputation.** Filling ~39% of `Checking account` with the mode may hide the signal that "missing" itself can carry.

## Planned Improvements

- Move SMOTE inside an `imblearn` `Pipeline` so it is fit only on training folds during CV.
- Treat missing account values as their own category instead of mode-imputing.
- Tune the decision threshold for recall using a cost-sensitive view (a missed defaulter costs more than a rejected good applicant).
- Add SHAP-based feature importance for explainability.
- Use repeated stratified CV to get stable estimates on a small dataset.

## Tech Stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, XGBoost

## How to Run

```bash
git clone <your-repo-url>
cd german-credit-risk
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
jupyter notebook German_Credit_Risk.ipynb
```

> Update the dataset path in the notebook (`pd.read_csv(...)`) to point to `german_credit_data.csv`.

## Project Structure

```
├── German_Credit_Risk.ipynb
├── german_credit_data.csv
├── smote_comparison.png
└── README.md
```
