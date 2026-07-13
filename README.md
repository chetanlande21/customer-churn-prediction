# Customer Churn Prediction (Intermediate Project — Task 4)

Predicts which telecom customers are likely to cancel their
subscription, using demographic, usage, and engagement features. Full
ML pipeline: feature engineering, model selection with cross-
validation, hyperparameter tuning, and ROC/AUC evaluation.

## Dataset

`Datasets/telecom_customer_churn.csv` — the IBM/Kaggle "Telco Customer
Churn" dataset: 7,043 customers, 38 columns (demographics, services
subscribed, contract/billing info, and churn outcome).

## Project structure

```
.
├── Datasets/
│   └── telecom_customer_churn.csv
├── churn_prediction.py       # full pipeline script
├── churn_model.pkl           # saved trained pipeline (best model)
├── requirements.txt
├── analysis_summary.md       # auto-generated results summary
└── figures/                  # EDA + evaluation charts
```

## How to run

```bash
pip install -r requirements.txt
python churn_prediction.py
```

## Pipeline

1. **Define the target** — `Customer Status` has 3 values (Stayed /
   Churned / Joined). "Joined" (brand-new) customers are excluded since
   they haven't had time to churn yet; target = Churned (1) vs Stayed (0)
2. **Drop leakage & identifiers** — `Churn Category`/`Churn Reason` are
   only populated *after* a customer churns (leakage), and
   `Customer ID`/`City`/`Zip`/`Lat`/`Long` are identifiers, not
   generalizable signal
3. **Handle missing values** — most missingness here is structural
   (e.g. `Internet Type` is blank exactly when there's no internet
   service), so it's filled with an explicit "No Internet Service" /
   "No Phone Service" category rather than imputed as random noise
4. **Feature engineering** — Avg Monthly Spend, Services Count, Has
   Multiple Services, Tenure Years
5. **Preprocessing pipeline** — `StandardScaler` (numeric) +
   `OneHotEncoder` (categorical) inside a single sklearn `Pipeline`
   (no leakage between CV folds)
6. **Model selection** — 5-fold stratified cross-validation comparing
   Logistic Regression, Random Forest, and Gradient Boosting, scored
   on ROC AUC
7. **Hyperparameter tuning** — `GridSearchCV` on the winning model
8. **Final evaluation** — accuracy, precision, recall, F1, ROC AUC on
   a held-out 20% test set, plus confusion matrix, ROC curve, and
   feature importance

## Results

Cross-validated ROC AUC (5-fold):

| Model | CV ROC AUC |
|---|---|
| Logistic Regression | 0.915 ± 0.012 |
| Random Forest | 0.925 ± 0.011 |
| **Gradient Boosting** | **0.937 ± 0.006** |

**Best model: Gradient Boosting** (tuned: `max_depth=3`, `n_estimators=100`)

Test set performance:

| Metric | Score |
|---|---|
| Accuracy | 0.860 |
| Precision (Churned) | 0.808 |
| Recall (Churned) | 0.666 |
| F1-score (Churned) | 0.730 |
| ROC AUC | 0.929 |

## Key drivers of churn

Ranked by feature importance (see `figures/08_feature_importance.png`):
1. **Tenure** (months/years) — newer customers churn far more
2. **Month-to-month contract** — much higher churn than 1/2-year
   contracts (see `figures/03_churn_by_contract.png`)
3. **Number of referrals** — engaged/referring customers churn less
4. **Monthly charge** — higher bills associate with higher churn
5. Lack of Online Security / Premium Tech Support add-ons

## Tech used

Python, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV),
pandas, numpy, matplotlib, seaborn
