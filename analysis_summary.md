# Customer Churn Prediction - Summary

## Dataset
Telco Customer Churn: 6589 customers with a complete observation
period (existing "Joined"/new customers excluded from modeling - they
haven't had a chance to churn yet). Churn rate: 28.4%.

## Approach
1. Defined target: Churned (1) vs Stayed (0); dropped `Churn
   Category`/`Churn Reason` (leakage - only populated after churn) and
   identifier columns (Customer ID, City, Zip, Lat/Long)
2. Handled structural missing values (e.g. "no internet" -> "No
   Internet Service" category, not random noise)
3. Feature engineering: Avg Monthly Spend, Services Count, Has Multiple
   Services, Tenure Years
4. Preprocessing pipeline: StandardScaler (numeric) + OneHotEncoder
   (categorical) inside a single sklearn Pipeline (no leakage between
   train/test folds)
5. Model selection via 5-fold stratified cross-validation, scored on
   ROC AUC:

| Model | CV ROC AUC |
|---|---|
| Logistic Regression | 0.915 +/- 0.012 |
| Random Forest | 0.925 +/- 0.011 |
| Gradient Boosting | 0.937 +/- 0.006 |

6. Hyperparameter tuning (GridSearchCV) on the best model: **Gradient Boosting**
   - Best params: {'clf__max_depth': 3, 'clf__n_estimators': 100}
7. Final evaluation on a held-out 20% test set

## Test set results (Gradient Boosting)
- Accuracy : 0.860
- Precision: 0.808
- Recall   : 0.666
- F1-score : 0.730
- ROC AUC  : 0.929

## Key drivers of churn
See `figures/08_feature_importance.png` and `figures/03_churn_by_contract.png`
- Month-to-month contracts churn far more than 1/2-year contracts
- Shorter tenure strongly associates with churn
- (see feature importance chart for the full ranked list)

## Files
- `churn_prediction.py` - full pipeline script
- `churn_model.pkl` - saved trained pipeline (preprocessing + model)
- `figures/` - EDA charts, model comparison, ROC curve, confusion
  matrix, feature importance
