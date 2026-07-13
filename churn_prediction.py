"""
Customer Churn Prediction
===========================
Task 4 (Intermediate Project): Predict which customers are likely to
cancel their subscription, using demographics, usage, and engagement
features. Covers feature engineering, model selection, cross-validation,
and ROC/AUC evaluation.

Dataset: telecom_customer_churn.csv (7,043 customers, 38 columns) -
the IBM/Kaggle "Telco Customer Churn" dataset.

Run:
    python churn_prediction.py

Outputs:
    - Printed EDA / cleaning / training / evaluation results
    - figures/ - churn rate charts, correlation heatmap, ROC curves,
      confusion matrix, feature importance
    - churn_model.pkl - best trained pipeline (preprocessing + model)
    - analysis_summary.md
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
RANDOM_STATE = 42


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------
print("=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

df = pd.read_csv("telecom_customer_churn.csv")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nCustomer Status distribution:")
print(df["Customer Status"].value_counts())

# ---------------------------------------------------------------------
# Step 2: Define the prediction target
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: DEFINE TARGET")
print("=" * 70)
# 'Customer Status' has 3 values: Stayed / Churned / Joined.
# "Joined" customers are brand new this quarter and haven't had a
# chance to churn yet, so their usage history is incomplete/misleading
# for a churn model. We drop them and predict Churned (1) vs Stayed (0)
# for customers with a full observation period - this is a deliberate
# scoping decision, not missing-data cleanup.
df = df[df["Customer Status"] != "Joined"].copy()
df["Churn"] = (df["Customer Status"] == "Churned").astype(int)
print(f"Rows after removing 'Joined' customers: {len(df)}")
print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")

# ---------------------------------------------------------------------
# Step 3: Drop leakage / non-predictive identifier columns
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: DROP LEAKAGE & IDENTIFIER COLUMNS")
print("=" * 70)
# Churn Category / Churn Reason are only populated AFTER a customer
# churns -> direct leakage of the target, must be dropped.
# Customer ID / City / Zip / Lat / Long are identifiers, not
# generalizable behavioral signals, so we drop them for this model.
# Customer Status is the raw column the target was derived from.
leak_and_id_cols = [
    "Customer ID", "Customer Status", "Churn Category", "Churn Reason",
    "City", "Zip Code", "Latitude", "Longitude"
]
df.drop(columns=leak_and_id_cols, inplace=True)
print(f"Dropped: {leak_and_id_cols}")
print(f"Remaining columns: {df.shape[1]}")

# ---------------------------------------------------------------------
# Step 4: Handle missing values (structural, not random)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: HANDLE MISSING VALUES")
print("=" * 70)
# Most missingness here is structural: e.g. "Internet Type" is NaN
# exactly when "Internet Service" == "No" (no internet -> no internet
# type). Same pattern for "Multiple Lines" when Phone Service == "No".
# We fill these with an explicit "No Internet Service" / "No Phone
# Service" category rather than imputing something that isn't true.
internet_dependent_cols = [
    "Internet Type", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support", "Streaming TV",
    "Streaming Movies", "Streaming Music", "Unlimited Data"
]
for c in internet_dependent_cols:
    df[c] = df[c].fillna("No Internet Service")

df["Multiple Lines"] = df["Multiple Lines"].fillna("No Phone Service")
df["Offer"] = df["Offer"].fillna("No Offer")
df["Avg Monthly GB Download"] = df["Avg Monthly GB Download"].fillna(0)
df["Avg Monthly Long Distance Charges"] = df["Avg Monthly Long Distance Charges"].fillna(0)

remaining_na = df.isnull().sum().sum()
print(f"Remaining missing values after structural imputation: {remaining_na}")
if remaining_na:
    print(df.isnull().sum()[df.isnull().sum() > 0])
    df.dropna(inplace=True)
    print(f"Dropped remaining rows with NA -> {len(df)} rows left")

# ---------------------------------------------------------------------
# Step 5: Feature engineering
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 70)

df["Avg Monthly Spend"] = df["Total Charges"] / df["Tenure in Months"].replace(0, 1)
df["Services Count"] = df[[
    "Phone Service", "Internet Service", "Streaming TV",
    "Streaming Movies", "Streaming Music"
]].apply(lambda row: sum(v == "Yes" for v in row), axis=1)
df["Has Multiple Services"] = (df["Services Count"] >= 3).astype(int)
df["Tenure Years"] = df["Tenure in Months"] / 12

print("Added: Avg Monthly Spend, Services Count, Has Multiple Services, Tenure Years")

target = "Churn"
y = df[target]
X = df.drop(columns=[target])

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# ---------------------------------------------------------------------
# Step 6: EDA visualizations
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: EXPLORATORY VISUALIZATIONS")
print("=" * 70)

plt.figure(figsize=(5, 4))
df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().plot(
    kind="bar", color=["seagreen", "tomato"])
plt.title("Class Balance")
plt.ylabel("Customers")
savefig("01_class_balance.png")

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="Tenure in Months")
plt.xticks([0, 1], ["Stayed", "Churned"])
plt.title("Tenure vs Churn")
savefig("02_tenure_vs_churn.png")

plt.figure(figsize=(7, 5))
churn_by_contract = df.groupby("Contract")["Churn"].mean().sort_values()
churn_by_contract.plot(kind="barh", color="steelblue")
plt.xlabel("Churn rate")
plt.title("Churn Rate by Contract Type")
savefig("03_churn_by_contract.png")

plt.figure(figsize=(8, 6))
corr = df[num_cols + ["Churn"]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap (numeric features)")
savefig("04_correlation_heatmap.png")

# ---------------------------------------------------------------------
# Step 7: Train / test split + preprocessing pipeline
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: TRAIN/TEST SPLIT & PREPROCESSING")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---------------------------------------------------------------------
# Step 8: Model selection with cross-validation
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: MODEL SELECTION (5-fold cross-validation, ROC AUC)")
print("=" * 70)

candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, clf in candidate_models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:22s} ROC AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest model by cross-validated ROC AUC: {best_name}")

plt.figure(figsize=(7, 5))
plt.boxplot(cv_results.values(), tick_labels=cv_results.keys())
plt.ylabel("ROC AUC (5-fold CV)")
plt.title("Model Comparison")
plt.xticks(rotation=15)
savefig("05_model_comparison_cv.png")

# ---------------------------------------------------------------------
# Step 9: Hyperparameter tuning for the best model
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: HYPERPARAMETER TUNING")
print("=" * 70)

best_pipe = Pipeline([("prep", preprocessor), ("clf", candidate_models[best_name])])

param_grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [1, 3, 5]},
    "Gradient Boosting": {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 4]},
}

grid = GridSearchCV(best_pipe, param_grids[best_name], cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV ROC AUC: {grid.best_score_:.3f}")

final_model = grid.best_estimator_

# ---------------------------------------------------------------------
# Step 10: Final evaluation on the held-out test set
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: TEST SET EVALUATION")
print("=" * 70)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")
print(f"ROC AUC  : {auc:.3f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
savefig("06_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"{best_name} (AUC = {auc:.3f})", color="darkorange")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
savefig("07_roc_curve.png")

# ---------------------------------------------------------------------
# Step 11: Feature importance
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11: FEATURE IMPORTANCE")
print("=" * 70)

feature_names = final_model.named_steps["prep"].get_feature_names_out()
clf = final_model.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
elif hasattr(clf, "coef_"):
    importances = np.abs(clf.coef_[0])
else:
    importances = None

if importances is not None:
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    imp_series.sort_values().plot(kind="barh", color="seagreen")
    plt.title(f"Top 15 Feature Importances - {best_name}")
    savefig("08_feature_importance.png")
    print(imp_series)

# ---------------------------------------------------------------------
# Step 12: Save model & summary
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 12: SAVE MODEL & SUMMARY")
print("=" * 70)

joblib.dump(final_model, "churn_model.pkl")
print("Saved trained pipeline -> churn_model.pkl")

cv_table = "\n".join(
    f"| {name} | {scores.mean():.3f} +/- {scores.std():.3f} |"
    for name, scores in cv_results.items()
)

summary = f"""# Customer Churn Prediction - Summary

## Dataset
Telco Customer Churn: {len(df)} customers with a complete observation
period (existing "Joined"/new customers excluded from modeling - they
haven't had a chance to churn yet). Churn rate: {df['Churn'].mean()*100:.1f}%.

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
{cv_table}

6. Hyperparameter tuning (GridSearchCV) on the best model: **{best_name}**
   - Best params: {grid.best_params_}
7. Final evaluation on a held-out 20% test set

## Test set results ({best_name})
- Accuracy : {acc:.3f}
- Precision: {prec:.3f}
- Recall   : {rec:.3f}
- F1-score : {f1:.3f}
- ROC AUC  : {auc:.3f}

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
"""

with open("analysis_summary.md", "w") as f:
    f.write(summary)

print("\nSummary written to analysis_summary.md")
print("DONE.")
