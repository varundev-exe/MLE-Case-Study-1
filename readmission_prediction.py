"""
Hospital Readmission Prediction — L2-Regularized Logistic Regression
======================================================================
Predicts 30-day readmission risk from diagnosis codes, vitals, and
prior visit history. Evaluates with ROC-AUC / PR-AUC and analyzes the
clinical cost trade-off between false negatives and false positives.

NOTE: This script uses a SYNTHETIC dataset generator so it runs
out-of-the-box. To use real data (e.g. the UCI "Diabetes 130-US
hospitals" dataset), replace `generate_synthetic_data()` with a
`pd.read_csv(...)` call and adjust FEATURE_COLUMNS accordingly.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
def generate_synthetic_data(n_samples=5000, random_state=RANDOM_STATE):
    """Synthetic patient-level data mimicking readmission risk factors."""
    rng = np.random.default_rng(random_state)

    age = rng.normal(65, 15, n_samples).clip(18, 100)
    num_diagnoses = rng.poisson(3, n_samples)
    num_prior_visits = rng.poisson(1.2, n_samples)
    length_of_stay = rng.gamma(2, 2, n_samples)
    systolic_bp = rng.normal(130, 20, n_samples)
    heart_rate = rng.normal(80, 15, n_samples)
    glucose = rng.normal(120, 40, n_samples)
    bmi = rng.normal(28, 6, n_samples)
    has_diabetes = rng.binomial(1, 0.3, n_samples)
    has_heart_failure = rng.binomial(1, 0.2, n_samples)
    discharge_home = rng.binomial(1, 0.7, n_samples)

    # True underlying risk (logit) — drives label generation
    logit = (
        -5
        + 0.03 * age
        + 0.35 * num_diagnoses
        + 0.55 * num_prior_visits
        + 0.15 * length_of_stay
        + 0.01 * (glucose - 120)
        + 0.02 * (systolic_bp - 130)
        + 0.9 * has_heart_failure
        + 0.5 * has_diabetes
        - 0.6 * discharge_home
    )
    prob = 1 / (1 + np.exp(-logit))
    readmitted = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "num_diagnoses": num_diagnoses,
        "num_prior_visits": num_prior_visits,
        "length_of_stay": length_of_stay,
        "systolic_bp": systolic_bp,
        "heart_rate": heart_rate,
        "glucose": glucose,
        "bmi": bmi,
        "has_diabetes": has_diabetes,
        "has_heart_failure": has_heart_failure,
        "discharge_home": discharge_home,
        "readmitted_30d": readmitted,
    })
    return df


FEATURE_COLUMNS = [
    "age", "num_diagnoses", "num_prior_visits", "length_of_stay",
    "systolic_bp", "heart_rate", "glucose", "bmi",
    "has_diabetes", "has_heart_failure", "discharge_home",
]
TARGET_COLUMN = "readmitted_30d"


# ---------------------------------------------------------------------
# 2. Train / tune model
# ---------------------------------------------------------------------
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",   # handles class imbalance
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best C (inverse of L2 strength lambda): {grid.best_params_['clf__C']}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


# ---------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nTest ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC : {pr_auc:.4f}")

    # Default 0.5 threshold report
    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\n--- Classification report @ threshold = 0.5 ---")
    print(classification_report(y_test, y_pred_default, digits=3))

    return y_proba, roc_auc, pr_auc


# ---------------------------------------------------------------------
# 4. Clinical cost-based threshold selection
# ---------------------------------------------------------------------
def cost_based_threshold(y_test, y_proba, cost_fn=100, cost_fp=1):
    """
    Sweep thresholds and pick the one minimizing total clinical cost.

    cost_fn: cost of a missed readmission (patient harm, penalty, etc.)
    cost_fp: cost of an unnecessary intervention (staff time, outreach)

    Default ratio (10:1) reflects that missing a readmission is far
    costlier than an unneeded follow-up call — adjust to your setting
    (published estimates often justify ratios anywhere from 5:1 to 50:1).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)

    best_idx = int(np.argmin(costs))
    best_threshold = thresholds[best_idx]

    print(f"\n--- Cost-based threshold optimization ---")
    print(f"Assumed cost ratio  FN:FP = {cost_fn}:{cost_fp}")
    print(f"Optimal threshold          = {best_threshold:.2f}")
    print(f"Minimum expected cost      = {costs[best_idx]:.0f}")

    y_pred_opt = (y_proba >= best_threshold).astype(int)
    print("\n--- Classification report @ cost-optimal threshold ---")
    print(classification_report(y_test, y_pred_opt, digits=3))

    return best_threshold, thresholds, costs


# ---------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------
def plot_curves(y_test, y_proba, save_path="model_evaluation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_test, y_proba):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    axes[2].hist(y_proba[y_test == 0], bins=30, alpha=0.6, label="Not readmitted")
    axes[2].hist(y_proba[y_test == 1], bins=30, alpha=0.6, label="Readmitted")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_title("Predicted Risk Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved evaluation plots to {save_path}")


# ---------------------------------------------------------------------
# 6. Feature importance (interpretability — key for clinical trust)
# ---------------------------------------------------------------------
def show_feature_importance(model):
    coefs = model.named_steps["clf"].coef_[0]
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\n--- Feature Importance (standardized coefficients) ---")
    print(importance.to_string(index=False))
    return importance


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Readmission rate: {df[TARGET_COLUMN].mean():.2%}\n")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)
    y_proba, roc_auc, pr_auc = evaluate_model(model, X_test, y_test)
    cost_based_threshold(y_test, y_proba, cost_fn=10, cost_fp=1)
    show_feature_importance(model)
    plot_curves(y_test, y_proba)
