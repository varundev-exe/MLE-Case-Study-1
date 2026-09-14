# Hospital Readmission Prediction

L2-regularized logistic regression to predict 30-day hospital readmission
risk from diagnosis codes, vitals, and prior visit history.

## Why logistic regression + L2

- **Interpretable**: coefficients / odds ratios show clinicians *why* a
  patient is flagged, which matters for adoption and trust.
- **L2 (ridge) penalty**: shrinks correlated feature weights (e.g. BP,
  heart rate, glucose often move together) and reduces overfitting when
  many diagnosis-code indicators are included.
- **Fast and well-calibrated** relative to black-box alternatives —
  good baseline before trying tree ensembles or neural nets.

## Data

`generate_synthetic_data()` creates a synthetic cohort with:
age, number of diagnoses, prior visits, length of stay, vitals
(systolic BP, heart rate, glucose, BMI), comorbidity flags
(diabetes, heart failure), and discharge disposition.

To use real data, swap in a dataset such as the
[UCI "Diabetes 130-US hospitals" dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
and update `FEATURE_COLUMNS` to match.

## Pipeline

1. Standardize features (`StandardScaler`)
2. `LogisticRegression(penalty="l2", class_weight="balanced")`
3. Tune `C` (inverse L2 strength) via 5-fold stratified CV, scoring on ROC-AUC
4. Evaluate on a held-out test set: ROC-AUC, PR-AUC, calibration
5. Select decision threshold using a clinical cost model (FN vs FP), not
   the default 0.5

## Running it

```bash
pip install -r requirements.txt
python readmission_prediction.py
```

Outputs:
- Console: CV results, ROC-AUC/PR-AUC, classification reports at both
  the default and cost-optimal thresholds, feature importance table
- `model_evaluation.png`: ROC curve, precision-recall curve, and
  predicted-risk distribution by class

## Clinical cost framing

| Error type | Real-world consequence | Relative cost |
|---|---|---|
| **False Negative** | Missed high-risk patient → preventable readmission, patient harm, CMS/HRRP penalty | High |
| **False Positive** | Unnecessary follow-up call / home visit | Low |

Because FN costs generally far outweigh FP costs, the script sweeps
thresholds and picks the one that **minimizes total expected cost**
under a configurable cost ratio (default 10:1), rather than using the
default 0.5 cutoff. This favors recall (catching more true
readmissions) at the expense of some precision — a defensible
trade-off in a clinical setting, but the ratio should be set with
input from clinical/finance stakeholders, not assumed.

## Metrics reported

- **ROC-AUC** — overall discrimination ability across thresholds
- **PR-AUC** — more informative than ROC-AUC given class imbalance
  (~15–35% positive rate is typical for readmission)
- **Sensitivity/Recall** at the cost-optimal threshold
- **Feature importance** — standardized coefficients and odds ratios,
  for clinical interpretability

## Next steps / extensions

- Calibration curve + Brier score for probability trustworthiness
- Compare against gradient-boosted trees (e.g. XGBoost) as an upper
  bound on discriminative performance
- Fairness audit across demographic subgroups
- External validation on a held-out hospital site
