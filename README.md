# Responsible AI on COMPAS --- Fairness Baseline

## Overview

This repo builds a **reproducible baseline** for fairness analysis on
the COMPAS dataset.\
It performs EDA, trains an interpretable model, evaluates performance
**overall and by sensitive groups**, and exports tables and plots for a
short report and GitHub Pages.

## Goals

-   Load and audit the dataset for imbalance and label issues.\
-   Train a baseline **Logistic Regression** to predict `is_recid`.\
-   Compute **group metrics** by `race` and `sex` and core **fairness
    metrics**.\
-   Provide artifacts for later **mitigation** (reweighting, resampling,
    thresholds).

## Dataset

-   Source: Kaggle "COMPAS" (ProPublica)\
-   File used: `/mnt/data/cox-violent-parsed_filt.csv` (columns
    incl. `race`, `sex`, `age`, `priors_count`, `decile_score`,
    `is_recid`)\
-   Sensitive attributes: `race`, `sex`\
-   Target: `is_recid ∈ {0,1}`

## Repository Structure

    .
    ├─ proyecto1-rai.ipynb              # Final Jupyter notebook (end-to-end)
    ├─ reports/
    │  ├─ metrics_overall.csv
    │  ├─ metrics_by_group_race.csv
    │  ├─ metrics_by_group_race_th_0.3.csv
    │  ├─ metrics_by_group_race_th_0.5.csv
    │  ├─ metrics_by_group_race_th_0.7.csv
    │  ├─ metrics_by_group_sex.csv
    │  ├─ metrics_by_group_sex_th_0.3.csv
    │  ├─ metrics_by_group_sex_th_0.5.csv
    │  ├─ metrics_by_group_sex_th_0.7.csv
    │  ├─ base_logreg.joblib            # Saved baseline model
    │  └─ figures/
    │     ├─ dist_race.png
    │     ├─ dist_sex.png
    │     ├─ target_rate_by_race.png
    │     ├─ target_rate_by_sex.png
    │     ├─ ppr_by_race.png
    │     ├─ ppr_by_sex.png
    │     ├─ tpr_by_race.png
    │     └─ tpr_by_sex.png
    └─ README.md

## Environment

``` bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## How to Run

1.  Open `proyecto1-rai.ipynb` and run all cells top-to-bottom.\
2.  Ensure the dataset path in the notebook points to your local copy of
    `cox-violent-parsed_filt.csv`.\
3.  Artifacts will be written under `reports/` and `reports/figures/`.

## What the Notebook Does

-   **EDA:** group counts and percentages, class balance, and target
    rate plots by `race` and `sex`.\
-   **Preprocessing:** numeric scaling, categorical one-hot encoding,
    imputers.\
-   **Model:** Logistic Regression (`class_weight="balanced"`).\
-   **Evaluation:** accuracy, precision, recall, F1, ROC-AUC; plus
    **group reports** for:
    -   **PPR** (Demographic Parity): ( P(`\hat{Y}`{=tex}=1) )\
    -   **TPR** (Equal Opportunity): ( `\frac{TP}{TP+FN}`{=tex} )\
    -   **FPR, FNR, PPV, TNR, BACC**, diffs vs reference, **Disparate
        Impact**\
-   **Threshold sweep:** optional reports at 0.3, 0.5, 0.7.

## Key Results (baseline)

-   Overall: \~0.66 accuracy, \~0.71 ROC-AUC (logistic baseline).\
-   Disparities observed in **PPR** and **TPR** across race and sex;
    **DI \< 0.8** for several groups.

## Reproducibility Tips

-   Set `random_state=42` in splits.\
-   Keep paths and feature lists as constants near the top of the
    notebook.\
-   Commit the generated CSVs and figures for transparency.

## Next Steps (Mitigation, optional)

-   **Reweighting / class_weight by group**\
-   **Resampling** (oversample under-represented groups)\
-   **Per-group thresholds** to equalize TPR/FPR trade-offs\
    Re-run the same reports to produce a **before vs after** comparison.

## GitHub Pages

Publish a simple results page that: - Links the notebook and data
instructions.\
- Embeds key plots from `reports/figures/`.\
- Summarizes overall metrics and 2--3 fairness findings.\
- Notes limitations and ethical considerations.

## License and Credits

-   Data original credit: ProPublica / COMPAS dataset on Kaggle.\
-   Code: MIT (or your choice).
