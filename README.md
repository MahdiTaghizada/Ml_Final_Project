# MLE-AI-201 Machine Learning — Final Project

Two models written **from scratch in NumPy** — a decision tree and a Pegasos SVM —
on the **bina.az** sale-listing dataset, benchmarked against scikit-learn.

- **Task A (regression):** predict `price` (modelled as `log(price)`).
- **Task B (classification):** premium vs standard, i.e. price above the **train-set median**
  (208,000 AZN).

Bonus problems (8–10) are intentionally not attempted.

## Headline results (test set, seed 42; full tables in `results/` and the report)

| Task | Model | Metric | Value |
|---|---|---|---|
| A | Ours: decision tree | R² (log price) / RMSE (AZN) | 0.859 / 133,878 |
| A | sklearn tree (matched) | R² (log price) / RMSE (AZN) | 0.859 / 133,720 |
| A | constant baseline | R² (log price) / RMSE (AZN) | 0.000 / 241,770 |
| B | Ours: decision tree | F1 [95% CI] | 0.923 [0.917, 0.928] |
| B | Ours: Pegasos + RFF (RBF) | F1 [95% CI] | 0.903 [0.897, 0.910] |
| B | Ours: Pegasos linear | F1 [95% CI] | 0.866 [0.859, 0.874] |
| B | sklearn SGDClassifier / SVC-RBF (15k subsample) | F1 | 0.866 / 0.904 |

5-fold CV (train+val, preprocessing re-fitted inside every fold) agrees: tree F1 0.919 ± 0.002,
RBF-SVM 0.904 ± 0.002, linear SVM 0.871 ± 0.002.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1. get the dataset (see data/README.md) and place it at data/house_sale.csv
# 2. run the tests (13 checks incl. comparisons with scikit-learn)
python -m tests.run_tests
# 3. reproduce EVERY number, table and figure (~5 min, CPU only)
python -m src.run_all            # add --quick for a ~2 min smoke test on a thin grid
# 4. build the report
cd report && pdflatex report.tex && pdflatex report.tex
```
`run_all` writes numbers to `results/` (CSV + `results.json`), figures to `report/figures/`
and LaTeX tables to `report/tables/`, which `report.tex` includes. Two independent full runs
gave identical metrics (only wall-clock timings differ).

## Repo layout
```
.
├── src/
│   ├── data_prep.py        # load, clean (logged), features, split, tier label, train-only imputer/scaler
│   ├── decision_tree.py    # DecisionTree: gini / entropy / mse, 4 stopping rules, importances
│   ├── svm.py              # PegasosSVM (+ RandomFourierFeatures, RFFPegasosSVM)
│   ├── evaluate.py         # metrics (RMSE/MAE/R2/P/R/F1/ROC-AUC/PR-AUC), CV, bootstrap CI
│   ├── run_all.py          # ONE command: whole pipeline, sweeps, baselines, figures
│   └── make_report_tables.py  # results -> LaTeX tables (called by run_all)
├── tests/                  # python -m tests.run_tests  (scikit-learn used only here as a reference)
├── report/                 # IEEE report: report.tex, report.pdf, figures/, tables/
├── results/                # generated numbers (CSV/JSON) + run_log.txt
├── data/                   # dataset goes here (not committed)
└── presentation/           # slides
```

## Design decisions worth knowing
- **Leakage:** `unit_price` and `total_price` are dropped (verified: `unit_price × area / price ≈ 1`).
  `views`, `vip`, `featured` are dropped too (set after the price, or paid promotion).
- **Duplicates:** 36,321 of 100,775 rows were repeated scrapes of the same listing; they are removed
  *before* splitting so one listing cannot sit in both train and test.
- **No test-set statistics:** imputation medians, scaling mean/std and the tier threshold come from the
  train split only; CV re-fits them inside each fold. The split is stratified on price deciles
  (these only decide which rows go to which split).
- **Selection rule:** hyper-parameters are chosen on validation only; among configurations within 0.002
  of the best score the simplest one wins. The test set is used once.
- **Pegasos budget:** convergence needs ~1/λ steps, so the epoch count scales as λ shrinks
  (`epochs_linear`, `epochs_rff` in `run_all.py`).
- **Differences from scikit-learn** (tie-breaking between equal-gain splits, float32 internals) are
  quantified in the report (`results/agreement_by_depth.csv`).

## Environment
Developed and tested with Python 3.12.3, numpy 2.4.4, pandas 3.0.2, matplotlib 3.10.8,
scikit-learn 1.8.0 (see `requirements.txt`). scikit-learn is used only for baselines and in tests.

## Still to be completed by the team
`contribution_report.md` (names, per-member work, effort split), the author list, repository link and
AI-assistance paragraph in `report/report.tex` (marked in red as `[...]`), and the slides in `presentation/`.
