# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Atayi, Rəşad           | @rashadatayi  | EDA, domain cleaning, data leakage elimination, Pegasos Linear SVM & RFF non-linear kernel SVM |
| Orucov, Bəhmən         | @bahmanorujov | Decision Tree implementation & hyperparameter tuning, evaluation metrics suite |
| Tağızadə, Məhdi        | @MahdiTaghizada| Bonus problems (Ridge, Random Forest, K-Means & PCA), cross-validation benchmarks, report compilation |

## Per-member summary
**Atayi, Rəşad —** Designed and implemented the complete data cleaning and leakage elimination pipeline (`src/data_prep.py`), performing exploratory data analysis (EDA), domain cleaning, removing price-derived leakages (`unit_price`, `total_price`), and de-duplicating re-posted listings. Implemented and tuned the from-scratch `PegasosSVM` (`src/svm.py`) with mini-batch subgradient descent, adaptive learning rate schedules, and suffix iterate averaging. Extended the SVM to non-linear spaces using Random Fourier Features (RFF) with 500 components (covering both linear and non-linear SVM variants).

**Orucov, Bəhmən —** Built and debugged the from-scratch pure NumPy `DecisionTree` classifier and regressor (`src/decision_tree.py`) supporting MSE, Gini, and Entropy impurity criteria along with stopping rules. Evaluated hyperparameter tuning, tree depth pruning curves, and feature importances. Developed the comprehensive evaluation metric module (`src/evaluate.py`), including exact RMSE, MAE, R2, F1, ROC-AUC, Average Precision, and percentile bootstrap confidence intervals.

**Tağızadə, Mehdi —** Designed and implemented the three from-scratch bonus extensions in `src/bonuses.py`: closed-form Ridge regression via normal equations, Random Forest bootstrap ensemble with feature subsampling, and geospatial K-Means clustering with PCA. Ran and validated the 5-fold cross-validation benchmarking against scikit-learn references (`src/run_all.py`), organized the unified notebook (`final_project.ipynb`), and contributed to the LaTeX report compilation.

## Rough effort split
All team members contributed equally to the theoretical formulation, implementation, debugging, and verification of the project.

| Member | Approx. % |
|--------|-----------|
| Atayi, Rəşad | 33.3% |
| Orucov, Bəhmən | 33.3% |
| Tağızadə, Mehdi | 33.4% |
