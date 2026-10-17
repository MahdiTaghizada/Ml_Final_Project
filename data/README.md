# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

## How to place it
1. Download the dataset archive from Kaggle.
2. Extract it so the CSV lives **directly in this folder**, e.g.:
   ```
   data/house_sale.csv
   ```
3. The code expects exactly `data/house_sale.csv` (141 MB, 100,775 rows, UTF-8). If your filename differs, set the path once at the top of
   `src/data_prep.py` (the `DATA_PATH` constant) — do not scatter paths
   across the codebase.

## Notes / gotchas
- Column names are in **Azerbaijani** (e.g. `Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, …). Keep the UTF-8 encoding when you read the file.
- **Leakage warning:** any column derived from the target `price`
  (e.g. a per-m² unit price, or a precomputed total) must be **dropped**
  before training. Name the columns you removed in your report.
- Record the dataset version/date you downloaded, for reproducibility.

## What our pipeline does with it
See `src/data_prep.py` (`clean()` logs every removed row). Short version: one row
per listing URL, residential categories only, fixed plausibility bounds, leakage
columns `unit_price` / `total_price` removed. 100,775 raw rows -> 57,619 modelled rows.
