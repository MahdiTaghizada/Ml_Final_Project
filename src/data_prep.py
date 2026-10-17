"""
data_prep.py - loading, cleaning, feature engineering, splitting, tier label.

Everything that touches the raw bina.az dump lives here; the rest of the code
only calls these functions.  Design rules (all graded in the project):

  * Every cleaning rule is a FIXED, domain-based rule (not fitted on data),
    and every rule logs how many rows it removed  -> `clean(...).attrs["cleaning_log"]`.
  * Anything that is a fitted statistic (imputation medians, scaling mean/std,
    the price-tier threshold) is computed from the TRAIN split only.
  * Target-derived columns (unit_price, total_price) are removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"   # the only place a data path is set

# Columns that LEAK the target (arithmetic functions of `price`).
# Verified on the data: unit_price * area / price ~= 1.000 and
# total_price == price for 99.99 % of rows.
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Raw columns we read (the free-text `description`, image URLs etc. are skipped).
USE_COLS = [
    "estate_rel_url_x", "datetime_scrape_x", "price", "currency_x", "currency_y",
    "unit_price", "total_price", "location", "city", "owner_title",
    "lat", "lng", "Kateqoriya", "Mərtəbə", "Otaq sayı", "Sahə",
    "Torpaq sahəsi", "Təmir", "Çıxarış", "İpoteka",
]

# Why the other raw columns are not used (reported in the paper).
DROPPED_REASONS = {
    "unit_price / total_price": "target leakage (price per m2 / price restated)",
    "id_*, estate_*, rel_url, img_url, owner_name, shop_*, address": "identifiers / PII",
    "views": "post-listing popularity, unknown when the price is set",
    "vip, featured, products_label": "paid promotion flags set by the seller after pricing",
    "datetime_*, day_*, hour_*, updated, city_when": "scrape/listing timestamps (single 2-week window)",
    "Binanin novu": "99.8 % missing",
    "repair, bill_of_sale, mortgage": "exact duplicates of Temir, Cixaris, Ipoteka",
    "city": "99.7 % Baku (information is already in lat/lng)",
    "description, attributes, extra_info": "free text; attributes re-parsed from clean columns",
}

CATEGORY_EN = {
    "Yeni tikili": "new_build",
    "Köhnə tikili": "old_stock",
    "Həyət evi/Bağ evi": "house_villa",
}
RESIDENTIAL = list(CATEGORY_EN)

# --- Fixed cleaning rules (domain knowledge, documented in the report) -------
AREA_RANGE = (15.0, 1000.0)             # m2
PRICE_RANGE = (5_000.0, 10_000_000.0)   # AZN
PPM_RANGE = (200.0, 15_000.0)           # AZN per m2
ROOMS_MAX = 10                          # winsorise (a few 11-20 room listings)
AZ_BBOX = (38.3, 41.95, 44.7, 50.9)     # lat_min, lat_max, lng_min, lng_max
BAKU_CENTER = (40.4093, 49.8671)        # (lat, lng), approx. city centre

FEATURE_NAMES = [
    "area_m2", "rooms", "floor", "total_floors", "floor_ratio",
    "is_first_floor", "is_last_floor", "land_sot", "lat", "lng", "dist_center_km",
    "repaired", "has_mortgage", "has_bill_of_sale", "is_owner",
    "cat_new_build", "cat_old_stock", "cat_house_villa",
    "loc_metro", "loc_district", "loc_settlement", "loc_other",
    "rooms_missing", "floor_missing", "repaired_missing",
]
# right-skewed columns that get log1p before scaling for the SVM
LOG_FEATURES = ["area_m2", "land_sot", "dist_center_km"]


# ---------------------------------------------------------------------------
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (UTF-8) and return it unmodified (selected columns only)."""
    return pd.read_csv(path, usecols=USE_COLS, encoding="utf-8")


def _num(series: pd.Series) -> pd.Series:
    """First number in a string like '145 m²' / '1.3 sot' -> float."""
    return pd.to_numeric(series.astype("string").str.extract(r"([\d]+(?:\.\d+)?)")[0], errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame and return a tidy frame with English column names.

    Steps (each logs the number of rows it removes):
      1. one row per listing (the dump has repeated scrapes) - keep the latest scrape;
      2. keep the residential market (new build, old stock, house/villa); land plots,
         garages, offices and commercial objects have a different price structure;
      3. parse area / floors / land area from strings, unit-check m2 vs sot;
      4. fixed plausibility bounds on area, price and price per m2;
      5. drop coordinates outside Azerbaijan;
      6. drop near-duplicates (same listing re-posted under a new URL);
      7. drop leakage + identifier + promotion columns.
    """
    log: list[tuple[str, int, int]] = []

    def step(name, before, after):
        log.append((name, int(before), int(after)))

    d = df.copy()
    step("raw rows", len(d), len(d))

    # 1. repeated scrapes of the same listing ------------------------------
    n0 = len(d)
    d = d.sort_values("datetime_scrape_x", kind="stable")
    d = d.drop_duplicates(subset="estate_rel_url_x", keep="last")
    step("duplicate scrapes of one listing (kept latest)", n0, len(d))

    # 2. residential scope ---------------------------------------------------
    n0 = len(d)
    d = d[d["Kateqoriya"].isin(RESIDENTIAL)].copy()
    step("non-residential categories (land, garage, office, commercial)", n0, len(d))

    # 3. parsing ---------------------------------------------------------------
    d["area_m2"] = _num(d["Sahə"])
    unit = d["Sahə"].astype("string").str.extract(r"([^\d.\s]+\S*)\s*$")[0]
    assert (unit.dropna() == "m²").all(), "residential area must be in m2"
    d["land_sot"] = _num(d["Torpaq sahəsi"])           # NaN for flats
    fl = d["Mərtəbə"].astype("string").str.extract(r"^(\d+)\s*/\s*(\d+)$").astype(float)
    d["floor"], d["total_floors"] = fl[0], fl[1]
    bad_floor = (d["floor"] > d["total_floors"]) | (d["total_floors"] > 50) | (d["floor"] < 1)
    d.loc[bad_floor, ["floor", "total_floors"]] = np.nan   # impossible -> missing
    d["rooms"] = pd.to_numeric(d["Otaq sayı"], errors="coerce").clip(upper=ROOMS_MAX)

    # 4. plausibility bounds ---------------------------------------------------
    n0 = len(d); d = d[d["area_m2"].between(*AREA_RANGE)]
    step(f"area outside {AREA_RANGE} m2 (or unparsable)", n0, len(d))
    n0 = len(d); d = d[d["price"].between(*PRICE_RANGE)]
    step(f"price outside {PRICE_RANGE} AZN", n0, len(d))
    n0 = len(d); ppm = d["price"] / d["area_m2"]; d = d[ppm.between(*PPM_RANGE)]
    step(f"price/m2 outside {PPM_RANGE}", n0, len(d))

    # 5. coordinates -------------------------------------------------------------
    n0 = len(d)
    la0, la1, lo0, lo1 = AZ_BBOX
    d = d[d["lat"].between(la0, la1) & d["lng"].between(lo0, lo1)]
    step("coordinates outside Azerbaijan", n0, len(d))

    # 6. near-duplicates (re-posted under a new URL) -------------------------------
    n0 = len(d)
    key = ["location", "Kateqoriya", "area_m2", "rooms", "floor", "total_floors",
           "price", "lat", "lng"]
    d = d.drop_duplicates(subset=key, keep="last")
    step("near-duplicate re-posts", n0, len(d))

    # 7. standard names / encodings --------------------------------------------------
    out = pd.DataFrame(index=d.index)
    out["price"] = d["price"].astype(float)
    out["area_m2"] = d["area_m2"]
    out["rooms"] = d["rooms"]
    out["floor"] = d["floor"]
    out["total_floors"] = d["total_floors"]
    out["land_sot"] = d["land_sot"]
    out["lat"], out["lng"] = d["lat"], d["lng"]
    out["location"] = d["location"].astype("string")
    out["category"] = d["Kateqoriya"].map(CATEGORY_EN).astype("string")
    out["repaired"] = d["Təmir"].map({"var": 1.0, "yoxdur": 0.0})            # NaN stays NaN
    out["has_mortgage"] = (d["İpoteka"].astype("string") == "var").fillna(False).astype(float)  # blank = not offered -> 0
    out["has_bill_of_sale"] = d["Çıxarış"].map({"var": 1.0, "yoxdur": 0.0})
    out["is_owner"] = d["owner_title"].map({"mülkiyyətçi": 1.0, "vasitəçi (agent)": 0.0})
    out = out.reset_index(drop=True)
    # guards against silent encoding bugs: this flag must be a clean 0/1 column
    assert out["has_mortgage"].isin([0.0, 1.0]).all() and out["has_mortgage"].nunique() == 2
    for c in LEAKAGE_COLUMNS:
        assert c not in out.columns
    out.attrs["cleaning_log"] = log
    return out


# ---------------------------------------------------------------------------
def _loc_type(loc: pd.Series) -> pd.DataFrame:
    """District-name suffix: m.=metro area, r.=district, q.=settlement."""
    suf = loc.astype("string").str.extract(r"\s(\w+)\.$")[0]
    return pd.DataFrame({
        "loc_metro": (suf == "m").astype(float),
        "loc_district": (suf == "r").astype(float),
        "loc_settlement": (suf == "q").astype(float),
        "loc_other": (~suf.isin(["m", "r", "q"])).astype(float),
    })


def _dist_center_km(lat, lng):
    """Flat-earth distance (km) to the Baku centre - fine at city scale."""
    dy = (lat - BAKU_CENTER[0]) * 111.2
    dx = (lng - BAKU_CENTER[1]) * 111.2 * np.cos(np.radians(BAKU_CENTER[0]))
    return np.sqrt(dx ** 2 + dy ** 2)


def make_features(df: pd.DataFrame):
    """
    Cleaned frame -> numeric design matrix (stateless, row-wise features only).

    Missing values are NOT filled here (the fill value is a train statistic);
    instead `rooms_missing`, `floor_missing`, `repaired_missing` flags are added
    and NaNs are filled later by `Imputer` fitted on the train split.
    Returns (X [n,d] float array with NaN, y = price [n], feature_names).
    """
    f = pd.DataFrame(index=df.index)
    f["area_m2"] = df["area_m2"]
    f["rooms"] = df["rooms"]
    f["floor"] = df["floor"]
    f["total_floors"] = df["total_floors"]
    f["floor_ratio"] = df["floor"] / df["total_floors"]
    f["is_first_floor"] = (df["floor"] == 1).astype(float).where(df["floor"].notna())
    f["is_last_floor"] = (df["floor"] == df["total_floors"]).astype(float).where(df["floor"].notna())
    f["land_sot"] = df["land_sot"].fillna(0.0)          # 0 = no land plot (flat)
    f["lat"], f["lng"] = df["lat"], df["lng"]
    f["dist_center_km"] = _dist_center_km(df["lat"], df["lng"])
    f["repaired"] = df["repaired"]
    f["has_mortgage"] = df["has_mortgage"]
    f["has_bill_of_sale"] = df["has_bill_of_sale"]
    f["is_owner"] = df["is_owner"]
    for c in ["new_build", "old_stock", "house_villa"]:
        f[f"cat_{c}"] = (df["category"] == c).astype(float)
    f = pd.concat([f, _loc_type(df["location"])], axis=1)
    f["rooms_missing"] = df["rooms"].isna().astype(float)
    f["floor_missing"] = df["floor"].isna().astype(float)
    f["repaired_missing"] = df["repaired"].isna().astype(float)
    assert list(f.columns) == FEATURE_NAMES, "FEATURE_NAMES out of sync"
    return f.to_numpy(dtype=float), df["price"].to_numpy(dtype=float), list(f.columns)


# ---------------------------------------------------------------------------
def make_tier_label(y_price, threshold=None):
    """
    Task B target: premium (1, price > threshold) vs standard (0).
    If `threshold` is None it is the MEDIAN of `y_price`, so call this first on
    the TRAIN prices and then reuse the returned threshold for val/test.
    Returns (y_tier, threshold).
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def split_indices(n, val_size=0.15, test_size=0.15, seed=SEED, strata=None):
    """
    Deterministic (optionally stratified) train/val/test split -> three index arrays.
    `strata`: integer group id per row; every group is divided in the same
    proportions.  We pass decile bins of log-price so that train, val and test
    each contain the same share of cheap and expensive listings; the tier
    THRESHOLD is still computed from the train prices only.
    """
    rng = np.random.default_rng(seed)
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(strata == g) for g in np.unique(strata)]
    tr, va, te = [], [], []
    for idx in groups:
        idx = rng.permutation(idx)
        n_te = int(round(len(idx) * test_size))
        n_va = int(round(len(idx) * val_size))
        te.append(idx[:n_te]); va.append(idx[n_te:n_te + n_va]); tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED, strata=None):
    """Return (X_tr, y_tr, X_val, y_val, X_te, y_te)."""
    tr, va, te = split_indices(len(y), val_size, test_size, seed, strata)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


# ---------------------------------------------------------------------------
class Imputer:
    """Median imputation with medians learned on the TRAIN split only."""

    def fit(self, X):
        self.medians_ = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        r, c = np.where(np.isnan(X))
        X[r, c] = self.medians_[c]
        return X


class Standardizer:
    """Zero-mean / unit-variance scaling with TRAIN statistics only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def impute(X_tr, *others):
    """Fill NaNs with TRAIN medians; returns arrays in the order given."""
    imp = Imputer().fit(X_tr)
    return (imp.transform(X_tr), *[imp.transform(o) for o in others])


def standardize(X_tr, *others):
    """Scale every array with the mean/std of X_tr; returns arrays in the order given."""
    sc = Standardizer().fit(X_tr)
    return (sc.transform(X_tr), *[sc.transform(o) for o in others])


def log_transform(X, feature_names, cols=LOG_FEATURES):
    """log1p on right-skewed columns (used for the SVM, which is scale-sensitive)."""
    X = np.array(X, dtype=float, copy=True)
    for c in cols:
        j = feature_names.index(c)
        X[:, j] = np.log1p(X[:, j])
    return X
