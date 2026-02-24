"""
core.py — Constants, data loading/preprocessing, feature engineering.

v2 changes (Priority 2):
  - Cyclical time encoding: hour_sin/cos, dow_sin/cos
  - Public holiday, school holiday, Ramadan flags
  - Updated ALL_FEATS (26 features, was 19)
  - Improved sample weights: raised LOS E/D boundary weights
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.environ.get("CSV_PATH", os.path.join(BASE_DIR, "traffic_los_dataset.csv"))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
PRED_DATE  = pd.Timestamp("2026-02-18")

TARGET_VOLS  = ["vol_car","vol_motorcycle","vol_van","vol_medium_lorry","vol_heavy_lorry","vol_bus"]
PCU_FACTORS  = [1.0, 0.5, 1.5, 2.0, 3.0, 3.0]
LOS_BREAKS   = [0.60, 0.70, 0.80, 0.90, 1.00]
LOS_LABELS   = ["A","B","C","D","E","F"]
LOS_ENCODE   = {g: i for i, g in enumerate(LOS_LABELS)}
WEEKDAYS     = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTHS       = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]
HOURS = [
    "12:00 AM","1:00 AM","2:00 AM","3:00 AM","4:00 AM","5:00 AM",
    "6:00 AM","7:00 AM","8:00 AM","9:00 AM","10:00 AM","11:00 AM",
    "12:00 PM","1:00 PM","2:00 PM","3:00 PM","4:00 PM","5:00 PM",
    "6:00 PM","7:00 PM","8:00 PM","9:00 PM","10:00 PM","11:00 PM",
]
ZERO_MOTO_ROADS   = {"Jalan Pahang", "Jalan Sultan Azlan Shah"}
CLOSED_HOURS_ROAD = "Jalan Sultan Yahya Petra"
CLOSED_HOURS      = set(range(10))
SHEET_ORDER = [
    "Jalan Ampang","Jalan Cheras","Jalan Istana","Jalan Kuching",
    "Jalan Maharajalela","Jalan Pahang","Jalan Pudu",
    "Jalan Sultan Azlan Shah","Jalan Sultan Yahya Petra","Jalan Syed Putra",
    "Jalan Travers","Jalan Tuanku Abdul Halim","Jalan Tun Razak",
    "Lebuhraya KL-Seremban",
]

# Malaysian gazetted public holidays (month, day) — fixed + approx floating
MY_PUBLIC_HOLIDAYS = {
    (1,1),(2,1),(5,1),(8,31),(9,16),(12,25),   # fixed national
    (1,29),(1,30),                              # CNY 2025
    (2,17),(2,18),                              # CNY 2026
    (2,11),                                    # Thaipusam 2025
    (1,31),                                    # Thaipusam 2026
    (3,31),(4,1),                              # Hari Raya Aidilfitri 2025
    (3,20),(3,21),                             # Hari Raya Aidilfitri 2026
    (6,7),(6,8),                               # Hari Raya Aidiladha 2025
    (5,27),(5,28),                             # Hari Raya Aidiladha 2026
    (5,12),                                    # Wesak 2025
    (5,1),                                     # Wesak 2026
    (9,5),                                     # Prophet Birthday 2025
    (8,25),                                    # Prophet Birthday 2026
    (10,20),(11,1),                            # Deepavali approx
}

# Ramadan: (year, start_month, start_day, end_month, end_day)
MY_RAMADAN_PERIODS = [
    (2025, 3, 1,  3, 30),
    (2026, 2, 18, 3, 19),
]

# Malaysian school holidays (year, sm, sd, em, ed)
MY_SCHOOL_HOLIDAYS = [
    (2025, 3,15,  3,23),(2025, 5,24, 6,8),(2025, 8,23, 8,31),(2025,11,22,12,31),
    (2026, 1, 1,  1, 4),(2026, 3,14, 3,22),(2026, 5,23, 6,7),(2026, 8,22, 8,30),
    (2026,11,21, 12,31),
]

# Feature lists — v2 has 26 features (was 19)
BASE_FEATS = [
    "hour_sin","hour_cos","dow_sin","dow_cos",          # cyclical encodings (new)
    "hour","day_of_week",                               # raw (kept for tree splits)
    "is_weekend","is_peak_morning","is_peak_evening","month",
    "is_public_holiday","is_school_holiday","is_ramadan", # calendar flags (new)
    "road_enc","lanes","design_speed","base_capacity","computed_capacity",
]
LAG_COLS = [
    "vol_total_lag1","vol_total_lag2","vol_total_lag3",
    "vc_ratio_lag1","vc_ratio_lag2",
    "vol_total_roll3_mean","vol_total_roll6_mean","vc_ratio_roll3_mean",
]
ALL_FEATS = BASE_FEATS + LAG_COLS

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Domain functions ───────────────────────────────────────────────────────────
def compute_los(vc):
    for t, g in zip(LOS_BREAKS, LOS_LABELS):
        if vc <= t: return g
    return "F"

def los_array(vc_arr):
    return np.array([compute_los(v) for v in vc_arr])

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    if mask.sum() == 0: return np.nan
    return 100 * float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

def make_weights(df):
    """Raised D/E boundary weights per audit recommendation."""
    return np.where(df["vc_ratio"] > 1.0, 3.0,
           np.where(df["vc_ratio"] > 0.9, 3.0,    # LOS E: raised 2.0 → 3.0
           np.where(df["vc_ratio"] > 0.8, 2.5,    # LOS D: new tier
           np.where(df["vc_ratio"] > 0.7, 1.5, 1.0))))


# ── Calendar helpers ───────────────────────────────────────────────────────────
def _holiday_set(years):
    s = set()
    for y in years:
        for m, d in MY_PUBLIC_HOLIDAYS:
            try: s.add(pd.Timestamp(y, m, d))
            except ValueError: pass
    return s

def _school_set():
    s = set()
    for y, sm, sd, em, ed in MY_SCHOOL_HOLIDAYS:
        for dt in pd.date_range(pd.Timestamp(y,sm,sd), pd.Timestamp(y,em,ed)):
            s.add(dt.normalize())
    return s

def _ramadan_set():
    s = set()
    for y, sm, sd, em, ed in MY_RAMADAN_PERIODS:
        for dt in pd.date_range(pd.Timestamp(y,sm,sd), pd.Timestamp(y,em,ed)):
            s.add(dt.normalize())
    return s

def _add_calendar_features(df):
    """Add cyclical encodings + calendar flags. Modifies df in-place."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    years   = df["date"].dt.year.unique().tolist()
    hol_s   = _holiday_set(years)
    sch_s   = _school_set()
    ram_s   = _ramadan_set()
    dn      = df["date"].dt.normalize()

    df["is_public_holiday"] = dn.isin(hol_s).astype(int)
    df["is_school_holiday"] = dn.isin(sch_s).astype(int)
    df["is_ramadan"]        = dn.isin(ram_s).astype(int)
    return df

def get_calendar_features_for_date(target_date):
    """
    Return dict of all new v2 features for a single prediction date.
    hour_sin / hour_cos are arrays[24] — index by hour.
    """
    dow      = target_date.weekday()
    dn       = target_date.normalize()
    years    = [target_date.year]
    is_ph    = int(dn in _holiday_set(years))
    is_sh    = int(dn in _school_set())
    is_ram   = int(dn in _ramadan_set())
    return {
        "hour_sin":          np.sin(2 * np.pi * np.arange(24) / 24),
        "hour_cos":          np.cos(2 * np.pi * np.arange(24) / 24),
        "dow_sin":           np.sin(2 * np.pi * dow / 7),
        "dow_cos":           np.cos(2 * np.pi * dow / 7),
        "is_public_holiday": is_ph,
        "is_school_holiday": is_sh,
        "is_ramadan":        is_ram,
    }


# ── Load & preprocess ──────────────────────────────────────────────────────────
def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    _validate_schema(df)

    road_order = (df.groupby("road")["computed_capacity"]
                    .first().sort_values().index.tolist())
    df["road_enc"] = df["road"].map({r: i for i, r in enumerate(road_order)})

    road_params = (df.groupby("road")[["lanes","design_speed","base_capacity",
                                       "adj_factor","computed_capacity","road_enc"]]
                     .first().to_dict("index"))

    df = _add_calendar_features(df)
    df[LAG_COLS] = df[LAG_COLS].fillna(0)
    return df, road_params, road_order

def train_test_split(df):
    cutoff = df["date"].max() - pd.Timedelta(days=6)
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()


# ── Schema validation ──────────────────────────────────────────────────────────
REQUIRED_COLS = (["date","road","hour","day_of_week","is_weekend","month",
                  "is_peak_morning","is_peak_evening","lanes","design_speed",
                  "base_capacity","adj_factor","computed_capacity","vc_ratio","los_grade"]
                 + TARGET_VOLS + LAG_COLS)

def _validate_schema(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing: raise ValueError(f"CSV missing required columns: {missing}")
    bad = df[TARGET_VOLS+["vc_ratio"]].isnull().mean()
    bad = bad[bad > 0.05]
    if not bad.empty: raise ValueError(f"Columns with >5% nulls: {bad.to_dict()}")


# ── Drift detection ────────────────────────────────────────────────────────────
def compute_psi(reference, current, bins=10):
    lo, hi = min(reference.min(), current.min()), max(reference.max(), current.max())
    if lo == hi: return 0.0
    edges = np.linspace(lo, hi, bins+1)
    r = (np.histogram(reference, bins=edges)[0].astype(float) + 1e-6)
    c = (np.histogram(current,   bins=edges)[0].astype(float) + 1e-6)
    r /= r.sum(); c /= c.sum()
    return float(np.sum((r - c) * np.log(r / c)))

def detect_drift(train_df, eval_df, feats=None):
    feats = feats or BASE_FEATS
    results = {}
    for f in feats:
        if f not in train_df.columns or f not in eval_df.columns: continue
        a, b = train_df[f].dropna().values, eval_df[f].dropna().values
        ks_stat, ks_p = ks_2samp(a, b)
        psi = compute_psi(a, b)
        results[f] = {
            "ks_stat": round(float(ks_stat), 4),
            "ks_p":    round(float(ks_p), 4),
            "psi":     round(psi, 4),
            "status":  "stable" if psi < 0.1 else ("slight" if psi < 0.2 else "significant"),
        }
    return results