#!/usr/bin/env python3
"""
run.py — CLI entry-point.  Preserves the exact input/output behaviour
         of the original predict.py:
           • Reads traffic_los_dataset.csv
           • Trains models
           • Generates Traffic_LOS_YYYYMMDD.xlsx

Usage:
    python run.py                          # train + predict 2026-02-18
    python run.py --date 2026-03-01        # predict another date
    python run.py --tune                   # enable Optuna-style tuning
    python run.py --n-iter 30              # more tuning candidates
    python run.py --serve                  # start Flask + dashboard
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

# Ensure the directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (CSV_PATH, load_data, train_test_split, detect_drift,
                  ALL_FEATS, TARGET_VOLS, HOURS, compute_los, PCU_FACTORS,
                  LOS_ENCODE, LOS_LABELS, BASE_DIR, PRED_DATE)
from model import tune_model, train, evaluate, save_model, load_model, list_versions
from predict_engine import predict_date, build_excel

def sep(c="─", w=72): print(c * w)
def header(t): sep(); print(f"  {t}"); sep()


def run_pipeline(target_date: pd.Timestamp, do_tune: bool = False,
                 n_iter: int = 20, n_splits: int = 5,
                 out_path: str = None) -> str:
    """Full train → evaluate → predict → Excel pipeline."""

    # ── 1. Load & preprocess ────────────────────────────────────────────────
    header("1 / 5  LOADING & PREPROCESSING")
    df, road_params, road_order = load_data(CSV_PATH)
    train_df, test_df = train_test_split(df)

    print(f"  Dataset      : {len(df):,} rows  |  {df['date'].nunique()} days  |  {df['road'].nunique()} roads")
    print(f"  Training set : {len(train_df):,} rows  ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test set     : {len(test_df):,} rows   ({test_df['date'].min().date()} → {test_df['date'].max().date()})")
    print(f"  Features     : {len(ALL_FEATS)} ({len(ALL_FEATS)-8} lag/rolling)")

    # ── 2. Tune / train ─────────────────────────────────────────────────────
    header("2 / 5  MODEL TRAINING  (multi-output HistGradientBoosting)")
    best_params = None
    if do_tune:
        print(f"  Hyperparameter search: {n_iter} candidates × {n_splits}-fold time-series CV")
        print("  Objective: 0.7 × V/C MAE − 0.3 × LOS Accuracy\n")
        best_params = tune_model(train_df, n_iter=n_iter, n_splits=n_splits, verbose=True)
        sep("-", 72)

    model = train(train_df, params=best_params, verbose=True)

    # ── 3. Evaluate ─────────────────────────────────────────────────────────
    header("3 / 5  EVALUATION REPORT")
    metrics = evaluate(model, test_df)

    print(f"\n  {'Target':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'sMAPE%':>8}")
    sep("-", 60)
    for t, m in metrics["per_target"].items():
        flag = " ⚠ LOW R²" if m["r2"] < 0.7 else ""
        print(f"  {t:<22} {m['mae']:>8,.1f} {m['rmse']:>8,.1f} {m['r2']:>8.3f} {m['smape']:>8.1f}{flag}")
    sep("-", 60)

    print(f"\n  V/C Ratio (derived)  MAE={metrics['vc_mae']:.4f}  RMSE={metrics['vc_rmse']:.4f}  R²={metrics['vc_r2']:.3f}")
    print(f"  LOS Grade accuracy   Exact={metrics['los_accuracy']*100:.1f}%   Within-1={metrics['los_within1']*100:.1f}%")

    print(f"\n  Residuals: mean={metrics['residuals']['mean']:+.4f}  std={metrics['residuals']['std']:.4f}"
          f"  P5/P95={metrics['residuals']['p5']:+.4f}/{metrics['residuals']['p95']:+.4f}")

    print(f"\n  Per-road V/C performance:")
    for road, m in metrics["per_road"].items():
        status = "✓ Good" if m["r2"] > 0.75 and m["mae"] < 0.15 else ("⚠  Watch" if m["r2"] > 0.5 else "✗ Poor")
        print(f"  {road:<32} MAE={m['mae']:.4f}  R²={m['r2']:.3f}  LOS={m['los_acc']*100:.1f}%  {status}")

    # Drift check
    drift = detect_drift(train_df, test_df)
    flagged = {f: v for f, v in drift.items() if v["status"] != "stable"}
    if flagged:
        print(f"\n  ⚠ Data drift detected on {len(flagged)} features:")
        for f, v in flagged.items():
            print(f"    {f:<28}  PSI={v['psi']:.4f}  KS={v['ks_stat']:.4f}  [{v['status']}]")
    else:
        print(f"\n  ✓ No significant data drift detected.")

    # ── 4. Retrain on full dataset ──────────────────────────────────────────
    header("4 / 5  RETRAINING ON FULL DATASET")
    model = train(df, params=best_params, verbose=True)
    version = save_model(model, best_params or {}, metrics, road_params, road_order, CSV_PATH)
    print(f"  Saved as version: {version}")

    # ── 5. Predict & build Excel ────────────────────────────────────────────
    header(f"5 / 5  PREDICTIONS FOR {target_date.strftime('%d %b %Y').upper()}")
    predictions, summary = predict_date(model, road_params, road_order, target_date)

    print(f"\n  {'Road':<30} {'Daily':>10} {'Peak V/C':>9} {'Peak Hour':>11} {'Avg V/C':>8} {'LOS':>5}")
    sep("-", 80)
    for row in summary:
        flag = "  ⚠ CONGESTED" if row["peak_los"] == "F" else ""
        print(f"  {row['road']:<30} {row['daily']:>10,} {row['peak_vc']:>9.3f} "
              f"{row['peak_hour']:>11} {row['avg_vc']:>8.3f} {row['peak_los']:>5}{flag}")
    sep("-", 80)
    congested = sum(1 for r in summary if r["peak_los"] == "F")
    print(f"\n  Roads at LOS F (over-capacity): {congested} / {len(summary)}")

    date_str  = target_date.strftime("%Y%m%d")
    out_path  = out_path or os.path.join(BASE_DIR, f"Traffic_LOS_{date_str}.xlsx")
    excel_path = build_excel(predictions, road_params, target_date, out_path)

    sep()
    print(f"\n  Output  →  {excel_path}")
    sep()
    return excel_path


def main():
    parser = argparse.ArgumentParser(description="Traffic LOS ML System")
    parser.add_argument("--date",    default="2026-02-18", help="Prediction date (YYYY-MM-DD)")
    parser.add_argument("--tune",    action="store_true",  help="Enable hyperparameter tuning")
    parser.add_argument("--n-iter",  type=int, default=20, help="Tuning candidates")
    parser.add_argument("--n-splits",type=int, default=5,  help="CV folds")
    parser.add_argument("--serve",   action="store_true",  help="Start Flask server on :5000")
    parser.add_argument("--out",     default=None,         help="Output Excel path override")
    args = parser.parse_args()

    if args.serve:
        print("  Starting Traffic LOS Intelligence Platform on http://0.0.0.0:5000")
        print("  Dashboard: http://localhost:5000/")
        os.environ["FLASK_APP"] = "app.py"
        from app import app
        app.run(host="0.0.0.0", port=5000, debug=False)
        return

    target_date = pd.Timestamp(args.date)
    run_pipeline(target_date, do_tune=args.tune,
                 n_iter=args.n_iter, n_splits=args.n_splits,
                 out_path=args.out)


if __name__ == "__main__":
    main()
