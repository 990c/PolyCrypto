"""
Model Evaluation Script
=======================
Loads the trained model and runs a detailed out-of-sample evaluation:
  - Accuracy, AUC, precision, recall
  - Probability distribution analysis
  - Calibration check (predicted vs actual win rate by bucket)
  - Simulated trade profitability at various confidence thresholds

Usage:
  python src/ml/evaluate_model.py --model catboost_model.cbm --data dataset_labeled.csv
"""
import os
import sys
import argparse
import csv

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ml.features import FeatureEngine

FEATURE_COLS = FeatureEngine.FEATURE_NAMES


def load_dataset(path: str, subsample: int = 10):
    X, y = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % subsample != 0:
                continue
            try:
                label = int(row["target_resolution"])
                feats = [float(row.get(col, 0)) for col in FEATURE_COLS]
                X.append(feats)
                y.append(label)
            except (ValueError, KeyError):
                continue
    return np.array(X), np.array(y)


def evaluate(model_path: str, data_path: str, test_size: float = 0.2,
             subsample: int = 10):
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    logger.info(f"Loading data from {data_path} (subsample=1/{subsample})...")
    X, y = load_dataset(data_path, subsample=subsample)
    logger.info(f"Total samples: {len(X)}")

    # Time-ordered split
    split_idx = int(len(X) * (1 - test_size))
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    logger.info(f"Test set: {len(X_test)} samples")

    # Class balance in test set
    up_count = int(np.sum(y_test == 1))
    down_count = int(np.sum(y_test == 0))
    base_rate = up_count / len(y_test)
    logger.info(f"Test class balance: UP={up_count} ({base_rate*100:.1f}%) | DOWN={down_count} ({(1-base_rate)*100:.1f}%)")

    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)

    logger.info("=" * 60)
    logger.info("BASIC METRICS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:    {acc:.4f}  (baseline: {max(base_rate, 1-base_rate):.4f})")
    logger.info(f"AUC-ROC:     {auc:.4f}")
    logger.info(f"Log Loss:    {ll:.4f}")

    # --- Probability distribution ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROBABILITY DISTRIBUTION (raw model output)")
    logger.info("=" * 60)
    logger.info(f"Mean:   {np.mean(y_proba):.4f}")
    logger.info(f"Std:    {np.std(y_proba):.4f}")
    logger.info(f"Min:    {np.min(y_proba):.4f}")
    logger.info(f"Max:    {np.max(y_proba):.4f}")
    logger.info(f"Median: {np.median(y_proba):.4f}")

    pctiles = [5, 10, 25, 75, 90, 95]
    for p in pctiles:
        logger.info(f"P{p:02d}:    {np.percentile(y_proba, p):.4f}")

    # --- Calibration by bucket ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION (predicted prob vs actual win rate)")
    logger.info("=" * 60)
    logger.info(f"{'Bucket':>12s} {'Count':>7s} {'Pred_P':>8s} {'Actual':>8s} {'Gap':>8s}")
    logger.info("-" * 50)

    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5),
               (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for lo, hi in buckets:
        mask = (y_proba >= lo) & (y_proba < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        pred_mean = float(np.mean(y_proba[mask]))
        actual_rate = float(np.mean(y_test[mask]))
        gap = actual_rate - pred_mean
        logger.info(f"[{lo:.2f},{hi:.2f}) {n:7d} {pred_mean:8.4f} {actual_rate:8.4f} {gap:+8.4f}")

    # --- Profitability simulation at confidence thresholds ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("EDGE CHECK: accuracy at confidence thresholds")
    logger.info("=" * 60)
    logger.info(f"{'Threshold':>10s} {'Trades':>7s} {'Accuracy':>10s} {'vs_Base':>10s}")
    logger.info("-" * 42)

    thresholds = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    for thresh in thresholds:
        # Predict UP when prob > thresh, predict DOWN when prob < (1-thresh)
        up_mask = y_proba >= thresh
        down_mask = y_proba <= (1 - thresh)
        trade_mask = up_mask | down_mask

        n_trades = int(np.sum(trade_mask))
        if n_trades == 0:
            logger.info(f"     {thresh:.2f}       0          -          -")
            continue

        preds = np.where(up_mask[trade_mask], 1, 0)
        actuals = y_test[trade_mask]
        trade_acc = float(np.mean(preds == actuals))
        vs_base = trade_acc - base_rate
        logger.info(f"     {thresh:.2f} {n_trades:7d} {trade_acc:10.4f} {vs_base:+10.4f}")

    # --- Feature importance ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)
    importances = model.get_feature_importance()
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp / 2)
        logger.info(f"  {name:20s} {imp:6.2f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", default="catboost_model.cbm")
    parser.add_argument("--data", default="dataset_labeled.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--subsample", type=int, default=10)
    args = parser.parse_args()

    evaluate(args.model, args.data, test_size=args.test_size,
             subsample=args.subsample)


if __name__ == "__main__":
    main()
