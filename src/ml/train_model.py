"""
CatBoost Model Trainer
======================
Trains a CatBoost binary classifier on the labeled feature dataset.

Pipeline:
  1. Collect data:   python src/main.py  (DATA_COLLECTION mode)
  2. Label data:     python src/ml/label_dataset.py
  3. Train model:    python src/ml/train_model.py
  4. Backtest:       python src/backtest.py --model catboost_model.cbm

Usage:
  python src/ml/train_model.py --input dataset_labeled.csv --output catboost_model.cbm
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


def load_dataset(path: str, subsample: int = 1):
    """Load labeled CSV and return X (features), y (labels).

    subsample: keep every Nth row (time-ordered). 10 = ~1 row/sec from 100ms data.
    """
    X, y = [], []
    skipped = 0
    kept = 0
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
                kept += 1
            except (ValueError, KeyError):
                skipped += 1
                continue

    logger.info(f"Loaded {kept} samples ({skipped} skipped, subsample=1/{subsample}). Features: {FEATURE_COLS}")
    return np.array(X), np.array(y)


def train(input_path: str, output_path: str, test_size: float = 0.2,
          iterations: int = 500, depth: int = 6, lr: float = 0.05,
          subsample: int = 1):
    X, y = load_dataset(input_path, subsample=subsample)

    if len(X) < 50:
        logger.error(f"Only {len(X)} samples. Need at least 50 for meaningful training.")
        return None

    # Class balance check
    up_count = int(np.sum(y == 1))
    down_count = int(np.sum(y == 0))
    logger.info(f"Class balance: UP={up_count} ({up_count/len(y)*100:.1f}%) | DOWN={down_count} ({down_count/len(y)*100:.1f}%)")

    # Three-way time-ordered split: train / calibration / test
    # 70% train, 10% calibration, 20% test
    cal_size = 0.10
    train_end = int(len(X) * (1 - test_size - cal_size))
    cal_end = int(len(X) * (1 - test_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    logger.info(f"Train: {len(X_train)} | Calibration: {len(X_cal)} | Test: {len(X_test)}")

    from catboost import CatBoostClassifier, Pool

    # Auto-handle class imbalance
    scale_pos_weight = down_count / up_count if up_count > 0 else 1.0

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=50,
        early_stopping_rounds=150,
        use_best_model=True,
        l2_leaf_reg=10.0,
        thread_count=-1,
        bootstrap_type="MVS",
    )

    train_pool = Pool(X_train, y_train, feature_names=FEATURE_COLS)
    cal_pool = Pool(X_cal, y_cal, feature_names=FEATURE_COLS)

    model.fit(train_pool, eval_set=cal_pool)

    # --- Platt scaling on held-out calibration set ---
    from sklearn.linear_model import LogisticRegression
    import pickle

    raw_cal_proba = model.predict_proba(X_cal)[:, 1]
    # Convert to log-odds for Platt scaling input
    log_odds_cal = np.log(raw_cal_proba / (1 - raw_cal_proba + 1e-10) + 1e-10).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(log_odds_cal, y_cal)

    calibrator_path = output_path.replace(".cbm", "_calibrator.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info(f"Platt scaling calibrator saved to {calibrator_path}")

    # --- Evaluation on test set ---
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss

    raw_test_proba = model.predict_proba(X_test)[:, 1]
    log_odds_test = np.log(raw_test_proba / (1 - raw_test_proba + 1e-10) + 1e-10).reshape(-1, 1)
    cal_test_proba = calibrator.predict_proba(log_odds_test)[:, 1]
    y_pred = (cal_test_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, raw_test_proba)
    ll_raw = log_loss(y_test, raw_test_proba)
    ll_cal = log_loss(y_test, cal_test_proba)

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION (out-of-sample, calibrated)")
    logger.info("=" * 60)
    logger.info(f"Accuracy:         {acc:.4f}")
    logger.info(f"Precision:        {prec:.4f}")
    logger.info(f"Recall:           {rec:.4f}")
    logger.info(f"AUC-ROC:          {auc:.4f}")
    logger.info(f"Log Loss (raw):   {ll_raw:.4f}")
    logger.info(f"Log Loss (cal):   {ll_cal:.4f}")
    logger.info("=" * 60)

    # Calibration check
    logger.info("")
    logger.info("CALIBRATION CHECK (calibrated probabilities)")
    logger.info(f"{'Bucket':>12s} {'Count':>7s} {'Pred_P':>8s} {'Actual':>8s} {'Gap':>8s}")
    logger.info("-" * 50)
    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5),
               (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for lo, hi in buckets:
        mask = (cal_test_proba >= lo) & (cal_test_proba < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        pred_mean = float(np.mean(cal_test_proba[mask]))
        actual_rate = float(np.mean(y_test[mask]))
        gap = actual_rate - pred_mean
        logger.info(f"[{lo:.2f},{hi:.2f}) {n:7d} {pred_mean:8.4f} {actual_rate:8.4f} {gap:+8.4f}")

    # Edge check at confidence thresholds
    base_rate = float(np.mean(y_test))
    logger.info("")
    logger.info("EDGE CHECK (calibrated, by confidence threshold)")
    logger.info(f"{'Threshold':>10s} {'Trades':>7s} {'Accuracy':>10s} {'vs_Base':>10s}")
    logger.info("-" * 42)
    for thresh in [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        up_mask = cal_test_proba >= thresh
        down_mask = cal_test_proba <= (1 - thresh)
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

    # Feature importance
    importances = model.get_feature_importance()
    logger.info("")
    logger.info("FEATURE IMPORTANCE:")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        logger.info(f"  {name:20s} {imp:.2f}")

    # Profitability sanity check
    if auc < 0.52:
        logger.warning("AUC < 0.52 — model barely better than random. Consider more data or different features.")
    elif auc < 0.55:
        logger.info("AUC 0.52-0.55: marginal edge. With Kelly sizing + fee advantage this may still be profitable.")
    else:
        logger.info(f"AUC {auc:.4f}: solid predictive edge detected.")

    # Save model
    model.save_model(output_path)
    logger.info(f"Model saved to {output_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost model for BTC 15M prediction")
    parser.add_argument("--input", default="dataset_labeled.csv", help="Labeled CSV")
    parser.add_argument("--output", default="catboost_model.cbm", help="Output model path")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--subsample", type=int, default=10, help="Keep every Nth row (10 = ~1/sec from 100ms data)")
    args = parser.parse_args()

    train(args.input, args.output, test_size=args.test_size,
          iterations=args.iterations, depth=args.depth, lr=args.lr,
          subsample=args.subsample)


if __name__ == "__main__":
    main()
