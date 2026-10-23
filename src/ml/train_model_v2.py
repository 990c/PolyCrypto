"""
CatBoost Model Trainer V2 — Epoch-Aggregated
=============================================
Trains on epoch-aggregated features from label_dataset_v2.py.
Each row = one 5-minute epoch, not one 100ms tick.

Usage:
  python src/ml/train_model_v2.py --input dataset_labeled_v2.csv
"""
import os
import sys
import argparse
import csv

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml.label_dataset_v3 import MODEL_FEATURE_NAMES

FEATURE_COLS = MODEL_FEATURE_NAMES


def load_dataset(path: str):
    X, y = [], []
    skipped = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = int(row["target_resolution"])
                feats = [float(row.get(col, 0)) for col in FEATURE_COLS]
                X.append(feats)
                y.append(label)
            except (ValueError, KeyError):
                skipped += 1

    logger.info(f"Loaded {len(X)} epoch samples ({skipped} skipped). Features: {len(FEATURE_COLS)}")
    return np.array(X), np.array(y)


def train(input_path: str, output_path: str):
    X, y = load_dataset(input_path)

    if len(X) < 30:
        logger.error(f"Only {len(X)} epochs. Need more data.")
        return None

    up_count = int(np.sum(y == 1))
    down_count = int(np.sum(y == 0))
    logger.info(f"Class balance: UP={up_count} ({up_count / len(y) * 100:.1f}%) | DOWN={down_count} ({down_count / len(y) * 100:.1f}%)")

    # Time-ordered split: 70% train / 10% cal / 20% test
    train_end = int(len(X) * 0.70)
    cal_end = int(len(X) * 0.80)

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    logger.info(f"Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

    from catboost import CatBoostClassifier, Pool

    scale_pos_weight = down_count / up_count if up_count > 0 else 1.0

    # Try multiple configs and pick best
    configs = [
        {"depth": 4, "lr": 0.01, "l2": 10.0, "iters": 1000, "es": 200},
        {"depth": 3, "lr": 0.005, "l2": 20.0, "iters": 2000, "es": 300},
        {"depth": 6, "lr": 0.03, "l2": 5.0, "iters": 500, "es": 100},
        {"depth": 4, "lr": 0.02, "l2": 15.0, "iters": 1000, "es": 150},
    ]

    best_auc = 0
    best_model = None
    best_config_name = ""

    for cfg in configs:
        model = CatBoostClassifier(
            iterations=cfg["iters"],
            depth=cfg["depth"],
            learning_rate=cfg["lr"],
            loss_function="Logloss",
            eval_metric="AUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=cfg["es"],
            use_best_model=True,
            l2_leaf_reg=cfg["l2"],
            thread_count=-1,
            bootstrap_type="MVS",
        )

        train_pool = Pool(X_train, y_train, feature_names=FEATURE_COLS)
        cal_pool = Pool(X_cal, y_cal, feature_names=FEATURE_COLS)
        model.fit(train_pool, eval_set=cal_pool)

        from sklearn.metrics import roc_auc_score
        # Select by CAL AUC — selecting on the test set leaks and inflated
        # earlier reported AUCs (see EXPERIMENTS.md Exp 4).
        cal_auc = roc_auc_score(y_cal, model.predict_proba(X_cal)[:, 1])
        best_iter = model.get_best_iteration()

        cfg_name = f"d={cfg['depth']} lr={cfg['lr']} l2={cfg['l2']}"
        logger.info(f"  Config [{cfg_name}]: calAUC={cal_auc:.4f} | BestIter={best_iter}")

        if cal_auc > best_auc:
            best_auc = cal_auc
            best_model = model
            best_config_name = cfg_name

    logger.info(f"\nBest config: [{best_config_name}] calAUC={best_auc:.4f}")

    # Full evaluation with best model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
    import pickle

    from ml.calibration import IsotonicLogOddsCalibrator

    # Isotonic calibration (Exp 3, EXPERIMENTS.md): beats Platt on held-out
    # log-loss/Brier and produces honest probabilities instead of compressing
    # everything into a narrow band around the class prior.
    raw_cal_proba = best_model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicLogOddsCalibrator().fit(raw_cal_proba, y_cal)

    calibrator_path = output_path.replace(".cbm", "_calibrator.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)

    # Test metrics
    raw_test_proba = best_model.predict_proba(X_test)[:, 1]
    log_odds_test = np.log(raw_test_proba / (1 - raw_test_proba + 1e-10) + 1e-10).reshape(-1, 1)
    cal_test_proba = calibrator.predict_proba(log_odds_test)[:, 1]
    y_pred = (cal_test_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, raw_test_proba)
    ll_cal = log_loss(y_test, cal_test_proba)

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION (out-of-sample, calibrated)")
    logger.info("=" * 60)
    logger.info(f"Accuracy:         {acc:.4f}")
    logger.info(f"Precision:        {prec:.4f}")
    logger.info(f"Recall:           {rec:.4f}")
    logger.info(f"AUC-ROC:          {auc:.4f}")
    logger.info(f"Log Loss (cal):   {ll_cal:.4f}")
    logger.info("=" * 60)

    # Edge check
    base_rate = float(np.mean(y_test))
    logger.info("")
    logger.info("EDGE CHECK (calibrated, by confidence threshold)")
    logger.info(f"{'Threshold':>10s} {'Trades':>7s} {'Accuracy':>10s} {'vs_Base':>10s}")
    logger.info("-" * 42)
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
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
    importances = best_model.get_feature_importance()
    logger.info("")
    logger.info("FEATURE IMPORTANCE:")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        if imp > 0:
            logger.info(f"  {name:28s} {imp:.2f}")

    # Calibration spread check
    logger.info("")
    logger.info(f"Calibrated prob range: [{cal_test_proba.min():.4f}, {cal_test_proba.max():.4f}]")
    logger.info(f"Prob std: {cal_test_proba.std():.4f}")

    best_model.save_model(output_path)
    logger.info(f"Model saved to {output_path}")

    # Version every artifact so a bad retrain can be rolled back
    import shutil
    from datetime import datetime

    archive_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)) or ".", "models")
    os.makedirs(archive_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(output_path))[0]
    shutil.copy2(output_path, os.path.join(archive_dir, f"{base}_{stamp}.cbm"))
    shutil.copy2(calibrator_path, os.path.join(archive_dir, f"{base}_{stamp}_calibrator.pkl"))
    logger.info(f"Artifacts archived to {archive_dir}/{base}_{stamp}.*")

    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset_labeled_v3.csv")
    parser.add_argument("--output", default="catboost_model.cbm")
    args = parser.parse_args()

    train(args.input, args.output)


if __name__ == "__main__":
    main()
