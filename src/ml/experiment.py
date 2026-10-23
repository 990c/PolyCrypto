"""
Run multiple training experiments and compare results.
Usage: python src/ml/experiment.py
"""
import os
import sys
import csv
import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ml.features import FeatureEngine

ALL_FEATURES = FeatureEngine.FEATURE_NAMES

# Experiment 1: Drop zero-importance features
LEAN_FEATURES = [
    "cvd_15m", "dist_from_high_15m", "obi_mean_5m", "dist_from_low_15m",
    "cvd_5m", "return_1m", "obi", "volume_accel", "trade_intensity", "price_accel",
]


def load_dataset(path: str, subsample: int = 1, feature_cols: list = None,
                 recent_fraction: float = 1.0):
    if feature_cols is None:
        feature_cols = ALL_FEATURES
    X, y = [], []
    skipped = 0

    # First pass: count total rows for recent_fraction
    if recent_fraction < 1.0:
        total = sum(1 for _ in open(path)) - 1  # minus header
        start_row = int(total * (1 - recent_fraction))
    else:
        start_row = 0

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start_row:
                continue
            if (i - start_row) % subsample != 0:
                continue
            try:
                label = int(row["target_resolution"])
                feats = [float(row.get(col, 0)) for col in feature_cols]
                X.append(feats)
                y.append(label)
            except (ValueError, KeyError):
                skipped += 1

    return np.array(X), np.array(y), feature_cols


def train_experiment(name: str, X: np.ndarray, y: np.ndarray, feature_cols: list,
                     iterations: int = 500, depth: int = 6, lr: float = 0.05,
                     l2_reg: float = 3.0, early_stop: int = 150):
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    import pickle

    up_count = int(np.sum(y == 1))
    down_count = int(np.sum(y == 0))
    scale_pos_weight = down_count / up_count if up_count > 0 else 1.0

    cal_size = 0.10
    test_size = 0.20
    train_end = int(len(X) * (1 - test_size - cal_size))
    cal_end = int(len(X) * (1 - test_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=early_stop,
        use_best_model=True,
        l2_leaf_reg=l2_reg,
        thread_count=-1,
        bootstrap_type="MVS",
    )

    train_pool = Pool(X_train, y_train, feature_names=feature_cols)
    cal_pool = Pool(X_cal, y_cal, feature_names=feature_cols)
    model.fit(train_pool, eval_set=cal_pool)

    # Platt scaling
    raw_cal_proba = model.predict_proba(X_cal)[:, 1]
    log_odds_cal = np.log(raw_cal_proba / (1 - raw_cal_proba + 1e-10) + 1e-10).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(log_odds_cal, y_cal)

    # Test evaluation
    raw_test_proba = model.predict_proba(X_test)[:, 1]
    log_odds_test = np.log(raw_test_proba / (1 - raw_test_proba + 1e-10) + 1e-10).reshape(-1, 1)
    cal_test_proba = calibrator.predict_proba(log_odds_test)[:, 1]
    y_pred = (cal_test_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, raw_test_proba)
    best_iter = model.get_best_iteration()

    # Edge check at 0.55 threshold
    base_rate = float(np.mean(y_test))
    up_mask = cal_test_proba >= 0.55
    down_mask = cal_test_proba <= 0.45
    trade_mask = up_mask | down_mask
    n_trades = int(np.sum(trade_mask))
    if n_trades > 0:
        preds = np.where(up_mask[trade_mask], 1, 0)
        actuals = y_test[trade_mask]
        trade_acc = float(np.mean(preds == actuals))
    else:
        trade_acc = 0.0

    # Feature importance
    importances = model.get_feature_importance()
    top_feats = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:5]

    return {
        "name": name,
        "samples": len(X),
        "auc": auc,
        "accuracy": acc,
        "best_iter": best_iter,
        "trades_55": n_trades,
        "acc_55": trade_acc,
        "top_features": top_feats,
        "model": model,
        "calibrator": calibrator,
    }


def main():
    INPUT = "dataset_labeled.csv"
    results = []

    # =========================================================
    # Experiment 1: Drop zero-importance features (subsample=10)
    # =========================================================
    logger.info("=" * 60)
    logger.info("EXP 1: Lean features (drop 10 zero-importance)")
    X, y, cols = load_dataset(INPUT, subsample=10, feature_cols=LEAN_FEATURES)
    logger.info(f"  Samples: {len(X)} | Features: {len(cols)}")
    r = train_experiment("1_lean_features", X, y, cols)
    results.append(r)
    logger.info(f"  AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f} | BestIter={r['best_iter']} | "
                f"Trades@0.55={r['trades_55']} Acc@0.55={r['acc_55']:.4f}")

    # =========================================================
    # Experiment 2: Tuned hyperparameters (lower LR, more depth, more regularization)
    # =========================================================
    logger.info("=" * 60)
    logger.info("EXP 2: Tuned hyperparams (lr=0.01, depth=4, l2=10, 1000 iters)")
    X, y, cols = load_dataset(INPUT, subsample=10, feature_cols=ALL_FEATURES)
    logger.info(f"  Samples: {len(X)} | Features: {len(cols)}")
    r = train_experiment("2_tuned_hyperparams", X, y, cols,
                         iterations=1000, depth=4, lr=0.01, l2_reg=10.0,
                         early_stop=200)
    results.append(r)
    logger.info(f"  AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f} | BestIter={r['best_iter']} | "
                f"Trades@0.55={r['trades_55']} Acc@0.55={r['acc_55']:.4f}")

    # =========================================================
    # Experiment 3: Higher subsample (20 and 30)
    # =========================================================
    for ss in [20, 30]:
        logger.info("=" * 60)
        logger.info(f"EXP 3{chr(96+ss//10)}: Subsample={ss}")
        X, y, cols = load_dataset(INPUT, subsample=ss, feature_cols=ALL_FEATURES)
        logger.info(f"  Samples: {len(X)} | Features: {len(cols)}")
        r = train_experiment(f"3_subsample_{ss}", X, y, cols)
        results.append(r)
        logger.info(f"  AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f} | BestIter={r['best_iter']} | "
                    f"Trades@0.55={r['trades_55']} Acc@0.55={r['acc_55']:.4f}")

    # =========================================================
    # Experiment 4: Recent data only (last 50% and last 25%)
    # =========================================================
    for frac in [0.50, 0.25]:
        logger.info("=" * 60)
        logger.info(f"EXP 4: Recent {int(frac*100)}% of data only, subsample=10")
        X, y, cols = load_dataset(INPUT, subsample=10, feature_cols=ALL_FEATURES,
                                  recent_fraction=frac)
        logger.info(f"  Samples: {len(X)} | Features: {len(cols)}")
        r = train_experiment(f"4_recent_{int(frac*100)}pct", X, y, cols)
        results.append(r)
        logger.info(f"  AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f} | BestIter={r['best_iter']} | "
                    f"Trades@0.55={r['trades_55']} Acc@0.55={r['acc_55']:.4f}")

    # =========================================================
    # Experiment 5: COMBO — lean features + tuned hyperparams + recent 50%
    # =========================================================
    logger.info("=" * 60)
    logger.info("EXP 5: COMBO (lean features + tuned HP + recent 50%)")
    X, y, cols = load_dataset(INPUT, subsample=10, feature_cols=LEAN_FEATURES,
                              recent_fraction=0.50)
    logger.info(f"  Samples: {len(X)} | Features: {len(cols)}")
    r = train_experiment("5_combo", X, y, cols,
                         iterations=1000, depth=4, lr=0.01, l2_reg=10.0,
                         early_stop=200)
    results.append(r)
    logger.info(f"  AUC={r['auc']:.4f} | Acc={r['accuracy']:.4f} | BestIter={r['best_iter']} | "
                f"Trades@0.55={r['trades_55']} Acc@0.55={r['acc_55']:.4f}")

    # =========================================================
    # Summary
    # =========================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Experiment':<30s} {'Samples':>8s} {'AUC':>7s} {'Acc':>7s} {'Iter':>5s} "
                f"{'Tr@.55':>7s} {'Acc@.55':>8s}")
    logger.info("-" * 80)

    baseline_auc = 0.5462  # from original run
    for r in sorted(results, key=lambda x: -x["auc"]):
        delta = r["auc"] - baseline_auc
        logger.info(f"{r['name']:<30s} {r['samples']:>8d} {r['auc']:>7.4f} {r['accuracy']:>7.4f} "
                    f"{r['best_iter']:>5d} {r['trades_55']:>7d} {r['acc_55']:>8.4f}  "
                    f"({'+'if delta>=0 else ''}{delta:.4f})")

    # Find best
    best = max(results, key=lambda x: x["auc"])
    logger.info("")
    logger.info(f"BEST: {best['name']} (AUC={best['auc']:.4f})")
    logger.info("Top features:")
    for name, imp in best["top_features"]:
        logger.info(f"  {name:20s} {imp:.2f}")

    # Save best model
    best["model"].save_model("catboost_model_best.cbm")
    import pickle
    with open("catboost_model_best_calibrator.pkl", "wb") as f:
        pickle.dump(best["calibrator"], f)
    logger.info(f"Best model saved to catboost_model_best.cbm")


if __name__ == "__main__":
    main()
