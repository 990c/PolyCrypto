"""Exp 8 Step 1 kill test.

1. Label agreement: do binance.com-derived epoch labels match our
   binance.us-collected labels on overlapping epochs?
2. Transfer: does a model trained on RECONSTRUCTED archive epochs predict
   our REAL collected June epochs?

Arms (identical features, identical protocol, cal-AUC selection, isotonic):
  A: train archive Apr+May, cal archive Jun 1-10  -> test collected Jun 11-26
  B: train collected Apr+May, cal collected Jun 1-10 -> test collected Jun 11-26
  C: train archive Apr+May, cal archive Jun 1-10  -> test archive Jun 11-26

A vs B isolates the data-source gap; C shows in-source generalization.
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from price_features import PRICE_FEATURE_NAMES

from ml.calibration import IsotonicLogOddsCalibrator

JUN1 = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp())
JUN11 = int(datetime(2026, 6, 11, tzinfo=timezone.utc).timestamp())

CONFIGS = [
    {"depth": 4, "lr": 0.01, "l2": 10.0, "iters": 1000, "es": 200},
    {"depth": 3, "lr": 0.005, "l2": 20.0, "iters": 2000, "es": 300},
    {"depth": 6, "lr": 0.03, "l2": 5.0, "iters": 500, "es": 100},
]


def train_eval(train_df, cal_df, test_df, name):
    from catboost import CatBoostClassifier
    from sklearn.metrics import log_loss, roc_auc_score

    X_tr, y_tr = train_df[PRICE_FEATURE_NAMES].to_numpy(), train_df["target_resolution"].to_numpy()
    X_cal, y_cal = cal_df[PRICE_FEATURE_NAMES].to_numpy(), cal_df["target_resolution"].to_numpy()
    X_te, y_te = test_df[PRICE_FEATURE_NAMES].to_numpy(), test_df["target_resolution"].to_numpy()

    best_auc, best = -1, None
    for cfg in CONFIGS:
        m = CatBoostClassifier(
            iterations=cfg["iters"], depth=cfg["depth"], learning_rate=cfg["lr"],
            loss_function="Logloss", eval_metric="AUC",
            scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
            random_seed=42, verbose=0, early_stopping_rounds=cfg["es"],
            use_best_model=True, l2_leaf_reg=cfg["l2"], thread_count=-1,
            bootstrap_type="MVS",
        )
        m.fit(X_tr, y_tr, eval_set=(X_cal, y_cal))
        auc = roc_auc_score(y_cal, m.predict_proba(X_cal)[:, 1])
        if auc > best_auc:
            best_auc, best = auc, m

    calib = IsotonicLogOddsCalibrator().fit(best.predict_proba(X_cal)[:, 1], y_cal)
    raw = best.predict_proba(X_te)[:, 1]
    lo = np.log(raw / (1 - raw + 1e-10) + 1e-10).reshape(-1, 1)
    p = calib.predict_proba(lo)[:, 1]
    test_auc = roc_auc_score(y_te, raw)
    ll = log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"{name}: n_train={len(train_df)} calAUC={best_auc:.4f} | "
          f"TEST auc={test_auc:.4f} logloss={ll:.4f} range=[{p.min():.3f},{p.max():.3f}]")
    imp = sorted(zip(PRICE_FEATURE_NAMES, best.get_feature_importance()), key=lambda x: -x[1])[:5]
    print(f"   top: {', '.join(f'{k}={v:.0f}' for k, v in imp)}")


def main():
    arch = pd.read_csv(os.path.join(HERE, "epochs_archive.csv"))
    coll = pd.read_csv(os.path.join(HERE, "epochs_collected.csv"))
    print(f"archive epochs: {len(arch)} | collected epochs: {len(coll)}")

    # ---- label agreement on overlap ----
    j = arch.merge(coll, on="epoch_start", suffixes=("_a", "_c"))
    agree = (j["target_resolution_a"] == j["target_resolution_c"]).mean()
    ret_a = (j["end_price_a"] - j["start_price_a"]) / j["start_price_a"]
    ret_c = (j["end_price_c"] - j["start_price_c"]) / j["start_price_c"]
    corr = np.corrcoef(ret_a, ret_c)[0, 1]
    feat_corr = np.mean([np.corrcoef(j[f"{f}_a"], j[f"{f}_c"])[0, 1]
                         for f in PRICE_FEATURE_NAMES])
    print(f"\noverlap epochs: {len(j)} | LABEL AGREEMENT: {agree*100:.2f}% | "
          f"epoch-return corr: {corr:.4f} | mean feature corr: {feat_corr:.4f}")

    # ---- transfer arms ----
    a_tr = arch[arch.epoch_start < JUN1]
    a_cal = arch[(arch.epoch_start >= JUN1) & (arch.epoch_start < JUN11)]
    a_te = arch[arch.epoch_start >= JUN11]
    c_tr = coll[coll.epoch_start < JUN1]
    c_cal = coll[(coll.epoch_start >= JUN1) & (coll.epoch_start < JUN11)]
    c_te = coll[coll.epoch_start >= JUN11]
    print(f"\ntest window (collected Jun11+): n={len(c_te)} UP rate={c_te.target_resolution.mean():.4f}\n")
    train_eval(a_tr, a_cal, c_te, "A: archive->collected ")
    train_eval(c_tr, c_cal, c_te, "B: collected->collected")
    train_eval(a_tr, a_cal, a_te, "C: archive->archive   ")


if __name__ == "__main__":
    main()
