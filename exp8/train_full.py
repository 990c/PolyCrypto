"""Exp 8 Step 2: train the price-only model on the multi-year archive dataset.

Chronological protocol: train < 2026-01, cal 2026-01..03, test 2026-04..06.
Extra evaluation: the same model scored on our REAL collected epochs
(Jun 11-26, binance.us feed) — the live-parity test bed from Step 1.
Artifacts saved for the dry-run gate.
"""
import os
import pickle
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

TS = lambda y, m, d: int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())

CONFIGS = [
    {"depth": 4, "lr": 0.01, "l2": 10.0, "iters": 2000, "es": 200},
    {"depth": 6, "lr": 0.03, "l2": 5.0, "iters": 1000, "es": 150},
    {"depth": 8, "lr": 0.05, "l2": 3.0, "iters": 1000, "es": 100},
]


def report(name, y, raw, p):
    from sklearn.metrics import log_loss, roc_auc_score

    auc = roc_auc_score(y, raw)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"\n{name}: n={len(y)} AUC={auc:.4f} logloss={ll:.4f} "
          f"range=[{p.min():.3f},{p.max():.3f}]")
    print("  gate table (trade when |p-0.5| >= t):")
    for t in (0.08, 0.15, 0.25):
        up = p >= 0.5 + t
        dn = p <= 0.5 - t
        mask = up | dn
        if mask.sum() == 0:
            continue
        pred = np.where(up[mask], 1, 0)
        acc = float((pred == y[mask]).mean())
        print(f"    t={t:.2f}: n={int(mask.sum()):5d} ({mask.sum()/len(y)*100:4.1f}%) acc={acc:.4f}")


def main():
    df = pd.read_csv(os.path.join(HERE, "epochs_multiyear.csv"))
    print(f"multi-year epochs: {len(df)} | "
          f"{datetime.utcfromtimestamp(df.epoch_start.min()).date()} -> "
          f"{datetime.utcfromtimestamp(df.epoch_start.max()).date()} | "
          f"UP rate {df.target_resolution.mean():.4f}")

    tr = df[df.epoch_start < TS(2026, 1, 1)]
    cal = df[(df.epoch_start >= TS(2026, 1, 1)) & (df.epoch_start < TS(2026, 4, 1))]
    te = df[df.epoch_start >= TS(2026, 4, 1)]
    print(f"train={len(tr)} cal={len(cal)} test={len(te)}")

    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    X_tr, y_tr = tr[PRICE_FEATURE_NAMES].to_numpy(), tr.target_resolution.to_numpy()
    X_cal, y_cal = cal[PRICE_FEATURE_NAMES].to_numpy(), cal.target_resolution.to_numpy()

    best_auc, best, best_name = -1, None, ""
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
        name = f"d={cfg['depth']} lr={cfg['lr']}"
        print(f"  config [{name}]: calAUC={auc:.4f} iter={m.get_best_iteration()}")
        if auc > best_auc:
            best_auc, best, best_name = auc, m, name
    print(f"best: [{best_name}] calAUC={best_auc:.4f}")

    calib = IsotonicLogOddsCalibrator().fit(best.predict_proba(X_cal)[:, 1], y_cal)

    def calibrated(X):
        raw = best.predict_proba(X)[:, 1]
        lo = np.log(raw / (1 - raw + 1e-10) + 1e-10).reshape(-1, 1)
        return raw, calib.predict_proba(lo)[:, 1]

    raw_te, p_te = calibrated(te[PRICE_FEATURE_NAMES].to_numpy())
    report("TEST archive 2026-04..06 (out-of-time)", te.target_resolution.to_numpy(), raw_te, p_te)

    fi = sorted(zip(PRICE_FEATURE_NAMES, best.get_feature_importance()), key=lambda x: -x[1])[:6]
    print("top importances:", ", ".join(f"{k}={v:.1f}" for k, v in fi))

    # Live-parity check needs our own collected feed (build_datasets.py collected);
    # the archive-only path still trains and saves without it.
    collected_path = os.path.join(HERE, "epochs_collected.csv")
    if os.path.exists(collected_path):
        coll = pd.read_csv(collected_path)
        coll_te = coll[coll.epoch_start >= TS(2026, 6, 11)]
        raw_c, p_c = calibrated(coll_te[PRICE_FEATURE_NAMES].to_numpy())
        report("TEST real collected feed Jun 11-26 (live parity)",
               coll_te.target_resolution.to_numpy(), raw_c, p_c)
    else:
        print("skipping live-parity test: epochs_collected.csv not found")

    best.save_model(os.path.join(HERE, "catboost_price_multiyear.cbm"))
    with open(os.path.join(HERE, "catboost_price_multiyear_calibrator.pkl"), "wb") as f:
        pickle.dump(calib, f)
    print("\nartifacts saved: exp8/catboost_price_multiyear.cbm + _calibrator.pkl")


if __name__ == "__main__":
    main()
