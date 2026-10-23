"""Exp 6 fetch: extend the price join to the CAL region (blend fitting) and a
denser TEST sample (evaluation). Reuses pm_price_cache.json — already-fetched
epochs cost nothing."""
import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3_price_join import CACHE, price_up_at_t90

ROOT = REPO_ROOT
SAMPLE_EVERY = 4


def main():
    import pickle

    from catboost import CatBoostClassifier

    from ml.label_dataset_v3 import MODEL_FEATURE_NAMES

    rows = list(csv.DictReader(open(f"{ROOT}/dataset_labeled_v3.csv")))
    n = len(rows)
    tr_end, cal_end = int(n * 0.70), int(n * 0.80)
    regions = {
        "cal": rows[tr_end:cal_end][::SAMPLE_EVERY],
        "test": rows[cal_end:][::SAMPLE_EVERY],
    }

    model = CatBoostClassifier()
    model.load_model(f"{ROOT}/catboost_model_v2.cbm")
    calibrator = pickle.load(open(f"{ROOT}/catboost_model_v2_calibrator.pkl", "rb"))

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out = []
    total = sum(len(v) for v in regions.values())
    done = 0
    for region, sample in regions.items():
        X = np.array([[float(r.get(c, 0)) for c in MODEL_FEATURE_NAMES] for r in sample])
        raw = model.predict_proba(X)[:, 1]
        lo = np.log(raw / (1 - raw + 1e-10) + 1e-10).reshape(-1, 1)
        prob = calibrator.predict_proba(lo)[:, 1]
        for i, r in enumerate(sample):
            es = int(float(r["epoch_start"]))
            c_up = price_up_at_t90(es, cache)
            done += 1
            if done % 100 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  {done}/{total} fetched...", flush=True)
            if c_up is None or not (0.02 <= c_up <= 0.98):
                continue
            out.append({"region": region, "epoch": es, "prob_up": float(prob[i]),
                        "price_up_t90": c_up, "label": int(r["target_resolution"])})
    json.dump(cache, open(CACHE, "w"))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp6_joined.json")
    json.dump(out, open(path, "w"))
    by_region = {k: sum(1 for o in out if o["region"] == k) for k in regions}
    print(f"joined: {by_region} -> {path}")


if __name__ == "__main__":
    main()
