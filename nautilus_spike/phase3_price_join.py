"""Phase 3: historical price-join study.

For a sample of out-of-sample test epochs (last 20% of dataset_labeled_v3.csv):
  1. model prediction (production .cbm + isotonic calibrator, computed offline)
  2. REAL Polymarket UP price at ~t+90s (actual executed trades via data-api)
  3. actual outcome (label)

Then simulates the EV-gated strategy buying at those real prices — the honest
version of the paper engine's fantasy fills.

Caches API responses in pm_price_cache.json (resumable).
"""
import csv
import json
import os
import sys
import time

import numpy as np
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

ROOT = REPO_ROOT
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pm_price_cache.json")
SAMPLE_EVERY = 8
WINDOW_LO, WINDOW_HI = 60, 130  # accept trades in t+60..t+130, pick closest to t+90
TARGET_T = 90
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
SLEEP = 0.12


def http_get(url, params):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.ok:
                time.sleep(SLEEP)
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.0 + attempt)
    return None


def price_up_at_t90(epoch_start: int, cache: dict) -> float | None:
    """UP-token price of the trade closest to t+90s, from real executed trades."""
    key = str(epoch_start)
    if key in cache:
        return cache[key]

    slug = f"btc-updown-5m-{epoch_start}"
    evs = http_get(f"{GAMMA}/events", {"slug": slug})
    if not evs:
        cache[key] = None
        return None
    cond = evs[0]["markets"][0].get("conditionId")
    if not cond:
        cache[key] = None
        return None

    best = None  # (dist_to_t90, up_price)
    for offset in range(0, 4000, 500):
        trades = http_get(f"{DATA_API}/trades", {"market": cond, "limit": 500, "offset": offset})
        if not trades:
            break
        min_rel = None
        for t in trades:
            try:
                rel = float(t["timestamp"]) - epoch_start
                price = float(t["price"])
                outcome = (t.get("outcome") or "").upper()
            except (KeyError, ValueError, TypeError):
                continue
            min_rel = rel if min_rel is None else min(min_rel, rel)
            if WINDOW_LO <= rel <= WINDOW_HI and outcome in ("UP", "DOWN"):
                up_price = price if outcome == "UP" else round(1.0 - price, 4)
                dist = abs(rel - TARGET_T)
                if best is None or dist < best[0]:
                    best = (dist, up_price)
        # pages are newest-first: once this page is entirely before the window, stop
        if min_rel is not None and min_rel < WINDOW_LO:
            break
        if len(trades) < 500:
            break

    cache[key] = best[1] if best else None
    return cache[key]


def main():
    import pickle

    from catboost import CatBoostClassifier

    from ml.label_dataset_v2 import EPOCH_FEATURE_NAMES  # noqa: F401 (column source)
    from ml.label_dataset_v3 import MODEL_FEATURE_NAMES

    rows = list(csv.DictReader(open(f"{ROOT}/dataset_labeled_v3.csv")))
    test_rows = rows[int(len(rows) * 0.80):]
    sampled = test_rows[::SAMPLE_EVERY]
    print(f"test epochs: {len(test_rows)} | sampled: {len(sampled)}")

    model = CatBoostClassifier()
    model.load_model(f"{ROOT}/catboost_model_v2.cbm")
    calibrator = pickle.load(open(f"{ROOT}/catboost_model_v2_calibrator.pkl", "rb"))

    X = np.array([[float(r.get(c, 0)) for c in MODEL_FEATURE_NAMES] for r in sampled])
    raw = model.predict_proba(X)[:, 1]
    lo = np.log(raw / (1 - raw + 1e-10) + 1e-10).reshape(-1, 1)
    prob_up = calibrator.predict_proba(lo)[:, 1]

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    results = []
    for i, r in enumerate(sampled):
        es = int(float(r["epoch_start"]))
        c_up = price_up_at_t90(es, cache)
        if i % 50 == 0:
            json.dump(cache, open(CACHE, "w"))
            print(f"  {i}/{len(sampled)} fetched...", flush=True)
        if c_up is None or not (0.02 <= c_up <= 0.98):
            continue
        results.append({
            "epoch": es, "prob_up": float(prob_up[i]),
            "price_up_t90": c_up, "label": int(r["target_resolution"]),
        })
    json.dump(cache, open(CACHE, "w"))
    out_path = os.path.join(os.path.dirname(CACHE), "phase3_joined.json")
    json.dump(results, open(out_path, "w"))
    print(f"joined rows: {len(results)} -> {out_path}")


if __name__ == "__main__":
    main()
