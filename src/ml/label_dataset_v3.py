"""
Label Dataset V3 — Streaming, Epoch-Aligned, Hybrid Features
=============================================================
Replaces label_dataset_v2's in-memory pipeline (which cannot hold 22M+ raw rows)
with a single streaming pass, and adds current-epoch features computed from the
FIRST 90 SECONDS of the epoch being predicted.

Live semantics: the predictor waits until t+90s into an epoch, then predicts the
epoch's start->end direction. The market has also seen the first 90s, so the
comparison against live prices is fair (no leakage relative to the instrument).

Requires 5 CONTIGUOUS prior epochs (v2 silently allowed gaps across collection
outages; v3 resets context on any gap, matching what the live ring buffer sees).

Usage:
  python src/ml/label_dataset_v3.py --input dataset.csv --output dataset_labeled_v3.csv
"""
import argparse
import csv
import sys
import os
import time
from collections import deque

import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml.features import FeatureEngine
from ml.label_dataset_v2 import EPOCH_FEATURE_NAMES, aggregate_epoch, build_feature_row

RAW_FEATURE_COLS = FeatureEngine.FEATURE_NAMES
EPOCH_SECONDS = 300
CURR_WINDOW_SECONDS = 90
MIN_ROWS_PER_EPOCH = 10

CURR_FEATURE_NAMES = [
    "curr_return_90s", "curr_obi_mean_90s", "curr_cvd_mean_90s",
    "curr_weighted_obi_mean_90s", "curr_toxicity_mean_90s",
    "curr_trade_intensity_mean_90s", "curr_volume_accel_mean_90s",
    "curr_n_samples_90s",
]

# Features the production model trains on. n_samples features are excluded:
# training data is 100ms-cadence (~2,600 rows/epoch) but live prediction runs at
# ~2s ticks (~150 rows/epoch), so sample counts are pure train/serve skew.
MODEL_FEATURE_NAMES = [
    c for c in EPOCH_FEATURE_NAMES + CURR_FEATURE_NAMES
    if c not in ("prev_n_samples", "curr_n_samples_90s")
]


def build_curr_features(rows_90s: list, epoch_start_price: float) -> dict:
    """First-90-seconds features of the epoch being predicted.
    All means are scale-free w.r.t. tick cadence except the sample count."""
    def mean(c: str) -> float:
        return float(np.mean([r[c] for r in rows_90s])) if rows_90s else 0.0

    last = rows_90s[-1]["last_price"] if rows_90s else 0.0
    ret = ((last - epoch_start_price) / epoch_start_price * 10000
           if epoch_start_price > 0 and last > 0 else 0.0)
    return {
        "curr_return_90s": ret,
        "curr_obi_mean_90s": mean("obi"),
        "curr_cvd_mean_90s": mean("cvd"),
        "curr_weighted_obi_mean_90s": mean("weighted_obi"),
        "curr_toxicity_mean_90s": mean("toxicity"),
        "curr_trade_intensity_mean_90s": mean("trade_intensity"),
        "curr_volume_accel_mean_90s": mean("volume_accel"),
        "curr_n_samples_90s": float(len(rows_90s)),
    }


def stream_label(input_path: str, output_path: str,
                 epoch_seconds: int = EPOCH_SECONDS) -> int:
    """Single pass over the raw CSV; constant memory. Returns rows written."""
    t0 = time.time()
    prior: deque = deque(maxlen=5)
    prior_starts: deque = deque(maxlen=5)
    cur_start = None
    cur_rows: list = []
    n_in = n_out = 0

    fieldnames = (["epoch_start"] + EPOCH_FEATURE_NAMES + CURR_FEATURE_NAMES
                  + ["start_price", "end_price", "target_resolution"])

    with open(output_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        def finalize(epoch_start: int, rows: list):
            nonlocal n_out
            agg = (aggregate_epoch(rows, epoch_start, epoch_seconds)
                   if len(rows) >= MIN_ROWS_PER_EPOCH else None)
            if agg is None:
                prior.clear()
                prior_starts.clear()
                return
            contiguous = (
                len(prior) == 5
                and all(prior_starts[i + 1] - prior_starts[i] == epoch_seconds for i in range(4))
                and epoch_start - prior_starts[-1] == epoch_seconds
            )
            if contiguous and agg["start_price"] > 0:
                row = build_feature_row(list(prior))
                rows_90s = [r for r in rows if r["timestamp"] < epoch_start + CURR_WINDOW_SECONDS]
                row.update(build_curr_features(rows_90s, agg["start_price"]))
                row["epoch_start"] = epoch_start
                row["start_price"] = agg["start_price"]
                row["end_price"] = agg["end_price"]
                row["target_resolution"] = 1 if agg["end_price"] > agg["start_price"] else 0
                writer.writerow(row)
                n_out += 1
            prior.append(agg)
            prior_starts.append(epoch_start)

        with open(input_path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    ts = float(r["timestamp"])
                    lp = float(r.get("last_price") or 0)
                    if lp <= 0:
                        continue
                    parsed = {"timestamp": ts, "last_price": lp}
                    for c in RAW_FEATURE_COLS:
                        parsed[c] = float(r.get(c, 0))
                except (ValueError, KeyError, TypeError):
                    continue
                n_in += 1
                es = int(ts) - (int(ts) % epoch_seconds)
                if cur_start is None:
                    cur_start = es
                if es != cur_start:
                    if es < cur_start:
                        prior.clear()
                        prior_starts.clear()
                        cur_rows = []
                    else:
                        finalize(cur_start, cur_rows)
                        cur_rows = []
                    cur_start = es
                cur_rows.append(parsed)
                if n_in % 2_000_000 == 0:
                    logger.info(f"{n_in / 1e6:.0f}M rows | epochs: {n_out} | {time.time() - t0:.0f}s")

        if cur_start is not None:
            finalize(cur_start, cur_rows)

    logger.info(f"Done: {n_in} raw rows -> {n_out} labeled epochs "
                f"in {time.time() - t0:.0f}s -> {output_path}")
    return n_out


def main():
    parser = argparse.ArgumentParser(description="Label dataset V3 — streaming hybrid labeler")
    parser.add_argument("--input", default="dataset.csv")
    parser.add_argument("--output", default="dataset_labeled_v3.csv")
    parser.add_argument("--epoch-seconds", type=int, default=EPOCH_SECONDS)
    args = parser.parse_args()
    stream_label(args.input, args.output, args.epoch_seconds)


if __name__ == "__main__":
    main()
