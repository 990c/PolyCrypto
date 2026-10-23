"""
Label Dataset Script
====================
Retroactively stamps each row with the actual UP/DOWN outcome.

For each row, compares last_price at timestamp to last_price at
timestamp + 300s. UP = 1 if price increased, DOWN = 0 otherwise.

Usage:
  python src/ml/label_dataset.py --input dataset.csv --output dataset_labeled.csv
"""
import csv
import sys
import os
import argparse
import bisect
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml.features import FeatureEngine

FEATURE_COLS = FeatureEngine.FEATURE_NAMES


def load_dataset(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["timestamp"] = float(r["timestamp"])
                r["last_price"] = float(r.get("last_price") or 0)
                # Convert all feature columns to float
                for col in FEATURE_COLS:
                    r[col] = float(r.get(col, 0))
                if r["last_price"] > 0:
                    rows.append(r)
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def label_rows(rows: list, lookahead_seconds: int = 300, max_gap_seconds: float = 30.0) -> list:
    if not rows:
        return rows

    # Sort by timestamp to guarantee correct bisect behavior
    rows.sort(key=lambda r: r["timestamp"])

    timestamps = [r["timestamp"] for r in rows]
    labeled = []
    skipped = 0

    for row in rows:
        target_ts = row["timestamp"] + lookahead_seconds
        idx = bisect.bisect_left(timestamps, target_ts)

        if idx >= len(rows):
            skipped += 1
            continue

        # Tolerance: ensure the matched row is within max_gap_seconds of target
        actual_gap = abs(rows[idx]["timestamp"] - target_ts)
        if actual_gap > max_gap_seconds:
            skipped += 1
            continue

        future_price = rows[idx]["last_price"]
        current_price = row["last_price"]

        if current_price <= 0:
            skipped += 1
            continue

        row["target_resolution"] = 1 if future_price > current_price else 0
        labeled.append(row)

    logger.info(f"Labeled {len(labeled)} rows. Skipped {skipped} (insufficient future data).")
    return labeled


def save_labeled(rows: list, output_path: str):
    if not rows:
        logger.warning("No rows to save.")
        return

    fieldnames = ["timestamp"] + FEATURE_COLS + ["last_price", "target_resolution"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved labeled dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Label dataset with UP/DOWN outcomes")
    parser.add_argument("--input", default="dataset.csv")
    parser.add_argument("--output", default="dataset_labeled.csv")
    parser.add_argument("--lookahead", type=int, default=300)
    args = parser.parse_args()

    rows = load_dataset(args.input)
    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    labeled = label_rows(rows, args.lookahead)
    save_labeled(labeled, args.output)

    if labeled:
        ups = sum(1 for r in labeled if r["target_resolution"] == 1)
        downs = len(labeled) - ups
        logger.info(f"Distribution: UP={ups} ({ups/len(labeled)*100:.1f}%) | DOWN={downs} ({downs/len(labeled)*100:.1f}%)")


if __name__ == "__main__":
    main()
