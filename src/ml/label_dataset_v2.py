"""
Label Dataset V2 — Epoch-Aligned Labeling
==========================================
Instead of comparing price at T vs T+300s for every 100ms row,
this version:
  1. Groups rows into 5-minute epochs (aligned to 300s boundaries)
  2. For each epoch, computes AGGREGATED features (mean, std, trend)
  3. Labels by comparing epoch_start_price vs epoch_end_price

This produces ~1 row per epoch instead of thousands of nearly-identical rows,
eliminating the correlated-sample problem entirely.

Usage:
  python src/ml/label_dataset_v2.py --input dataset.csv --output dataset_labeled_v2.csv
"""
import csv
import sys
import os
import argparse
import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml.features import FeatureEngine

RAW_FEATURE_COLS = FeatureEngine.FEATURE_NAMES

# Aggregated features per epoch — mean + std + trend for key features
# Plus epoch-level derived features
EPOCH_FEATURE_NAMES = [
    # === PREVIOUS EPOCH (t-1) features ===
    "prev_obi_mean", "prev_cvd_mean", "prev_toxicity_mean", "prev_volatility_mean",
    "prev_spread_bps_mean", "prev_trade_intensity_mean", "prev_weighted_obi_mean",
    "prev_cvd_5m_mean", "prev_cvd_15m_mean", "prev_obi_mean_5m_mean",
    "prev_volume_accel_mean",
    "prev_obi_std", "prev_cvd_std", "prev_spread_bps_std",
    "prev_obi_trend", "prev_cvd_trend", "prev_volatility_trend",
    "prev_return_epoch",
    "prev_return_1m_mean",
    "prev_price_accel_mean",
    "prev_n_samples",

    # === MULTI-EPOCH features (t-2 to t-5 lookback) ===
    # Price momentum across multiple prior epochs
    "return_2ep",          # cumulative return over last 2 epochs (10 min)
    "return_3ep",          # cumulative return over last 3 epochs (15 min)
    "return_5ep",          # cumulative return over last 5 epochs (25 min)
    # Win/loss streak: how many of last N epochs went UP?
    "up_ratio_3ep",        # fraction of last 3 epochs that were UP
    "up_ratio_5ep",        # fraction of last 5 epochs that were UP
    # Momentum consistency
    "streak_sign",         # +N if last N epochs all UP, -N if all DOWN, 0 mixed
    # Volatility regime
    "vol_mean_3ep",        # avg volatility over last 3 epochs
    "vol_mean_5ep",        # avg volatility over last 5 epochs
    "vol_expanding",       # 1 if prev vol > 3-epoch avg vol, else 0
    # CVD regime
    "cvd_mean_3ep",        # avg CVD over last 3 epochs
    "cvd_mean_5ep",        # avg CVD over last 5 epochs
    "cvd_reversal",        # 1 if prev CVD sign differs from 3-epoch avg sign
    # Spread regime
    "spread_mean_3ep",     # avg spread over last 3 epochs
    "spread_widening",     # 1 if prev spread > 3-epoch avg
    # OBI regime
    "obi_mean_3ep",        # avg OBI over last 3 epochs
    # Trade intensity regime
    "intensity_mean_3ep",  # avg trade intensity over last 3 epochs
    "intensity_surge",     # 1 if prev intensity > 1.5x 3-epoch avg

    # === DERIVED cross-epoch features ===
    # Mean reversion signal: did prev epoch reverse from 2-epoch trend?
    "prev_vs_2ep_return",  # prev_return_epoch - return_2ep (divergence)
    # Acceleration: is momentum increasing or decreasing?
    "return_accel",        # prev return - (t-2) return
]


def load_raw_dataset(path: str):
    """Load the raw dataset with all 100ms rows."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ts = float(r["timestamp"])
                lp = float(r.get("last_price") or 0)
                if lp <= 0:
                    continue
                parsed = {"timestamp": ts, "last_price": lp}
                for col in RAW_FEATURE_COLS:
                    parsed[col] = float(r.get(col, 0))
                rows.append(parsed)
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def group_into_epochs(rows: list, epoch_seconds: int = 300):
    """Group rows into 300-second aligned epochs."""
    if not rows:
        return {}

    rows.sort(key=lambda r: r["timestamp"])
    epochs = {}

    for row in rows:
        ts = row["timestamp"]
        epoch_start = int(ts) - (int(ts) % epoch_seconds)
        if epoch_start not in epochs:
            epochs[epoch_start] = []
        epochs[epoch_start].append(row)

    return epochs


def aggregate_epoch(rows: list, epoch_start: int, epoch_seconds: int = 300) -> dict:
    """Compute aggregated features for one epoch."""
    n = len(rows)
    if n < 10:  # too few samples
        return None

    def col_array(col: str) -> np.ndarray:
        return np.array([r[col] for r in rows])

    # Split into quarters for trend calculation
    q1_end = n // 4
    q4_start = 3 * n // 4

    def trend(col: str) -> float:
        vals = col_array(col)
        return float(np.mean(vals[q4_start:])) - float(np.mean(vals[:q1_end]))

    # Price at epoch boundaries
    start_price = rows[0]["last_price"]
    end_price = rows[-1]["last_price"]
    return_epoch = ((end_price - start_price) / start_price * 10000) if start_price > 0 else 0.0

    # Epoch timing - average progress
    timestamps = col_array("timestamp")
    epoch_progress = float(np.mean((timestamps - epoch_start) / epoch_seconds))

    result = {
        "obi_mean": float(np.mean(col_array("obi"))),
        "cvd_mean": float(np.mean(col_array("cvd"))),
        "toxicity_mean": float(np.mean(col_array("toxicity"))),
        "volatility_mean": float(np.mean(col_array("volatility"))),
        "spread_bps_mean": float(np.mean(col_array("spread_bps"))),
        "trade_intensity_mean": float(np.mean(col_array("trade_intensity"))),
        "weighted_obi_mean": float(np.mean(col_array("weighted_obi"))),
        "cvd_5m_mean": float(np.mean(col_array("cvd_5m"))),
        "cvd_15m_mean": float(np.mean(col_array("cvd_15m"))),
        "obi_mean_5m_mean": float(np.mean(col_array("obi_mean_5m"))),
        "volume_accel_mean": float(np.mean(col_array("volume_accel"))),
        # Stds
        "obi_std": float(np.std(col_array("obi"))),
        "cvd_std": float(np.std(col_array("cvd"))),
        "spread_bps_std": float(np.std(col_array("spread_bps"))),
        # Trends
        "obi_trend": trend("obi"),
        "cvd_trend": trend("cvd"),
        "volatility_trend": trend("volatility"),
        # Price
        "return_epoch": return_epoch,
        "return_1m_mean": float(np.mean(col_array("return_1m"))),
        "return_5m_last": float(rows[-1]["return_5m"]),
        "price_accel_mean": float(np.mean(col_array("price_accel"))),
        "dist_from_high_15m_last": float(rows[-1]["dist_from_high_15m"]),
        "dist_from_low_15m_last": float(rows[-1]["dist_from_low_15m"]),
        # Meta
        "epoch_progress": epoch_progress,
        "n_samples": float(n),
    }

    result["epoch_start"] = epoch_start
    result["start_price"] = start_price
    result["end_price"] = end_price

    return result


def build_feature_row(prior_aggs: list) -> dict:
    """Builds the 40-feature V2 row from the 5 prior epoch aggregates
    (ordered oldest-first: [t-5, t-4, t-3, t-2, t-1]).

    Shared by the offline labeler and experiment pipelines so the feature
    construction can never drift between them.
    """
    prev_agg = prior_aggs[4]  # t-1 (most recent)

    # === Previous epoch (t-1) features ===
    row = {
        "prev_obi_mean": prev_agg["obi_mean"],
        "prev_cvd_mean": prev_agg["cvd_mean"],
        "prev_toxicity_mean": prev_agg["toxicity_mean"],
        "prev_volatility_mean": prev_agg["volatility_mean"],
        "prev_spread_bps_mean": prev_agg["spread_bps_mean"],
        "prev_trade_intensity_mean": prev_agg["trade_intensity_mean"],
        "prev_weighted_obi_mean": prev_agg["weighted_obi_mean"],
        "prev_cvd_5m_mean": prev_agg["cvd_5m_mean"],
        "prev_cvd_15m_mean": prev_agg["cvd_15m_mean"],
        "prev_obi_mean_5m_mean": prev_agg["obi_mean_5m_mean"],
        "prev_volume_accel_mean": prev_agg["volume_accel_mean"],
        "prev_obi_std": prev_agg["obi_std"],
        "prev_cvd_std": prev_agg["cvd_std"],
        "prev_spread_bps_std": prev_agg["spread_bps_std"],
        "prev_obi_trend": prev_agg["obi_trend"],
        "prev_cvd_trend": prev_agg["cvd_trend"],
        "prev_volatility_trend": prev_agg["volatility_trend"],
        "prev_return_epoch": prev_agg["return_epoch"],
        "prev_return_1m_mean": prev_agg["return_1m_mean"],
        "prev_price_accel_mean": prev_agg["price_accel_mean"],
        "prev_n_samples": prev_agg["n_samples"],
    }

    # === Multi-epoch price momentum ===
    returns = [a["return_epoch"] for a in prior_aggs]  # [t-5, t-4, t-3, t-2, t-1]
    # Cumulative returns (sum of per-epoch bps returns)
    row["return_2ep"] = returns[3] + returns[4]        # t-2 + t-1
    row["return_3ep"] = sum(returns[2:])               # t-3 to t-1
    row["return_5ep"] = sum(returns)                   # t-5 to t-1

    # UP ratio: fraction of last N epochs that went UP
    directions = [1 if r > 0 else 0 for r in returns]
    row["up_ratio_3ep"] = sum(directions[2:]) / 3.0
    row["up_ratio_5ep"] = sum(directions) / 5.0

    # Streak: consecutive same-direction epochs from most recent
    streak = 0
    last_dir = directions[4]
    for d in reversed(directions):
        if d == last_dir:
            streak += 1
        else:
            break
    row["streak_sign"] = streak if last_dir == 1 else -streak

    # === Volatility regime ===
    vols = [a["volatility_mean"] for a in prior_aggs]
    row["vol_mean_3ep"] = float(np.mean(vols[2:]))
    row["vol_mean_5ep"] = float(np.mean(vols))
    vol_3ep_avg = row["vol_mean_3ep"]
    row["vol_expanding"] = 1.0 if vols[4] > vol_3ep_avg and vol_3ep_avg > 0 else 0.0

    # === CVD regime ===
    cvds = [a["cvd_mean"] for a in prior_aggs]
    row["cvd_mean_3ep"] = float(np.mean(cvds[2:]))
    row["cvd_mean_5ep"] = float(np.mean(cvds))
    cvd_3ep_sign = 1 if row["cvd_mean_3ep"] > 0 else -1
    prev_cvd_sign = 1 if cvds[4] > 0 else -1
    row["cvd_reversal"] = 1.0 if prev_cvd_sign != cvd_3ep_sign else 0.0

    # === Spread regime ===
    spreads = [a["spread_bps_mean"] for a in prior_aggs]
    row["spread_mean_3ep"] = float(np.mean(spreads[2:]))
    row["spread_widening"] = 1.0 if spreads[4] > row["spread_mean_3ep"] else 0.0

    # === OBI regime ===
    obis = [a["obi_mean"] for a in prior_aggs]
    row["obi_mean_3ep"] = float(np.mean(obis[2:]))

    # === Trade intensity regime ===
    intensities = [a["trade_intensity_mean"] for a in prior_aggs]
    row["intensity_mean_3ep"] = float(np.mean(intensities[2:]))
    int_3ep_avg = row["intensity_mean_3ep"]
    row["intensity_surge"] = 1.0 if int_3ep_avg > 0 and intensities[4] > 1.5 * int_3ep_avg else 0.0

    # === Derived cross-epoch features ===
    row["prev_vs_2ep_return"] = returns[4] - row["return_2ep"]
    row["return_accel"] = returns[4] - returns[3]  # t-1 return minus t-2 return

    return row


def label_epochs(epochs: dict, epoch_seconds: int = 300) -> list:
    """
    Use features from previous 1-5 epochs to predict CURRENT epoch's direction.
    No data leakage — only information available before the prediction epoch.

    Requires 5 prior epochs of context, so first 5 epochs are skipped.
    """
    sorted_starts = sorted(epochs.keys())

    # Pre-compute aggregated features for all epochs
    agg_by_start = {}
    for epoch_start in sorted_starts:
        agg = aggregate_epoch(epochs[epoch_start], epoch_start, epoch_seconds)
        if agg is not None:
            agg_by_start[epoch_start] = agg

    labeled = []

    for i in range(5, len(sorted_starts)):
        curr_start = sorted_starts[i]
        if curr_start not in agg_by_start:
            continue

        # Gather the 5 prior epochs (t-1 through t-5)
        prior_starts = sorted_starts[i - 5:i]
        prior_aggs = [agg_by_start.get(s) for s in prior_starts]
        if any(a is None for a in prior_aggs):
            continue

        curr_agg = agg_by_start[curr_start]

        # Label: did price go up in the CURRENT epoch?
        current_start_price = curr_agg["start_price"]
        current_end_price = curr_agg["end_price"]
        if current_start_price <= 0:
            continue

        row = build_feature_row(prior_aggs)
        row["epoch_start"] = curr_start
        row["start_price"] = current_start_price
        row["end_price"] = current_end_price
        row["target_resolution"] = 1 if current_end_price > current_start_price else 0

        labeled.append(row)

    return labeled


def save_labeled(rows: list, output_path: str):
    if not rows:
        logger.warning("No rows to save.")
        return

    fieldnames = ["epoch_start"] + EPOCH_FEATURE_NAMES + ["start_price", "end_price", "target_resolution"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} epoch-aggregated labeled rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Label dataset V2 — epoch-aligned aggregation")
    parser.add_argument("--input", default="dataset.csv")
    parser.add_argument("--output", default="dataset_labeled_v2.csv")
    parser.add_argument("--epoch-seconds", type=int, default=300)
    args = parser.parse_args()

    logger.info(f"Loading raw dataset from {args.input}...")
    rows = load_raw_dataset(args.input)
    logger.info(f"Loaded {len(rows)} raw rows")

    logger.info(f"Grouping into {args.epoch_seconds}s epochs...")
    epochs = group_into_epochs(rows, args.epoch_seconds)
    logger.info(f"Found {len(epochs)} epochs")

    logger.info("Aggregating features and labeling...")
    labeled = label_epochs(epochs, args.epoch_seconds)

    save_labeled(labeled, args.output)

    if labeled:
        ups = sum(1 for r in labeled if r["target_resolution"] == 1)
        downs = len(labeled) - ups
        logger.info(f"Distribution: UP={ups} ({ups / len(labeled) * 100:.1f}%) | DOWN={downs} ({downs / len(labeled) * 100:.1f}%)")


if __name__ == "__main__":
    main()
