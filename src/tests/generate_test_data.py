"""
Generate Synthetic Dataset for Backtest Verification
=====================================================
Creates a labeled dataset with realistic microstructure feature distributions
and directional bias matching the label for verifiable backtesting.

Usage:
  python src/tests/generate_test_data.py
  python src/backtest.py --dataset test_dataset_labeled.csv
"""
import csv
import random
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml.features import FeatureEngine

FEATURE_COLS = FeatureEngine.FEATURE_NAMES
TICK_SECONDS = 10


def _bps(now: float, then: float) -> float:
    return (now - then) / then * 10000 if then > 0 else 0.0


def _multi_timeframe_features(history: list) -> dict:
    """Multi-timeframe and price features, defined as in FeatureEngine, over 10s ticks.

    history: (price, cvd, obi, volume) per tick, newest last. cvd and volume stand
    in for the per-trade signed flow and size that FeatureEngine aggregates.
    """
    per_min = 60 // TICK_SECONDS
    prices = [h[0] for h in history]
    price = prices[-1]

    def price_ago(minutes: int) -> float:
        n = minutes * per_min
        return prices[-1 - n] if len(prices) > n else prices[0]

    last_5m, last_15m = history[-5 * per_min:], history[-15 * per_min:]
    vol_1m = sum(h[3] for h in history[-per_min:])
    avg_vol_per_min_5m = sum(h[3] for h in last_5m) / 5.0 or 1.0
    return_1m = _bps(price, price_ago(1))
    return {
        "cvd_5m": sum(h[1] for h in last_5m),
        "cvd_15m": sum(h[1] for h in last_15m),
        "obi_mean_5m": sum(h[2] for h in last_5m) / len(last_5m),
        "volume_accel": vol_1m / avg_vol_per_min_5m - 1.0,
        "return_1m": return_1m,
        "return_5m": _bps(price, price_ago(5)),
        "return_15m": _bps(price, price_ago(15)),
        "price_accel": return_1m - _bps(price_ago(1), price_ago(2)),
        "dist_from_high_15m": _bps(price, max(h[0] for h in last_15m)),
        "dist_from_low_15m": _bps(price, min(h[0] for h in last_15m)),
    }


def generate_synthetic_dataset(output_path: str, num_epochs: int = 200, ticks_per_epoch: int = 90):
    """
    Each epoch has ~90 ticks (one per ~10s over 900s).
    Features have directional bias correlated with the label, with realistic noise.
    """
    rows = []
    history = []
    base_price = 87000.0
    timestamp = 1700000000.0

    for epoch in range(num_epochs):
        goes_up = random.random() > 0.5
        label = 1 if goes_up else 0

        drift = random.uniform(15, 100) if goes_up else random.uniform(-100, -15)
        price_start = base_price + random.uniform(-2000, 2000)

        # Epoch-level regime: signal is NOISY — only ~60% directional correlation
        # This simulates real market conditions where features are weakly predictive
        signal_strength = random.uniform(0.05, 0.20)  # Weak signal
        if random.random() < 0.35:
            signal_strength *= -1  # 35% of epochs have contra-directional features (noise)

        regime_obi = signal_strength if goes_up else -signal_strength
        regime_cvd = (signal_strength * 100) if goes_up else (-signal_strength * 100)
        regime_tox = (0.5 - signal_strength * 0.5) if goes_up else (0.5 + signal_strength * 0.5)

        prev_obi = 0.0
        prev_cvd = 0.0

        for tick in range(ticks_per_epoch):
            t = tick / ticks_per_epoch
            price = price_start + drift * t + random.gauss(0, 8)

            noise = random.gauss(0, 0.08)
            obi = regime_obi + noise
            obi = max(-1.0, min(1.0, obi))

            cvd = regime_cvd + random.gauss(0, 5)
            toxicity = regime_tox + random.gauss(0, 0.05)
            toxicity = max(0.0, min(1.0, toxicity))

            volatility = abs(random.gauss(20, 8))
            spread_bps = abs(random.gauss(1.5, 0.5))
            vwap_distance = random.gauss(0, 15) + (5 if goes_up else -5)
            trade_intensity = max(0, random.gauss(50, 15))

            # Gradients (delta from 2 ticks ago)
            obi_gradient = obi - prev_obi if tick >= 2 else 0.0
            cvd_gradient = cvd - prev_cvd if tick >= 2 else 0.0

            # Weighted OBI: correlated with flat OBI but with added noise
            # Simulates the effect of inverse-distance weighting on deeper book levels
            weighted_obi = obi * random.uniform(0.7, 1.1) + random.gauss(0, 0.04)
            weighted_obi = max(-1.0, min(1.0, weighted_obi))

            history.append((price, cvd, obi, trade_intensity))
            multi_timeframe = _multi_timeframe_features(history)

            rows.append({
                "timestamp": timestamp,
                "obi": round(obi, 6),
                "cvd": round(cvd, 6),
                "toxicity": round(toxicity, 6),
                "volatility": round(volatility, 6),
                "spread_bps": round(spread_bps, 6),
                "vwap_distance": round(vwap_distance, 6),
                "trade_intensity": round(trade_intensity, 6),
                "obi_gradient": round(obi_gradient, 6),
                "cvd_gradient": round(cvd_gradient, 6),
                "weighted_obi": round(weighted_obi, 6),
                **{name: round(value, 6) for name, value in multi_timeframe.items()},
                "last_price": round(price, 2),
                "target_resolution": label,
            })

            prev_obi = obi
            prev_cvd = cvd
            timestamp += 10

        timestamp += 60

    fieldnames = ["timestamp"] + FEATURE_COLS + ["last_price", "target_resolution"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ups = sum(1 for r in rows if r["target_resolution"] == 1)
    downs = len(rows) - ups
    print(f"Generated {len(rows)} rows across {num_epochs} epochs -> {output_path}")
    print(f"Class balance: UP={ups} ({ups/len(rows)*100:.1f}%) | DOWN={downs} ({downs/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    output = "test_dataset_labeled.csv"
    generate_synthetic_dataset(output, num_epochs=200, ticks_per_epoch=90)
