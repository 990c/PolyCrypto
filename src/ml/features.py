import threading
import numpy as np
import time
from collections import deque
from typing import List, Any, Dict


class FeatureEngine:
    """
    Computes microstructure features from streaming Binance data.
    All features are computed incrementally over rolling windows.

    Three timeframes: 1min, 5min, 15min.
    """

    FEATURE_NAMES = [
        # --- Single-window (60s) microstructure ---
        "obi", "cvd", "toxicity", "volatility",
        "spread_bps", "vwap_distance", "trade_intensity",
        "obi_gradient", "cvd_gradient", "weighted_obi",
        # --- Multi-timeframe ---
        "cvd_5m", "cvd_15m",
        "obi_mean_5m",
        "volume_accel",
        # --- Price-derived ---
        "return_1m", "return_5m", "return_15m",
        "price_accel",
        "dist_from_high_15m", "dist_from_low_15m",
    ]

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()

        # Trade buffer: (timestamp, price, qty, is_buyer_maker)
        # Keep 15 min of trades for multi-timeframe features
        self.trades: deque = deque()
        self.last_price = 0.0

        # OBI history for gradient and rolling mean (sampled ~1/sec)
        self.obi_history: deque = deque(maxlen=300)  # ~5 min of snapshots
        self.cvd_history: deque = deque(maxlen=20)

        # Price history: (timestamp, price) sampled each compute call
        self.price_history: deque = deque(maxlen=900)  # ~15 min at 1/sec

    def add_trade(self, price: float, qty: float, is_buyer_maker: bool):
        now = time.time()
        with self._lock:
            self.trades.append((now, price, qty, is_buyer_maker))
            self.last_price = price
            self._cleanup(now)

    def _cleanup(self, now: float):
        cutoff = now - 900  # keep 15 minutes of trades
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()

    def _trades_in_window(self, trades: list, now: float, seconds: int) -> list:
        cutoff = now - seconds
        return [t for t in trades if t[0] >= cutoff]

    def compute_features(self, bids: List[List[Any]], asks: List[List[Any]]) -> Dict[str, float]:
        now = time.time()
        with self._lock:
            self._cleanup(now)
            trades_snapshot = list(self.trades)
            last_price = self.last_price

        # Record price sample for momentum features
        if last_price > 0:
            self.price_history.append((now, last_price))

        # --- Order Book Imbalance (OBI) [-1, 1] ---
        bid_vol = sum(float(lvl[1]) for lvl in bids) if bids else 0.0
        ask_vol = sum(float(lvl[1]) for lvl in asks) if asks else 0.0
        total_book = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / total_book if total_book > 0 else 0.0

        # --- Weighted OBI (inverse-distance weighted) [-1, 1] ---
        w_bid = sum(float(lvl[1]) / (i + 1) for i, lvl in enumerate(bids)) if bids else 0.0
        w_ask = sum(float(lvl[1]) / (i + 1) for i, lvl in enumerate(asks)) if asks else 0.0
        w_total = w_bid + w_ask
        weighted_obi = (w_bid - w_ask) / w_total if w_total > 0 else 0.0

        # --- Spread in basis points ---
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else last_price
        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid > 0 and best_ask > best_bid else 0.0

        # --- Trade-derived features (60s window) ---
        trades_1m = self._trades_in_window(trades_snapshot, now, 60)
        trade_count = len(trades_1m)

        if trade_count > 0:
            prices = np.array([t[1] for t in trades_1m])
            qtys = np.array([t[2] for t in trades_1m])
            makers = np.array([t[3] for t in trades_1m])

            signs = np.where(makers, -1.0, 1.0)
            cvd = float(np.sum(qtys * signs))

            total_vol = float(np.sum(qtys))
            sell_vol = float(np.sum(qtys[makers]))
            toxicity = sell_vol / total_vol if total_vol > 0 else 0.5

            if trade_count > 1:
                log_returns = np.diff(np.log(prices[prices > 0]))
                volatility = float(np.std(log_returns)) if len(log_returns) > 0 else 0.0
            else:
                volatility = 0.0

            vwap = float(np.sum(prices * qtys) / np.sum(qtys)) if total_vol > 0 else mid
            vwap_distance = ((last_price - vwap) / vwap * 10000) if vwap > 0 else 0.0

            elapsed = now - trades_1m[0][0] if trade_count > 1 else self.window_seconds
            trade_intensity = trade_count / max(elapsed, 1.0)
        else:
            cvd = 0.0
            toxicity = 0.5
            volatility = 0.0
            vwap_distance = 0.0
            trade_intensity = 0.0

        # --- OBI / CVD gradients ---
        self.obi_history.append((now, obi))
        self.cvd_history.append(cvd)

        obi_gradient = 0.0
        if len(self.obi_history) >= 3:
            obi_gradient = self.obi_history[-1][1] - self.obi_history[-3][1]

        cvd_gradient = 0.0
        if len(self.cvd_history) >= 3:
            recent = list(self.cvd_history)
            cvd_gradient = recent[-1] - recent[-3]

        # =============================================
        # MULTI-TIMEFRAME FEATURES
        # =============================================

        # --- CVD 5min & 15min ---
        trades_5m = self._trades_in_window(trades_snapshot, now, 300)
        trades_15m = trades_snapshot  # already capped at 15 min

        cvd_5m = 0.0
        vol_5m = 0.0
        if trades_5m:
            qtys_5 = np.array([t[2] for t in trades_5m])
            signs_5 = np.where(np.array([t[3] for t in trades_5m]), -1.0, 1.0)
            cvd_5m = float(np.sum(qtys_5 * signs_5))
            vol_5m = float(np.sum(qtys_5))

        cvd_15m = 0.0
        if trades_15m:
            qtys_15 = np.array([t[2] for t in trades_15m])
            signs_15 = np.where(np.array([t[3] for t in trades_15m]), -1.0, 1.0)
            cvd_15m = float(np.sum(qtys_15 * signs_15))

        # --- OBI rolling mean 5min ---
        obi_mean_5m = 0.0
        cutoff_5m = now - 300
        obi_vals_5m = [v for t, v in self.obi_history if t >= cutoff_5m]
        if obi_vals_5m:
            obi_mean_5m = float(np.mean(obi_vals_5m))

        # --- Volume acceleration: current 1min vol vs 5min avg per minute ---
        vol_1m = float(np.sum([t[2] for t in trades_1m])) if trades_1m else 0.0
        avg_vol_per_min_5m = vol_5m / 5.0 if vol_5m > 0 else 1.0
        volume_accel = (vol_1m / avg_vol_per_min_5m) - 1.0 if avg_vol_per_min_5m > 0 else 0.0

        # =============================================
        # PRICE-DERIVED FEATURES
        # =============================================
        price_hist = list(self.price_history)

        def _price_at_offset(offset_secs: int) -> float:
            target = now - offset_secs
            best = None
            for ts, px in price_hist:
                if ts <= target:
                    best = px
            return best if best is not None else last_price

        price_1m_ago = _price_at_offset(60)
        price_5m_ago = _price_at_offset(300)
        price_15m_ago = _price_at_offset(900)

        # --- Returns (in basis points) ---
        return_1m = ((last_price - price_1m_ago) / price_1m_ago * 10000) if price_1m_ago > 0 else 0.0
        return_5m = ((last_price - price_5m_ago) / price_5m_ago * 10000) if price_5m_ago > 0 else 0.0
        return_15m = ((last_price - price_15m_ago) / price_15m_ago * 10000) if price_15m_ago > 0 else 0.0

        # --- Price acceleration (second derivative): return_1m change rate ---
        # Approximate: (return_1m now) vs (return from 2min ago to 1min ago)
        price_2m_ago = _price_at_offset(120)
        return_prev_1m = ((price_1m_ago - price_2m_ago) / price_2m_ago * 10000) if price_2m_ago > 0 else 0.0
        price_accel = return_1m - return_prev_1m

        # --- Distance from 15min high/low (in basis points) ---
        if price_hist:
            prices_15m = [px for ts, px in price_hist]
            high_15m = max(prices_15m)
            low_15m = min(prices_15m)
            dist_from_high_15m = ((last_price - high_15m) / high_15m * 10000) if high_15m > 0 else 0.0
            dist_from_low_15m = ((last_price - low_15m) / low_15m * 10000) if low_15m > 0 else 0.0
        else:
            dist_from_high_15m = 0.0
            dist_from_low_15m = 0.0

        return {
            "obi": obi,
            "cvd": cvd,
            "toxicity": toxicity,
            "volatility": volatility,
            "spread_bps": spread_bps,
            "vwap_distance": vwap_distance,
            "trade_intensity": trade_intensity,
            "obi_gradient": obi_gradient,
            "cvd_gradient": cvd_gradient,
            "weighted_obi": weighted_obi,
            "cvd_5m": cvd_5m,
            "cvd_15m": cvd_15m,
            "obi_mean_5m": obi_mean_5m,
            "volume_accel": volume_accel,
            "return_1m": return_1m,
            "return_5m": return_5m,
            "return_15m": return_15m,
            "price_accel": price_accel,
            "dist_from_high_15m": dist_from_high_15m,
            "dist_from_low_15m": dist_from_low_15m,
            "last_price": last_price,
        }
