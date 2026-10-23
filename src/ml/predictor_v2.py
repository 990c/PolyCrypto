"""
V2 Predictor — Epoch-Aligned Hybrid Inference
==============================================
Mirrors the training pipeline in label_dataset_v3.py:
  - Accumulates per-tick features within the current 300s epoch
  - On epoch transition, finalizes an aggregate and appends to a 5-deep ring buffer
  - Waits until 90s into the current epoch, then builds the hybrid feature row
    (5 prior epochs + first-90s current-epoch features) and predicts
  - Caches the prediction for the remainder of the epoch
  - Returns 0.5 during warmup (~25 min from cold start), during the first 90s of
    each epoch, and in DATA_COLLECTION mode
"""
import os
import csv
import time
import pickle
import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from ml.features import FeatureEngine
from ml.label_dataset_v2 import build_feature_row
from ml.label_dataset_v3 import (
    CURR_WINDOW_SECONDS,
    MODEL_FEATURE_NAMES,
    build_curr_features,
)

EPOCH_SECONDS = 300
MIN_ROWS_PER_EPOCH = 10
WARMUP_EPOCHS = 5


class PredictorV2:
    RAW_FEATURE_COLS = FeatureEngine.FEATURE_NAMES

    def __init__(self, mode: str, model_path: str, dataset_path: str,
                 epoch_seconds: int = EPOCH_SECONDS):
        self.mode = mode
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.epoch_seconds = epoch_seconds
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.model = None
        self.calibrator = None
        self._model_loaded = False

        self._state_lock = threading.Lock()
        self._current_epoch_start: int = -1
        self._current_epoch_rows: list = []
        self._completed_epochs: deque = deque(maxlen=WARMUP_EPOCHS)
        self._cached_prediction: float = 0.5
        self._prediction_epoch_start: int = -1

        if self.mode == "DATA_COLLECTION":
            self._csv_lock = threading.Lock()
            is_new = not os.path.exists(self.dataset_path)
            with open(self.dataset_path, "a", newline="") as f:
                if is_new:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp"] + self.RAW_FEATURE_COLS +
                                    ["last_price", "target_resolution"])

    async def evaluate(self, features: dict) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, features)

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        if self.mode in ["LIVE", "PAPER"] and os.path.exists(self.model_path):
            logger.info(f"Loading V2 CatBoost model from {self.model_path}...")
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            logger.info("V2 model loaded.")

            calibrator_path = self.model_path.replace(".cbm", "_calibrator.pkl")
            if os.path.exists(calibrator_path):
                with open(calibrator_path, "rb") as f:
                    self.calibrator = pickle.load(f)
                logger.info(f"V2 calibrator loaded from {calibrator_path}")
            else:
                logger.warning("No V2 calibrator found — using raw probabilities.")
        elif self.mode in ["LIVE", "PAPER"]:
            logger.warning(f"V2 model not found at {self.model_path}. Returning 0.5.")

    def _predict_sync(self, features: dict) -> float:
        if self.mode == "DATA_COLLECTION":
            self._save_raw(features)
            return 0.5

        self._load_model()

        ts = time.time()
        epoch_start = int(ts) - (int(ts) % self.epoch_seconds)

        with self._state_lock:
            # Detect epoch transition: finalize previous epoch into the buffer
            if self._current_epoch_start == -1:
                self._current_epoch_start = epoch_start
            elif epoch_start != self._current_epoch_start:
                if self._current_epoch_rows:
                    agg = self._aggregate(self._current_epoch_rows,
                                          self._current_epoch_start)
                    if agg is not None:
                        # Training data uses 5 CONTIGUOUS prior epochs; a gap
                        # (WS outage) makes the buffered context stale — reset.
                        if (self._completed_epochs and
                                agg["epoch_start"] - self._completed_epochs[-1]["epoch_start"]
                                != self.epoch_seconds):
                            logger.warning("V2: epoch gap detected — resetting context buffer.")
                            self._completed_epochs.clear()
                        self._completed_epochs.append(agg)
                        logger.debug(f"V2: closed epoch {self._current_epoch_start} "
                                     f"with {len(self._current_epoch_rows)} ticks "
                                     f"(buffer={len(self._completed_epochs)}/{WARMUP_EPOCHS})")
                self._current_epoch_start = epoch_start
                self._current_epoch_rows = []
                self._prediction_epoch_start = -1  # invalidate cache

            # Append this tick to the current epoch
            row = {col: features.get(col, 0) for col in self.RAW_FEATURE_COLS}
            row["last_price"] = features.get("last_price", 0)
            row["timestamp"] = ts
            self._current_epoch_rows.append(row)

            # Return cached prediction if we already computed for this epoch
            if self._prediction_epoch_start == self._current_epoch_start:
                return self._cached_prediction

            # Warmup or model unavailable: no signal (cached for the whole epoch)
            if len(self._completed_epochs) < WARMUP_EPOCHS or self.model is None:
                self._cached_prediction = 0.5
                self._prediction_epoch_start = self._current_epoch_start
                return 0.5

            # Hybrid model needs the first 90s of the current epoch.
            # Do NOT cache — we predict as soon as the window is complete.
            if ts < self._current_epoch_start + CURR_WINDOW_SECONDS:
                return 0.5

            # Build hybrid feature row (5 prior epochs + first-90s current) and predict
            row = build_feature_row(list(self._completed_epochs))
            rows_90s = [r for r in self._current_epoch_rows
                        if r["timestamp"] < self._current_epoch_start + CURR_WINDOW_SECONDS]
            start_price = (self._current_epoch_rows[0]["last_price"]
                           if self._current_epoch_rows else 0.0)
            row.update(build_curr_features(rows_90s, start_price))
            x = [row.get(k, 0.0) for k in MODEL_FEATURE_NAMES]
            raw = float(self.model.predict_proba([x])[0][1])
            calibrated = self._calibrate(raw)

            self._cached_prediction = calibrated
            self._prediction_epoch_start = self._current_epoch_start
            logger.info(f"V2 prediction for epoch {self._current_epoch_start} "
                        f"(+{ts - self._current_epoch_start:.0f}s): "
                        f"raw={raw:.4f} cal={calibrated:.4f}")
            return calibrated

    def _aggregate(self, rows: list, epoch_start: int) -> dict | None:
        """Mirror label_dataset_v2.aggregate_epoch."""
        n = len(rows)
        if n < MIN_ROWS_PER_EPOCH:
            return None

        def col(c: str) -> np.ndarray:
            return np.array([r[c] for r in rows])

        q1_end = n // 4
        q4_start = 3 * n // 4

        def trend(c: str) -> float:
            v = col(c)
            return float(np.mean(v[q4_start:])) - float(np.mean(v[:q1_end]))

        start_price = rows[0]["last_price"]
        end_price = rows[-1]["last_price"]
        return_epoch = (((end_price - start_price) / start_price * 10000)
                        if start_price > 0 else 0.0)

        return {
            "obi_mean": float(np.mean(col("obi"))),
            "cvd_mean": float(np.mean(col("cvd"))),
            "toxicity_mean": float(np.mean(col("toxicity"))),
            "volatility_mean": float(np.mean(col("volatility"))),
            "spread_bps_mean": float(np.mean(col("spread_bps"))),
            "trade_intensity_mean": float(np.mean(col("trade_intensity"))),
            "weighted_obi_mean": float(np.mean(col("weighted_obi"))),
            "cvd_5m_mean": float(np.mean(col("cvd_5m"))),
            "cvd_15m_mean": float(np.mean(col("cvd_15m"))),
            "obi_mean_5m_mean": float(np.mean(col("obi_mean_5m"))),
            "volume_accel_mean": float(np.mean(col("volume_accel"))),
            "obi_std": float(np.std(col("obi"))),
            "cvd_std": float(np.std(col("cvd"))),
            "spread_bps_std": float(np.std(col("spread_bps"))),
            "obi_trend": trend("obi"),
            "cvd_trend": trend("cvd"),
            "volatility_trend": trend("volatility"),
            "return_epoch": return_epoch,
            "return_1m_mean": float(np.mean(col("return_1m"))),
            "price_accel_mean": float(np.mean(col("price_accel"))),
            "n_samples": float(n),
            "epoch_start": epoch_start,
        }

    def _calibrate(self, raw_prob: float) -> float:
        if self.calibrator is not None:
            log_odds = np.log(raw_prob / (1 - raw_prob + 1e-10) + 1e-10)
            cal = float(self.calibrator.predict_proba([[log_odds]])[0][1])
            return max(0.01, min(0.99, cal))
        return max(0.01, min(0.99, raw_prob))

    def _save_raw(self, features: dict):
        with self._csv_lock:
            with open(self.dataset_path, "a", newline="") as f:
                writer = csv.writer(f)
                r = [time.time()]
                r += [features.get(c, 0) for c in self.RAW_FEATURE_COLS]
                r += [features.get("last_price", 0), ""]
                writer.writerow(r)

    def shutdown(self):
        self.executor.shutdown(wait=False)
