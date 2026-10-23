import os
import csv
import time
import asyncio
import pickle
import threading
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from ml.features import FeatureEngine


class Predictor:
    FEATURE_COLUMNS = FeatureEngine.FEATURE_NAMES

    def __init__(self, mode: str, model_path: str, dataset_path: str):
        self.mode = mode
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.model = None
        self.calibrator = None
        self._model_loaded = False
        self._csv_lock = threading.Lock()

        if self.mode == "DATA_COLLECTION":
            is_new = not os.path.exists(self.dataset_path)
            with open(self.dataset_path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["timestamp"] + self.FEATURE_COLUMNS + ["last_price", "target_resolution"])

    async def evaluate(self, features: dict) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, features)

    def _load_model(self):
        """Lazy-load CatBoost model and isotonic calibrator on first prediction."""
        if self._model_loaded:
            return
        self._model_loaded = True
        if self.mode in ["LIVE", "PAPER"] and os.path.exists(self.model_path):
            logger.info(f"Loading CatBoost model from {self.model_path}...")
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            logger.info("CatBoost model loaded.")

            calibrator_path = self.model_path.replace(".cbm", "_calibrator.pkl")
            if os.path.exists(calibrator_path):
                with open(calibrator_path, "rb") as f:
                    self.calibrator = pickle.load(f)
                logger.info(f"Platt scaling calibrator loaded from {calibrator_path}")
            else:
                logger.warning("No calibrator found — using raw model probabilities.")
        elif self.mode in ["LIVE", "PAPER"]:
            logger.warning(f"Model not found at {self.model_path}. Using dummy probabilities.")

    def _calibrate(self, raw_prob: float) -> float:
        """Apply Platt scaling to map raw model output to true probabilities.

        Uses a logistic regression fitted on a held-out calibration set.
        Platt scaling preserves the model's ranking (unlike isotonic which
        collapses low-signal models into flat staircases).

        Falls back to raw probability if no calibrator is loaded.
        """
        if self.calibrator is not None:
            # calibrator expects log-odds input, produces calibrated probability
            import numpy as np
            log_odds = np.log(raw_prob / (1 - raw_prob + 1e-10) + 1e-10)
            calibrated = float(self.calibrator.predict_proba([[log_odds]])[0][1])
            return max(0.01, min(0.99, calibrated))
        return max(0.01, min(0.99, raw_prob))

    def _predict_sync(self, features: dict) -> float:
        if self.mode == "DATA_COLLECTION":
            self._save_features(features)
            return 0.5

        self._load_model()
        if self.model:
            model_input = [features.get(col, 0) for col in self.FEATURE_COLUMNS]
            raw_prob = self.model.predict_proba([model_input])[0][1]
            return self._calibrate(float(raw_prob))

        # Dummy: OBI-based signal for paper testing
        dummy = 0.5 + (features.get("obi", 0) * 0.15) + (features.get("cvd_gradient", 0) * 0.05)
        return float(max(0.05, min(0.95, dummy)))

    def _save_features(self, features: dict):
        with self._csv_lock:
            with open(self.dataset_path, "a", newline="") as f:
                writer = csv.writer(f)
                row = [time.time()]
                row += [features.get(col, 0) for col in self.FEATURE_COLUMNS]
                row += [features.get("last_price", 0), ""]
                writer.writerow(row)

    def shutdown(self):
        self.executor.shutdown(wait=False)
