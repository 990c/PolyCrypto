"""Probability calibrators for trading models.

The production artifact (``*_calibrator.pkl``) must expose the sklearn-style
``predict_proba(log_odds_2d) -> [[p_down, p_up]]`` interface that
``PredictorV2._calibrate`` and the trainers already use, so isotonic calibration
is wrapped to be a drop-in replacement for the old Platt LogisticRegression.
"""
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicLogOddsCalibrator:
    """Isotonic calibration behind a Platt-compatible predict_proba interface.

    Fit on raw model probabilities; queried with log-odds (what the existing
    predictor pipeline passes). Output probabilities are clipped to
    [y_min, y_max] so sparse isotonic steps at the tails can't produce
    extreme probabilities that would blow up Kelly sizing.
    """

    def __init__(self, y_min: float = 0.10, y_max: float = 0.90):
        self.y_min = y_min
        self.y_max = y_max
        self._iso = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip")

    def fit(self, raw_probs: Sequence[float], labels: Sequence[int]) -> "IsotonicLogOddsCalibrator":
        self._iso.fit(np.asarray(raw_probs, dtype=float), np.asarray(labels, dtype=float))
        return self

    def predict_proba(self, log_odds: Sequence[Sequence[float]]) -> np.ndarray:
        lo = np.asarray(log_odds, dtype=float).reshape(-1)
        raw = 1.0 / (1.0 + np.exp(-lo))
        p_up = np.clip(self._iso.predict(raw), self.y_min, self.y_max)
        return np.column_stack([1.0 - p_up, p_up])
