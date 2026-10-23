"""Tests for the FeatureEngine."""
import sys
import os
import time
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml.features import FeatureEngine


class TestFeatureEngine(unittest.TestCase):

    def setUp(self):
        self.engine = FeatureEngine(window_seconds=60)

    def test_empty_features(self):
        """Features with no trades should return safe defaults."""
        bids = [["100.0", "1.0"], ["99.0", "2.0"]]
        asks = [["101.0", "1.5"], ["102.0", "2.5"]]
        f = self.engine.compute_features(bids, asks)

        self.assertAlmostEqual(f["cvd"], 0.0)
        self.assertAlmostEqual(f["toxicity"], 0.5)
        self.assertAlmostEqual(f["volatility"], 0.0)
        self.assertEqual(f["last_price"], 0.0)

    def test_obi_calculation(self):
        """OBI should reflect bid/ask volume imbalance."""
        # Bids heavier than asks
        bids = [["100.0", "10.0"]]
        asks = [["101.0", "2.0"]]
        f = self.engine.compute_features(bids, asks)
        self.assertGreater(f["obi"], 0)

        # Asks heavier than bids
        bids = [["100.0", "2.0"]]
        asks = [["101.0", "10.0"]]
        f = self.engine.compute_features(bids, asks)
        self.assertLess(f["obi"], 0)

    def test_obi_range(self):
        """OBI should always be in [-1, 1]."""
        bids = [["100.0", "100.0"]]
        asks = [["101.0", "0.01"]]
        f = self.engine.compute_features(bids, asks)
        self.assertLessEqual(f["obi"], 1.0)
        self.assertGreaterEqual(f["obi"], -1.0)

    def test_cvd_positive_on_buys(self):
        """CVD should be positive when buyers dominate."""
        for _ in range(10):
            self.engine.add_trade(100.0, 1.0, is_buyer_maker=False)  # buy taker

        bids = [["99.0", "1.0"]]
        asks = [["101.0", "1.0"]]
        f = self.engine.compute_features(bids, asks)
        self.assertGreater(f["cvd"], 0)

    def test_cvd_negative_on_sells(self):
        """CVD should be negative when sellers dominate."""
        for _ in range(10):
            self.engine.add_trade(100.0, 1.0, is_buyer_maker=True)  # sell taker

        f = self.engine.compute_features([["99", "1"]], [["101", "1"]])
        self.assertLess(f["cvd"], 0)

    def test_toxicity_range(self):
        """Toxicity should be in [0, 1]."""
        self.engine.add_trade(100.0, 5.0, True)
        self.engine.add_trade(100.0, 3.0, False)
        f = self.engine.compute_features([["99", "1"]], [["101", "1"]])
        self.assertGreaterEqual(f["toxicity"], 0.0)
        self.assertLessEqual(f["toxicity"], 1.0)

    def test_toxicity_high_on_sells(self):
        """Toxicity should be high when mostly sell-taker volume."""
        for _ in range(20):
            self.engine.add_trade(100.0, 1.0, is_buyer_maker=True)
        self.engine.add_trade(100.0, 0.1, is_buyer_maker=False)

        f = self.engine.compute_features([["99", "1"]], [["101", "1"]])
        self.assertGreater(f["toxicity"], 0.8)

    def test_spread_bps(self):
        """Spread should be computed correctly in basis points."""
        bids = [["100.0", "1.0"]]
        asks = [["100.10", "1.0"]]
        f = self.engine.compute_features(bids, asks)
        expected_bps = (0.10 / 100.05) * 10000
        self.assertAlmostEqual(f["spread_bps"], expected_bps, places=1)

    def test_window_cleanup(self):
        """Trades outside the 60s cvd window must not contribute to cvd,
        and trades older than 15 minutes are evicted entirely."""
        engine = FeatureEngine()
        now = time.time()
        # Stale aggressive sell (outside 60s cvd window, inside 15m retention)
        engine.trades.append((now - 61, 100.0, 5.0, True))
        # Ancient trade (outside 15m retention) — must be evicted on next add
        engine.trades.appendleft((now - 901, 100.0, 9.0, True))
        # Fresh aggressive buy
        engine.add_trade(200.0, 1.0, False)

        self.assertTrue(all(t[0] >= now - 900 for t in engine.trades))
        f = engine.compute_features([["99", "1"]], [["101", "1"]])
        # Only the fresh buy is inside the 60s cvd window
        self.assertGreater(f["cvd"], 0)

    def test_all_feature_names_present(self):
        """All declared FEATURE_NAMES should appear in output."""
        self.engine.add_trade(100.0, 1.0, False)
        f = self.engine.compute_features([["99", "1"]], [["101", "1"]])
        for name in FeatureEngine.FEATURE_NAMES:
            self.assertIn(name, f, f"Missing feature: {name}")

    def test_feature_count_is_20(self):
        """The V1 model contract: exactly 20 features in FEATURE_NAMES."""
        self.assertEqual(len(FeatureEngine.FEATURE_NAMES), 20)

    def test_weighted_obi_range(self):
        """Weighted OBI should be in [-1, 1]."""
        bids = [["100.0", "100.0"], ["99.0", "50.0"]]
        asks = [["101.0", "0.01"], ["102.0", "0.01"]]
        f = self.engine.compute_features(bids, asks)
        self.assertLessEqual(f["weighted_obi"], 1.0)
        self.assertGreaterEqual(f["weighted_obi"], -1.0)

    def test_weighted_obi_positive_on_bid_heavy(self):
        """Weighted OBI should be positive when bids outweigh asks."""
        bids = [["100.0", "10.0"], ["99.0", "8.0"]]
        asks = [["101.0", "2.0"], ["102.0", "1.0"]]
        f = self.engine.compute_features(bids, asks)
        self.assertGreater(f["weighted_obi"], 0)

    def test_weighted_obi_emphasizes_top_levels(self):
        """
        Weighted OBI should be more influenced by level 1 than deeper levels.
        Scenario: level 1 bids heavy, but levels 2-5 asks heavy.
        Flat OBI goes negative, weighted OBI should be less negative (or positive).
        """
        bids = [["100.0", "10.0"], ["99.0", "1.0"], ["98.0", "1.0"]]
        asks = [["101.0", "2.0"], ["102.0", "8.0"], ["103.0", "8.0"]]
        f = self.engine.compute_features(bids, asks)
        # Flat OBI: bids=12, asks=18 -> negative
        self.assertLess(f["obi"], 0)
        # Weighted OBI: level 1 bid (10*1.0) vs level 1 ask (2*1.0) dominates
        # So weighted_obi should be more positive than flat obi
        self.assertGreater(f["weighted_obi"], f["obi"])

    def test_weighted_obi_correct_math(self):
        """Verify weighted OBI formula with known inputs."""
        # 2 bids, 2 asks, known volumes
        bids = [["100.0", "6.0"], ["99.0", "4.0"]]  # w_bid = 6/1 + 4/2 = 8.0
        asks = [["101.0", "3.0"], ["102.0", "2.0"]]  # w_ask = 3/1 + 2/2 = 4.0
        f = self.engine.compute_features(bids, asks)
        expected = (8.0 - 4.0) / (8.0 + 4.0)  # = 4/12 = 0.3333
        self.assertAlmostEqual(f["weighted_obi"], expected, places=4)

    def test_volatility_increases_with_price_variance(self):
        """Volatility should be higher with more price variation."""
        engine_stable = FeatureEngine(window_seconds=60)
        for _ in range(50):
            engine_stable.add_trade(100.0, 1.0, False)

        engine_volatile = FeatureEngine(window_seconds=60)
        for i in range(50):
            price = 100.0 + (10.0 if i % 2 == 0 else -10.0)
            engine_volatile.add_trade(price, 1.0, False)

        f_stable = engine_stable.compute_features([["99", "1"]], [["101", "1"]])
        f_volatile = engine_volatile.compute_features([["99", "1"]], [["101", "1"]])

        self.assertGreater(f_volatile["volatility"], f_stable["volatility"])


if __name__ == "__main__":
    unittest.main()
