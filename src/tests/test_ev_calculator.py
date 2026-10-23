"""Tests for the EVCalculator."""
import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.ev_calculator import EVCalculator


class TestEVCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = EVCalculator(kelly_fraction=0.25, min_ev_threshold=0.01)

    def test_hold_on_empty_book(self):
        result = self.calc.evaluate_ev(0.5, {}, {}, 0.0, 1000)
        self.assertEqual(result["action"], "HOLD")

    def test_hold_on_fair_price(self):
        """50/50 probability at 0.52 ask should not trigger a trade (negative EV after cost)."""
        result = self.calc.evaluate_ev(
            prob_up=0.50,
            poly_bids={0.48: 100},
            poly_asks={0.52: 100},
            fee_rate=0.0,
            bankroll=1000
        )
        self.assertEqual(result["action"], "HOLD")

    def test_buy_up_on_strong_signal(self):
        """High prob_up should trigger BUY_UP."""
        result = self.calc.evaluate_ev(
            prob_up=0.75,
            poly_bids={0.48: 100},
            poly_asks={0.52: 100},
            fee_rate=0.0,
            bankroll=1000
        )
        self.assertEqual(result["action"], "BUY_UP")
        self.assertGreater(result["ev"], 0)
        self.assertGreater(result["bet_size_usd"], 0)

    def test_buy_down_on_bearish_signal(self):
        """Low prob_up (high prob_down) should trigger BUY_DOWN."""
        result = self.calc.evaluate_ev(
            prob_up=0.25,
            poly_bids={0.48: 100},
            poly_asks={0.52: 100},
            fee_rate=0.0,
            bankroll=1000
        )
        self.assertEqual(result["action"], "BUY_DOWN")
        self.assertGreater(result["ev"], 0)

    def test_fees_reduce_ev(self):
        """Higher fees should reduce EV and potentially prevent trades."""
        # Marginal signal
        no_fee = self.calc.evaluate_ev(0.60, {0.48: 100}, {0.52: 100}, 0.0, 1000)
        with_fee = self.calc.evaluate_ev(0.60, {0.48: 100}, {0.52: 100}, 0.05, 1000)

        # EV should be lower with fees
        self.assertGreaterEqual(no_fee["ev"], with_fee["ev"])

    def test_kelly_sizing_bounded(self):
        """Bet size should never exceed kelly_fraction of bankroll."""
        result = self.calc.evaluate_ev(
            prob_up=0.95,
            poly_bids={0.48: 100},
            poly_asks={0.52: 100},
            fee_rate=0.0,
            bankroll=1000
        )
        max_possible = 1000 * self.calc.kelly_fraction
        self.assertLessEqual(result["bet_size_usd"], max_possible + 0.01)

    def test_min_ev_threshold(self):
        """Marginal EV below threshold should be filtered."""
        calc = EVCalculator(kelly_fraction=0.25, min_ev_threshold=0.50)
        result = calc.evaluate_ev(0.60, {0.48: 100}, {0.52: 100}, 0.0, 1000)
        self.assertEqual(result["action"], "HOLD")

    def test_symmetric_ev(self):
        """prob_up=0.7 buying UP should have same EV as prob_up=0.3 buying DOWN on symmetric book."""
        ev_up = self.calc.evaluate_ev(0.70, {0.48: 100}, {0.52: 100}, 0.0, 1000)
        ev_down = self.calc.evaluate_ev(0.30, {0.48: 100}, {0.52: 100}, 0.0, 1000)
        self.assertAlmostEqual(ev_up["ev"], ev_down["ev"], places=4)

    def test_bet_size_zero_for_hold(self):
        result = self.calc.evaluate_ev(0.50, {0.48: 100}, {0.52: 100}, 0.0, 1000)
        self.assertEqual(result["bet_size_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
