"""Tests for the RiskManager."""
import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.rm = RiskManager(starting_bankroll=1000.0)

    def test_valid_trade(self):
        self.assertTrue(self.rm.validate_trade(50.0))

    def test_reject_zero_bet(self):
        self.assertFalse(self.rm.validate_trade(0.0))
        self.assertFalse(self.rm.validate_trade(-10.0))

    def test_exposure_limit(self):
        """Should reject when epoch exposure exceeds limit."""
        # Max exposure = 20% of 1000 = 200
        self.rm.register_position(180.0)  # active_exposure is now 180
        self.assertFalse(self.rm.validate_trade(30.0))  # 180 + 30 = 210 > 200

    def test_single_bet_cap(self):
        """Single bet should not exceed max_single_bet_pct."""
        # Max single bet = 10% of 1000 = 100
        self.assertFalse(self.rm.validate_trade(150.0))

    def test_drawdown_halt(self):
        """Should halt when daily drawdown limit is breached."""
        self.rm.current_bankroll = 750.0  # 25% drawdown > 20% limit
        self.assertFalse(self.rm.validate_trade(10.0))

    def test_clear_exposure(self):
        self.rm.register_position(100.0)
        self.rm.clear_exposure()
        self.assertEqual(self.rm.active_exposure, 0.0)

    def test_consecutive_loss_protection(self):
        """Should stop trading after max consecutive losses."""
        for _ in range(5):
            self.rm.register_result(False)
        self.assertFalse(self.rm.validate_trade(10.0))

    def test_win_resets_consecutive_losses(self):
        for _ in range(4):
            self.rm.register_result(False)
        self.rm.register_result(True)
        self.assertTrue(self.rm.validate_trade(10.0))

    def test_available_balance_check(self):
        """Cannot bet more than available (bankroll - exposure)."""
        self.rm.register_position(950.0)
        self.rm.active_exposure = 950.0
        self.assertFalse(self.rm.validate_trade(60.0))


if __name__ == "__main__":
    unittest.main()
