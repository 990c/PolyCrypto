"""Tests for the PaperEngine."""
import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trading.paper_engine import PaperEngine


class TestPaperEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PaperEngine(start_balance=1000.0)

    def test_initial_state(self):
        self.assertEqual(self.engine.balance, 1000.0)
        self.assertEqual(self.engine.positions["UP"], 0.0)
        self.assertEqual(self.engine.positions["DOWN"], 0.0)

    def test_buy_up(self):
        success = self.engine.execute_order("BUY_UP", 0.52, 52.0)
        self.assertTrue(success)
        self.assertAlmostEqual(self.engine.balance, 948.0, places=2)
        self.assertAlmostEqual(self.engine.positions["UP"], 100.0, places=2)

    def test_buy_down(self):
        success = self.engine.execute_order("BUY_DOWN", 0.48, 48.0)
        self.assertTrue(success)
        self.assertAlmostEqual(self.engine.balance, 952.0, places=2)
        self.assertAlmostEqual(self.engine.positions["DOWN"], 100.0, places=2)

    def test_reject_insufficient_balance(self):
        success = self.engine.execute_order("BUY_UP", 0.52, 5000.0)
        self.assertFalse(success)

    def test_reject_zero_price(self):
        success = self.engine.execute_order("BUY_UP", 0.0, 50.0)
        self.assertFalse(success)

    def test_settle_up_wins(self):
        self.engine.execute_order("BUY_UP", 0.52, 52.0)  # 100 tokens
        pnl = self.engine.settle_market("UP")
        # Payout: 100 tokens * $1 = $100. Cost was $52. PnL = $100 - $52 = $48
        self.assertAlmostEqual(pnl, 48.0, places=2)
        self.assertAlmostEqual(self.engine.balance, 1048.0, places=2)

    def test_settle_up_loses(self):
        self.engine.execute_order("BUY_UP", 0.52, 52.0)
        pnl = self.engine.settle_market("DOWN")
        # UP tokens worth $0, lost $52
        self.assertAlmostEqual(pnl, -52.0, places=2)
        self.assertAlmostEqual(self.engine.balance, 948.0, places=2)

    def test_settle_down_wins(self):
        self.engine.execute_order("BUY_DOWN", 0.48, 48.0)  # 100 tokens
        pnl = self.engine.settle_market("DOWN")
        # Payout: 100 * $1 = $100. Cost was $48. PnL = $100 - $48 = $52
        self.assertAlmostEqual(pnl, 52.0, places=2)
        self.assertAlmostEqual(self.engine.balance, 1052.0, places=2)

    def test_positions_reset_after_settlement(self):
        self.engine.execute_order("BUY_UP", 0.52, 52.0)
        self.engine.settle_market("UP")
        self.assertEqual(self.engine.positions["UP"], 0.0)
        self.assertEqual(self.engine.positions["DOWN"], 0.0)

    def test_summary_uses_start_balance(self):
        """Summary should reflect PnL relative to start_balance, not hardcoded 1000."""
        engine = PaperEngine(start_balance=500.0)
        engine.execute_order("BUY_UP", 0.50, 50.0)
        engine.settle_market("UP")
        summary = engine.summary()
        self.assertIn("550.00", summary)
        self.assertIn("50.00", summary)

    def test_multiple_bets_same_epoch(self):
        self.engine.execute_order("BUY_UP", 0.52, 26.0)  # 50 tokens
        self.engine.execute_order("BUY_UP", 0.52, 26.0)  # 50 more tokens
        self.assertAlmostEqual(self.engine.positions["UP"], 100.0, places=2)
        pnl = self.engine.settle_market("UP")
        self.assertAlmostEqual(pnl, 48.0, places=2)


if __name__ == "__main__":
    unittest.main()
