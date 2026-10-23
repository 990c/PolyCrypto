import csv
import os
import time
from loguru import logger
from typing import Optional

class TradeLogger:
    """
    Persistent CSV trade journal for measuring win rate, PnL, and Sharpe ratio.
    Every trade entry and every settlement is recorded with full context.
    """
    def __init__(self, log_path: str = "trades.csv", starting_bankroll: float = 1000.0):
        self.log_path = log_path
        self.starting_bankroll = starting_bankroll
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.cumulative_pnl = 0.0
        
        is_new = not os.path.exists(self.log_path)
        if is_new:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id",
                    "timestamp",
                    "market_id",
                    "action",         # BUY_UP or BUY_DOWN
                    "entry_price",
                    "size_usd",
                    "tokens",
                    "prob_up",
                    "ev",
                    "kelly_fraction",
                    "outcome",        # WIN, LOSS, or PENDING
                    "pnl",
                    "balance_after",
                    "cumulative_pnl",
                    "settlement_source",  # "resolution" (Polymarket) or "heuristic" (BTC fallback)
                ])
                
    def log_trade(self, market_id: str, action: str, price: float, size_usd: float,
                  prob_up: float, ev: float, balance_after: float) -> int:
        """Log an executed trade. Returns the trade_id for later settlement linking."""
        self.trade_count += 1
        tokens = size_usd / price if price > 0 else 0

        row = [
            self.trade_count,
            time.time(),
            market_id,
            action,
            f"{price:.4f}",
            f"{size_usd:.2f}",
            f"{tokens:.4f}",
            f"{prob_up:.4f}",
            f"{ev:.6f}",
            "",  # kelly_fraction filled by caller if desired
            "PENDING",
            "",
            f"{balance_after:.2f}",
            "",
            "",  # settlement_source only applies to SETTLE rows
        ]
        
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        logger.info(f"Trade #{self.trade_count} logged: {action} @ {price:.4f} for ${size_usd:.2f}")
        return self.trade_count
        
    def log_settlement(self, market_id: str, winning_side: str,
                       positions: dict, balance_after: float,
                       source: str = ""):
        """
        Log the settlement of an epoch. Records one row per held position
        with outcome = WIN or LOSS and the realized PnL.
        source: "resolution" (Polymarket's actual outcome) or "heuristic"
        (BTC price fallback) — label quality depends on it.
        """
        for side in ["UP", "DOWN"]:
            tokens = positions.get(side, 0.0)
            if tokens <= 0:
                continue
                
            is_win = (side == winning_side)
            # Winning tokens resolve to $1 each, losing tokens to $0
            pnl = tokens * 1.0 if is_win else 0.0
            # The cost was already deducted at entry, so net PnL relative to cost:
            # We don't have the original cost per token here, so we log gross proceeds
            outcome = "WIN" if is_win else "LOSS"
            
            if is_win:
                self.wins += 1
            else:
                self.losses += 1
                
            self.cumulative_pnl = balance_after - self.starting_bankroll
            
            row = [
                "",  # no new trade_id for settlement rows
                time.time(),
                market_id,
                f"SETTLE_{side}",
                "",
                "",
                f"{tokens:.4f}",
                "",
                "",
                "",
                outcome,
                f"{pnl:.2f}",
                f"{balance_after:.2f}",
                f"{self.cumulative_pnl:.2f}",
                source,
            ]
            
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        logger.info(f"Settlement logged for {market_id}: Winner={winning_side}")

    def get_stats(self) -> dict:
        total = self.wins + self.losses
        win_rate = (self.wins / total * 100) if total > 0 else 0.0
        return {
            "total_trades": self.trade_count,
            "settlements": total,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": win_rate,
            "cumulative_pnl": self.cumulative_pnl,
        }
