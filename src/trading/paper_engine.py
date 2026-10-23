from loguru import logger


class PaperEngine:
    # Realistic price limits
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95
    MAX_BET_USD = None  # Set to a dollar amount to cap bet size (e.g. 100.0)

    def __init__(self, start_balance: float = 1000.0):
        self.start_balance = start_balance
        self.balance = start_balance
        self.positions = {"UP": 0.0, "DOWN": 0.0}
        self.total_cost = {"UP": 0.0, "DOWN": 0.0}

    def execute_order(self, action: str, price: float, size_usd: float,
                      available_liquidity_usd: float = None) -> bool:
        # Reject extreme prices — these don't have real liquidity
        if price < self.MIN_PRICE or price > self.MAX_PRICE:
            logger.debug(f"Paper: rejected {action} @ {price:.4f} — price outside {self.MIN_PRICE}-{self.MAX_PRICE}")
            return False

        # Cap bet size if configured
        if self.MAX_BET_USD is not None:
            size_usd = min(size_usd, self.MAX_BET_USD)

        # Cap to available liquidity if provided
        if available_liquidity_usd is not None and available_liquidity_usd > 0:
            size_usd = min(size_usd, available_liquidity_usd)

        if self.balance < size_usd or size_usd <= 0:
            return False

        tokens = size_usd / price
        if action == "BUY_UP":
            self.balance -= size_usd
            self.positions["UP"] += tokens
            self.total_cost["UP"] += size_usd
            return True
        elif action == "BUY_DOWN":
            self.balance -= size_usd
            self.positions["DOWN"] += tokens
            self.total_cost["DOWN"] += size_usd
            return True

        return False

    def settle_market(self, winning_side: str) -> float:
        """
        Winning tokens resolve to $1 each, losing tokens to $0.
        Returns the net PnL for this epoch.
        """
        payout = 0.0
        cost = 0.0

        if winning_side == "UP":
            payout = self.positions["UP"] * 1.0
            cost = self.total_cost["UP"] + self.total_cost["DOWN"]
        elif winning_side == "DOWN":
            payout = self.positions["DOWN"] * 1.0
            cost = self.total_cost["UP"] + self.total_cost["DOWN"]

        self.balance += payout
        epoch_pnl = payout - cost

        # Reset positions for the next 5-minute epoch
        self.positions = {"UP": 0.0, "DOWN": 0.0}
        self.total_cost = {"UP": 0.0, "DOWN": 0.0}

        return epoch_pnl

    def summary(self) -> str:
        pnl = self.balance - self.start_balance
        pct = (pnl / self.start_balance) * 100 if self.start_balance > 0 else 0
        return f"Balance: ${self.balance:.2f} | PnL: ${pnl:.2f} ({pct:+.1f}%)"
