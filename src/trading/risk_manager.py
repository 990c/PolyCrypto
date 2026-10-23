import time
from loguru import logger


class RiskManager:
    def __init__(self, starting_bankroll: float, daily_drawdown_limit: float = 0.20,
                 max_epoch_exposure_pct: float = 0.20, max_single_bet_pct: float = 0.10):
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.daily_drawdown_limit = daily_drawdown_limit
        self.max_epoch_exposure_pct = max_epoch_exposure_pct
        self.max_single_bet_pct = max_single_bet_pct
        self.active_exposure = 0.0

        # Rolling window drawdown tracking (resets every N seconds)
        self.drawdown_window_secs = 7200  # 2 hours
        self.window_start_bankroll = starting_bankroll
        self.window_start_time = time.time()

        # Consecutive loss tracking for tilt protection
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.tilt_cooldown_epochs = 3  # skip N epochs after tilt triggers
        self._tilt_skipped = 0  # how many epochs skipped so far

        # Avoid spamming the same warning
        self._drawdown_warned = False
        self._tilt_warned = False

    def _check_window_rollover(self):
        now = time.time()
        if now - self.window_start_time >= self.drawdown_window_secs:
            logger.info(
                f"Drawdown window reset. Baseline ${self.window_start_bankroll:.2f} -> ${self.current_bankroll:.2f}"
            )
            self.window_start_bankroll = self.current_bankroll
            self.window_start_time = now
            self._drawdown_warned = False

    def validate_trade(self, bet_size_usd: float) -> bool:
        if bet_size_usd <= 0:
            return False

        self._check_window_rollover()

        # Rolling window drawdown circuit breaker
        if self.current_bankroll < (self.window_start_bankroll * (1.0 - self.daily_drawdown_limit)):
            if not self._drawdown_warned:
                logger.warning("Daily drawdown limit reached. Halting trades until next day.")
                self._drawdown_warned = True
            return False

        # Tilt protection: pause after N consecutive losses, resume after cooldown
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self._tilt_skipped < self.tilt_cooldown_epochs:
                if not self._tilt_warned:
                    logger.warning(
                        f"Tilt protection: {self.consecutive_losses} consecutive losses. "
                        f"Cooling down for {self.tilt_cooldown_epochs} epochs."
                    )
                    self._tilt_warned = True
                return False
            # Cooldown expired — reset and resume trading
            logger.info(f"Tilt cooldown complete after {self._tilt_skipped} epochs. Resuming.")
            self.consecutive_losses = 0
            self._tilt_skipped = 0
            self._tilt_warned = False

        # Per-epoch exposure cap
        max_exposure = self.current_bankroll * self.max_epoch_exposure_pct
        if (self.active_exposure + bet_size_usd) > max_exposure:
            return False

        # Single bet size cap
        max_bet = self.current_bankroll * self.max_single_bet_pct
        if bet_size_usd > max_bet:
            return False

        # Don't bet more than available
        available = self.current_bankroll - self.active_exposure
        if available < bet_size_usd:
            return False

        return True

    def register_position(self, cost: float):
        self.active_exposure += cost

    def register_result(self, is_win: bool):
        if is_win:
            self.consecutive_losses = 0
            self._tilt_warned = False
        else:
            self.consecutive_losses += 1

    def update_bankroll(self, new_bankroll: float):
        self.current_bankroll = new_bankroll

    def seed_bankroll(self, bankroll: float):
        """Re-baseline every bankroll reference. Used at LIVE startup once the
        real wallet balance is known, so drawdown/exposure caps track real money."""
        self.starting_bankroll = bankroll
        self.current_bankroll = bankroll
        self.window_start_bankroll = bankroll
        logger.info(f"Risk manager bankroll seeded to ${bankroll:.2f}")

    def tick_tilt_cooldown(self):
        """Call once per epoch. Advances the cooldown counter when in tilt state."""
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._tilt_skipped += 1

    def clear_exposure(self):
        self.active_exposure = 0.0
