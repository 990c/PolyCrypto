"""Phase 4: exec-enabled strategy — SignalStrategy plus order submission.

Safety layers (all independent):
  - ``dry_run=True`` by default: full decision path runs, orders are logged
    and NOT submitted. Flip only for the funded canary.
  - fixed canary stake (no Kelly compounding until live parity is proven)
  - Nautilus RiskEngine ``max_notional_per_order`` caps anything that slips through
  - one trade per epoch
"""
import time

from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Price, Quantity

from strategy_signal import EPOCH_SECONDS, SignalStrategy, SignalStrategyConfig


class TradingStrategyConfig(SignalStrategyConfig, frozen=True):
    dry_run: bool = True
    stake_usd: float = 10.0
    min_ev: float = 0.03


class TradingStrategy(SignalStrategy):
    def __init__(self, config: TradingStrategyConfig):
        super().__init__(config)
        self._traded_epoch = -1

    def on_start(self):
        super().on_start()
        mode = "DRY-RUN (no orders)" if self.config.dry_run else "LIVE ORDERS ENABLED"
        self.log.warning(f"NAUT-TRADE: {mode} | stake ${self.config.stake_usd:.0f}/trade")

    # SignalStrategy.on_event computes + logs the decision; we extend it to act.
    def on_event(self, event):
        if not str(getattr(event, "name", "")).startswith("predict-"):
            return
        decision = self._predict_and_decide()
        if decision is None:
            return
        action, ev, prob = decision["action"], decision["ev"], decision["prob_up"]
        if action not in ("BUY_UP", "BUY_DOWN") or ev < self.config.min_ev:
            return
        if self._traded_epoch == self._epoch_start:
            return

        outcome = "UP" if action == "BUY_UP" else "DOWN"
        inst = self._current_epoch_instrument(outcome)
        if inst is None:
            self.log.warning(f"NAUT-TRADE: no {outcome} instrument for epoch {self._epoch_start}")
            return
        book = self.cache.order_book(inst.id)
        if book is None or not book.asks():
            self.log.warning("NAUT-TRADE: no book for target instrument")
            return
        ask = self._levels(book.asks(), 1)
        price = ask[0][0]
        if not (0.05 <= price <= 0.95):
            self.log.info(f"NAUT-TRADE: skip — ask {price:.3f} outside sane range")
            return
        qty = round(self.config.stake_usd / price, 2)

        if self.config.dry_run:
            self.log.warning(
                f"NAUT-TRADE DRY-RUN: would submit {action} {qty} @ {price:.3f} "
                f"(${self.config.stake_usd:.2f}, prob_up={prob:.3f}, ev={ev:.3f}) on {inst.id}"
            )
            self._traded_epoch = self._epoch_start
            return

        order = self.order_factory.limit(
            instrument_id=inst.id,
            order_side=OrderSide.BUY,
            price=Price(price, precision=inst.price_precision),
            quantity=Quantity(qty, precision=inst.size_precision),
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)
        self._traded_epoch = self._epoch_start
        self.log.info(f"NAUT-TRADE: submitted {action} {qty} @ {price:.3f} on {inst.id}")

    def _current_epoch_instrument(self, outcome: str):
        from nautilus_trader.adapters.polymarket.common.constants import POLYMARKET_VENUE

        target_expiry_ns = (self._epoch_start + EPOCH_SECONDS) * 1_000_000_000
        for inst in self.cache.instruments(venue=POLYMARKET_VENUE):
            if (getattr(inst, "outcome", "") or "").upper() == outcome \
                    and inst.expiration_ns == target_expiry_ns:
                return inst
        return None

    def _predict_and_decide(self):
        """Runs the parent's prediction path and returns the decision dict."""
        self._arm_predict_alert()
        if len(self._completed) < 5 or not self._epoch_rows:
            self.log.info(f"NAUT-TRADE: warmup {len(self._completed)}/5 — no signal")
            return None

        import numpy as np

        from ml.label_dataset_v2 import build_feature_row
        from ml.label_dataset_v3 import CURR_WINDOW_SECONDS, MODEL_FEATURE_NAMES, build_curr_features

        row = build_feature_row(list(self._completed))
        rows_90s = [r for r in self._epoch_rows
                    if r["timestamp"] < self._epoch_start + CURR_WINDOW_SECONDS]
        start_price = self._epoch_rows[0].get("last_price", 0.0)
        row.update(build_curr_features(rows_90s, start_price))
        x = [row.get(k, 0.0) for k in MODEL_FEATURE_NAMES]
        raw = float(self.model.predict_proba([x])[0][1])
        log_odds = float(np.log(raw / (1 - raw + 1e-10) + 1e-10))
        cal = float(self.calibrator.predict_proba([[log_odds]])[0][1])

        bids, asks = self._poly_up_book()
        decision = self.ev_calculator.evaluate_ev(
            prob_up=cal, poly_bids=bids, poly_asks=asks, fee_rate=0.0,
            bankroll=self.config.stake_usd * 10,
        ) if (bids or asks) else {"action": "NO-BOOK", "ev": 0.0}
        decision["prob_up"] = cal

        self.log.info(
            f"NAUT-TRADE prediction epoch={self._epoch_start} "
            f"(+{int(time.time()) - self._epoch_start}s): raw={raw:.4f} cal={cal:.4f} "
            f"| decision: {decision['action']} ev={decision.get('ev', 0):.4f}"
        )
        return decision
