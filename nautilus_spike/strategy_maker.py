"""Maker variant: post resting bids on the model's favored side instead of
crossing the spread (Exp 10).

At t+90 (same signal, same gate as the taker strategy):
  quote = min(best_bid + tick, fair - MARGIN), capped below best_ask
  posted GTD, expiring at epoch_start + QUOTE_END (before resolution).

Adverse-selection guard: if BTC moves more than ADVERSE_BPS against the quote
after posting, cancel (in dry-run: log the would-cancel time so the report can
truncate the fill window honestly).

Dry-run logs are self-contained for reconstruction:
  MAKER would-quote epoch=... side=UP qty=.. @0.xx window=[t+90,t+270] fair=0.xx
  MAKER would-cancel epoch=... at t+XXXs (adverse move)
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "exp8"))

from nautilus_trader.adapters.polymarket.common.constants import POLYMARKET_VENUE
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce

from strategy_price import EPOCH_SECONDS, PriceStrategyConfig, PriceTradingStrategy

QUOTE_END = 270          # cancel/expire resting quotes 30s before epoch close
MARGIN = 0.03            # required model edge at our quote price
ADVERSE_BPS = 8.0        # cancel if BTC moves this much against us post-quote


class MakerStrategyConfig(PriceStrategyConfig, frozen=True):
    pass


class MakerTradingStrategy(PriceTradingStrategy):
    def __init__(self, config: MakerStrategyConfig):
        super().__init__(config)
        self._quote: dict | None = None  # active virtual/real quote this epoch

    def on_start(self):
        super().on_start()
        self.log.warning(f"NAUT-TRADE: MAKER mode | margin {MARGIN:.2f} | "
                         f"adverse-cancel {ADVERSE_BPS}bps | quote window t+90..t+{QUOTE_END}")

    # ---- adverse-move guard rides the Binance tick stream ----

    def on_trade_tick(self, tick: TradeTick):
        super().on_trade_tick(tick)
        q = self._quote
        if q is None:
            return
        now_s = tick.ts_event / 1_000_000_000
        if now_s >= q["epoch_start"] + QUOTE_END:
            self._quote = None
            return
        px = float(tick.price)
        move_bps = (px - q["btc_at_quote"]) / q["btc_at_quote"] * 1e4
        adverse = -move_bps if q["side"] == "UP" else move_bps
        if adverse > ADVERSE_BPS:
            t_rel = int(now_s - q["epoch_start"])
            if self.config.dry_run:
                self.log.warning(f"MAKER would-cancel epoch={q['epoch_start']} "
                                 f"at t+{t_rel}s (adverse {adverse:.1f}bps)")
            elif q.get("order") is not None:
                self.cancel_order(q["order"])
                self.log.info(f"MAKER cancelled quote (adverse {adverse:.1f}bps)")
            self._quote = None

    # ---- quoting decision at t+90 (TradingStrategy.on_event drives this) ----

    def on_event(self, event):
        if not str(getattr(event, "name", "")).startswith("predict-"):
            return
        decision = self._predict_and_decide()
        if decision is None:
            return
        action = decision["action"]
        p = decision["prob_up"]
        if action not in ("BUY_UP", "BUY_DOWN"):
            return
        if self._traded_epoch == self._epoch_start:
            return

        side = "UP" if action == "BUY_UP" else "DOWN"
        fair = p if side == "UP" else 1.0 - p
        inst = self._current_epoch_instrument(side)
        if inst is None:
            self.log.warning(f"MAKER: no {side} instrument for epoch {self._epoch_start}")
            return
        book = self.cache.order_book(inst.id)
        if book is None or not book.bids() or not book.asks():
            self.log.info("MAKER: no two-sided book — skip")
            return
        tick_sz = 10 ** -inst.price_precision
        best_bid = self._levels(book.bids(), 1)[0][0]
        best_ask = self._levels(book.asks(), 1)[0][0]

        quote = min(best_bid + tick_sz, round(fair - MARGIN, inst.price_precision))
        quote = min(quote, best_ask - tick_sz)  # never cross
        quote = round(quote, inst.price_precision)
        if quote < 0.05 or quote > 0.95:
            self.log.info(f"MAKER: quote {quote:.2f} outside sane range — skip")
            return
        if fair - quote < MARGIN:
            self.log.info(f"MAKER: insufficient edge at quote "
                          f"(fair {fair:.2f}, quote {quote:.2f}) — skip")
            return
        qty = round(self.config.stake_usd / quote, 2)

        btc_now = 0.0
        if self._grid:
            btc_now = self._grid[max(self._grid)]
        self._quote = {
            "epoch_start": self._epoch_start, "side": side, "price": quote,
            "qty": qty, "btc_at_quote": btc_now, "order": None,
        }
        self._traded_epoch = self._epoch_start

        if self.config.dry_run:
            self.log.warning(
                f"MAKER would-quote epoch={self._epoch_start} side={side} "
                f"qty={qty} @{quote:.2f} window=[t+90,t+{QUOTE_END}] "
                f"fair={fair:.3f} bid={best_bid:.2f} ask={best_ask:.2f}"
            )
            return

        import pandas as pd

        from nautilus_trader.model.objects import Price, Quantity

        order = self.order_factory.limit(
            instrument_id=inst.id,
            order_side=OrderSide.BUY,
            price=Price(quote, precision=inst.price_precision),
            quantity=Quantity(qty, precision=inst.size_precision),
            time_in_force=TimeInForce.GTD,
            expire_time=pd.Timestamp(self._epoch_start + QUOTE_END, unit="s", tz="UTC"),
        )
        self._quote["order"] = order
        self.submit_order(order)
        self.log.info(f"MAKER quoted {side} {qty} @ {quote:.2f} GTD t+{QUOTE_END}")
