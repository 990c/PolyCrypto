"""Phase 2: PolyCrypto signal stack inside a Nautilus Strategy — LOG-ONLY.

Reuses the production PolyCrypto modules verbatim (FeatureEngine, epoch
aggregation, hybrid model + isotonic calibrator, EV gate) so signal parity with
the running bot is a code-identity property, not a reimplementation hope.

Pipeline:
  Binance.US BTCUSDT trades + 20-level book (2s interval, same cadence as bot)
    -> FeatureEngine per-tick rows -> epoch ring buffer (5 contiguous epochs)
    -> time alert at t+90s each epoch -> 46-feature hybrid model -> isotonic
    -> EV gate vs live Polymarket UpDown book -> logged decision (NO ORDERS).
"""
import os
import sys
import time
from collections import deque

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from nautilus_trader.adapters.polymarket.common.constants import POLYMARKET_VENUE
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide, BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

EPOCH_SECONDS = 300
PREDICT_AT_SECONDS = 90
WARMUP_EPOCHS = 5
BOOK_DEPTH = 20
BOOK_INTERVAL_MS = 2000
MIN_ROWS_PER_EPOCH = 10


class SignalStrategyConfig(StrategyConfig, frozen=True):
    binance_instrument_id: str = "BTCUSDT.BINANCE"
    model_path: str = os.path.join(REPO_ROOT, "catboost_model_v2.cbm")
    min_model_confidence: float = 0.08


class SignalStrategy(Strategy):
    def __init__(self, config: SignalStrategyConfig):
        super().__init__(config)
        self.feature_engine = None
        self.model = None
        self.calibrator = None
        self.ev_calculator = None
        self._epoch_start = -1
        self._epoch_rows: list = []
        self._completed: deque = deque(maxlen=WARMUP_EPOCHS)

    def on_start(self):
        import pickle

        from catboost import CatBoostClassifier

        from ml.features import FeatureEngine
        from trading.ev_calculator import EVCalculator

        self.model = CatBoostClassifier()
        self.model.load_model(self.config.model_path)
        with open(self.config.model_path.replace(".cbm", "_calibrator.pkl"), "rb") as f:
            self.calibrator = pickle.load(f)
        self.feature_engine = FeatureEngine(window_seconds=60)
        self.ev_calculator = EVCalculator(
            min_model_confidence=self.config.min_model_confidence,
            model_version="v2",
        )
        self.log.info(f"NAUT-SIGNAL: model loaded ({len(self.model.feature_names_)} features)")

        bin_id = InstrumentId.from_str(self.config.binance_instrument_id)
        self.subscribe_trade_ticks(bin_id)
        self.subscribe_order_book_at_interval(
            bin_id, book_type=BookType.L2_MBP, depth=BOOK_DEPTH, interval_ms=BOOK_INTERVAL_MS,
        )

        # Polymarket books for the EV side. Markets roll every 5 minutes, so we
        # must subscribe to instruments as the provider refresh loads them —
        # startup-only subscriptions expire within ~20 minutes (dry-run finding).
        self._book_subscribed: set = set()
        self._scan_and_subscribe_books()

        # Predict at t+90s of every epoch (one alert per epoch, re-armed on fire)
        self._arm_predict_alert()

    def _arm_predict_alert(self):
        now = time.time()
        epoch_start = int(now) - (int(now) % EPOCH_SECONDS)
        next_fire = epoch_start + PREDICT_AT_SECONDS
        if now >= next_fire - 1:
            next_fire += EPOCH_SECONDS
        self.clock.set_time_alert(
            f"predict-{next_fire}",
            alert_time=pd.Timestamp(int(next_fire), unit="s", tz="UTC"),
        )
        self._scan_and_subscribe_books()
        self.log.info(f"NAUT-SIGNAL: next prediction at {pd.Timestamp(int(next_fire), unit='s', tz='UTC')}")

    def _scan_and_subscribe_books(self) -> None:
        """Subscribe to books of every Polymarket instrument currently in the
        cache. Called at start and on every predict alert, so instruments loaded
        by the rolling provider refresh are picked up well before they become
        the active market. (The adapter has no venue-wide instrument stream.)"""
        for inst in self.cache.instruments(venue=POLYMARKET_VENUE):
            self._subscribe_poly_book(inst)

    def _subscribe_poly_book(self, instrument) -> None:
        if instrument.id in self._book_subscribed:
            return
        self._book_subscribed.add(instrument.id)
        self.subscribe_order_book_deltas(instrument.id)

    # ---------------- Binance feeds ----------------

    def on_trade_tick(self, tick: TradeTick):
        self.feature_engine.add_trade(
            price=float(tick.price),
            qty=float(tick.size),
            is_buyer_maker=(tick.aggressor_side == AggressorSide.SELLER),
        )

    @staticmethod
    def _levels(levels, depth):
        out = []
        for lv in levels[:depth]:
            size = lv.size() if callable(lv.size) else lv.size
            price = lv.price() if callable(lv.price) else lv.price
            out.append([float(price), float(size)])
        return out

    def on_order_book(self, book):
        bids = self._levels(book.bids(), BOOK_DEPTH)
        asks = self._levels(book.asks(), BOOK_DEPTH)
        if not bids or not asks:
            return
        features = self.feature_engine.compute_features(bids, asks)

        ts = time.time()
        epoch_start = int(ts) - (int(ts) % EPOCH_SECONDS)
        if self._epoch_start == -1:
            self._epoch_start = epoch_start
        elif epoch_start != self._epoch_start:
            self._finalize_epoch()
            self._epoch_start = epoch_start
            self._epoch_rows = []

        row = dict(features)
        row["timestamp"] = ts
        self._epoch_rows.append(row)

    def _finalize_epoch(self):
        from ml.label_dataset_v2 import aggregate_epoch

        if len(self._epoch_rows) < MIN_ROWS_PER_EPOCH:
            self._completed.clear()
            self.log.warning("NAUT-SIGNAL: thin epoch — buffer reset")
            return
        agg = aggregate_epoch(self._epoch_rows, self._epoch_start, EPOCH_SECONDS)
        if agg is None:
            self._completed.clear()
            return
        if (self._completed
                and agg["epoch_start"] - self._completed[-1]["epoch_start"] != EPOCH_SECONDS):
            self.log.warning("NAUT-SIGNAL: epoch gap — buffer reset")
            self._completed.clear()
        self._completed.append(agg)
        self.log.info(f"NAUT-SIGNAL: epoch {self._epoch_start} closed "
                      f"({len(self._epoch_rows)} rows, buffer {len(self._completed)}/{WARMUP_EPOCHS})")

    # ---------------- prediction ----------------

    def on_event(self, event):
        if not str(getattr(event, "name", "")).startswith("predict-"):
            return
        self._arm_predict_alert()  # re-arm for the next epoch first
        if len(self._completed) < WARMUP_EPOCHS:
            self.log.info(f"NAUT-SIGNAL: warmup {len(self._completed)}/{WARMUP_EPOCHS} — no signal")
            return
        if not self._epoch_rows:
            return

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
            prob_up=cal, poly_bids=bids, poly_asks=asks, fee_rate=0.0, bankroll=1000.0,
        ) if bids or asks else {"action": "NO-BOOK", "ev": 0.0}

        self.log.info(
            f"NAUT-SIGNAL prediction epoch={self._epoch_start} "
            f"(+{int(time.time()) - self._epoch_start}s): raw={raw:.4f} cal={cal:.4f} "
            f"| would-be: {decision['action']} ev={decision.get('ev', 0):.4f} (LOG-ONLY)"
        )

    def _poly_up_book(self):
        """Top-of-book dicts for the CURRENT epoch's UP instrument, {price: size}."""
        target_expiry_ns = (self._epoch_start + EPOCH_SECONDS) * 1_000_000_000
        for inst in self.cache.instruments(venue=POLYMARKET_VENUE):
            if (getattr(inst, "outcome", "") or "").upper() == "UP" \
                    and inst.expiration_ns == target_expiry_ns:
                book = self.cache.order_book(inst.id)
                if book is None:
                    return {}, {}
                bids = {p: s for p, s in self._levels(book.bids(), 5)}
                asks = {p: s for p, s in self._levels(book.asks(), 5)}
                return bids, asks
        return {}, {}
