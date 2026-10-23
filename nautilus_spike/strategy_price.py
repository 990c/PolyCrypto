"""Price-only strategy: the Exp 8 multi-year model live.

Feature path: Binance trade ticks -> per-second last-price grid (forward-filled
at build time, matching the collected-data semantics that passed the Step 1
transfer test) -> exp8.price_features.assemble_row — the exact code that built
the training data. No order books, no FeatureEngine.

Inherits the order/dry-run path, Polymarket book handling, and alert plumbing
from TradingStrategy; overrides data ingestion and prediction.
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "exp8"))

from nautilus_trader.model.data import TradeTick

from strategy_trading import TradingStrategy, TradingStrategyConfig

from price_features import (  # noqa: E402  (exp8 path injected above)
    EPOCH_SECONDS,
    PRICE_FEATURE_NAMES,
    assemble_row,
    epoch_aggregate,
)

PREDICT_AT = 90
HISTORY_SECONDS = 5 * EPOCH_SECONDS  # 5 prior epochs
PRUNE_SECONDS = HISTORY_SECONDS + 600


class PriceStrategyConfig(TradingStrategyConfig, frozen=True):
    model_path: str = os.path.join(REPO_ROOT, "exp8/catboost_price_multiyear.cbm")


class PriceTradingStrategy(TradingStrategy):
    def __init__(self, config: PriceStrategyConfig):
        super().__init__(config)
        self._grid: dict = {}

    def on_start(self):
        import pickle

        from catboost import CatBoostClassifier

        from nautilus_trader.model.identifiers import InstrumentId
        from trading.ev_calculator import EVCalculator

        self.model = CatBoostClassifier()
        self.model.load_model(self.config.model_path)
        with open(self.config.model_path.replace(".cbm", "_calibrator.pkl"), "rb") as f:
            self.calibrator = pickle.load(f)
        self.ev_calculator = EVCalculator(
            min_model_confidence=self.config.min_model_confidence,
            model_version="v2",
        )
        self.log.info(f"PRICE-MODEL loaded ({len(self.model.feature_names_)} features) "
                      f"from {self.config.model_path}")

        self.subscribe_trade_ticks(InstrumentId.from_str(self.config.binance_instrument_id))
        self._book_subscribed = set()
        self._scan_and_subscribe_books()
        self._arm_predict_alert()
        mode = "DRY-RUN (no orders)" if self.config.dry_run else "LIVE ORDERS ENABLED"
        self.log.warning(f"NAUT-TRADE: {mode} | stake ${self.config.stake_usd:.0f}/trade "
                         f"| price-only model")

    # ---- data ingestion: trade ticks only ----

    def on_trade_tick(self, tick: TradeTick):
        self._grid[int(tick.ts_event // 1_000_000_000)] = float(tick.price)

    def on_order_book(self, book):  # Polymarket books are cache-managed; nothing to do
        pass

    # ---- prediction at t+90 (called by TradingStrategy.on_event) ----

    def _predict_and_decide(self):
        import numpy as np

        self._arm_predict_alert()
        self._scan_and_subscribe_books()

        now = time.time()
        es = int(now) - (int(now) % EPOCH_SECONDS)
        self._epoch_start = es  # used by instrument lookup + trade dedup

        # prune old grid entries
        cutoff = es - PRUNE_SECONDS
        for s in [s for s in self._grid if s < cutoff]:
            del self._grid[s]

        if not self._grid or min(self._grid) > es - HISTORY_SECONDS:
            have = 0 if not self._grid else max(0, (es - min(self._grid)) // EPOCH_SECONDS)
            self.log.info(f"NAUT-TRADE: price-warmup {have}/5 prior epochs — no signal")
            return None

        # forward-fill the window [es - HISTORY, es + PREDICT_AT)
        start = es - HISTORY_SECONDS
        seed_keys = [s for s in self._grid if s < start]
        last = self._grid[max(seed_keys)] if seed_keys else None
        ffill: dict = {}
        for s in range(start, es + PREDICT_AT):
            if s in self._grid:
                last = self._grid[s]
            if last is not None:
                ffill[s] = last

        priors = []
        for k in range(5, 0, -1):
            agg = epoch_aggregate(ffill, es - k * EPOCH_SECONDS)
            if agg is None:
                self.log.warning("NAUT-TRADE: prior epoch coverage too thin — skipping")
                return None
            priors.append(agg)
        first90 = [ffill[s] for s in range(es, es + PREDICT_AT) if s in ffill]

        row = assemble_row(priors, first90)
        if row is None:
            self.log.warning("NAUT-TRADE: first-90s window too thin — skipping")
            return None
        x = [row[k] for k in PRICE_FEATURE_NAMES]
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
            f"NAUT-TRADE prediction epoch={es} (+{int(now) - es}s): "
            f"raw={raw:.4f} cal={cal:.4f} | decision: {decision['action']} "
            f"ev={decision.get('ev', 0):.4f}"
        )
        return decision
