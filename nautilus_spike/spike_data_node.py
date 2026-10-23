"""Spike: live Polymarket data node for BTC 5-minute UpDown markets.

Verifies, against production Polymarket:
  1. instrument discovery via our 5m event_slug_builder
  2. order book deltas / quotes flowing for the current epoch's market
  3. rotation — the next epoch's (pre-loaded) market starts ticking after the boundary

Run for ~4.5 minutes (crosses at least one epoch boundary), then Ctrl-C / SIGTERM.
"""
import time
from collections import defaultdict

from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import PolymarketDataClientConfig
from nautilus_trader.adapters.polymarket import PolymarketLiveDataClientFactory
from nautilus_trader.adapters.polymarket.common.constants import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import OrderBookDeltas, QuoteTick
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.trading.strategy import Strategy


class BookWatcherConfig(StrategyConfig, frozen=True):
    pass


class BookWatcher(Strategy):
    """Subscribes to every loaded UpDown instrument and counts updates."""

    def __init__(self, config: BookWatcherConfig):
        super().__init__(config)
        self.delta_counts: dict = defaultdict(int)
        self.quote_counts: dict = defaultdict(int)

    def on_start(self):
        instruments = self.cache.instruments(venue=POLYMARKET_VENUE)
        self.log.info(f"SPIKE: {len(instruments)} instruments in cache")
        for inst in instruments:
            self.subscribe_order_book_deltas(inst.id)
            self.subscribe_quote_ticks(inst.id)
            self.log.info(f"SPIKE: subscribed {inst.id} (expires {inst.expiration_ns})")
        self.clock.set_timer("report", interval=__import__("pandas").Timedelta(seconds=30))

    def on_order_book_deltas(self, deltas: OrderBookDeltas):
        iid = str(deltas.instrument_id)
        if self.delta_counts[iid] == 0:
            self.log.info(f"SPIKE FIRST-DELTA {iid[:50]} at t+{int(time.time()) % 300}s")
        self.delta_counts[iid] += 1

    def on_quote_tick(self, tick: QuoteTick):
        self.quote_counts[str(tick.instrument_id)] += 1

    def on_event(self, event):
        now = int(time.time())
        epoch_pos = now % 300
        self.log.info(f"SPIKE REPORT t+{epoch_pos}s into epoch:")
        for iid, n in sorted(self.delta_counts.items()):
            self.log.info(f"SPIKE   {iid[:50]} deltas={n} quotes={self.quote_counts.get(iid, 0)}")


instrument_config = PolymarketInstrumentProviderConfig(
    event_slug_builder="slug_builder_5m:build_btc_updown_5m_slugs",
)

config_node = TradingNodeConfig(
    trader_id=TraderId("SPIKE-001"),
    logging=LoggingConfig(log_level="INFO"),
    exec_engine=LiveExecEngineConfig(reconciliation=False),
    data_clients={
        POLYMARKET: PolymarketDataClientConfig(
            instrument_config=instrument_config,
            compute_effective_deltas=True,
            subscribe_new_markets=False,
        ),
    },
    timeout_connection=30.0,
    timeout_disconnection=10.0,
    timeout_post_stop=2.0,
)

node = TradingNode(config=config_node)
node.trader.add_strategy(BookWatcher(BookWatcherConfig()))
node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
node.build()

if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
