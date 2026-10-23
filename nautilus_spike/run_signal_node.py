"""Phase 2 runner: log-only signal node (Binance.US data + Polymarket data, NO execution)."""
import sys

sys.path.insert(0, ".")

from nautilus_trader.adapters.binance import BINANCE
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import PolymarketDataClientConfig
from nautilus_trader.adapters.polymarket import PolymarketLiveDataClientFactory
from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from strategy_signal import SignalStrategy, SignalStrategyConfig

config_node = TradingNodeConfig(
    trader_id=TraderId("NAUTSIG-001"),
    logging=LoggingConfig(log_level="INFO"),
    exec_engine=LiveExecEngineConfig(reconciliation=False),
    data_clients={
        BINANCE: BinanceDataClientConfig(
            account_type=BinanceAccountType.SPOT,
            us=True,  # binance.us — same feed the bot trained/collects on
            instrument_provider=InstrumentProviderConfig(
                load_ids=frozenset(["BTCUSDT.BINANCE"]),
            ),
        ),
        POLYMARKET: PolymarketDataClientConfig(
            instrument_config=PolymarketInstrumentProviderConfig(
                event_slug_builder="slug_builder_5m:build_btc_updown_5m_slugs",
            ),
            compute_effective_deltas=True,
        ),
    },
    timeout_connection=30.0,
    timeout_disconnection=10.0,
    timeout_post_stop=2.0,
)

node = TradingNode(config=config_node)
node.trader.add_strategy(SignalStrategy(SignalStrategyConfig()))
node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
node.build()

if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
