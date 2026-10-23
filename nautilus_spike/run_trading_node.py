"""Phase 4 runner: full trading node (data + execution) with layered caps.

DRY-RUN by default — set POLY_LIVE_ORDERS=1 AND real POLYMARKET_* credentials
to actually submit orders. RiskEngine caps notional per order independently
of the strategy's own stake logic.

Env (live): POLYMARKET_PK, POLYMARKET_FUNDER, POLYMARKET_API_KEY/SECRET/PASSPHRASE
(or let the adapter derive API creds from the key), POLYMARKET_SIGNATURE_TYPE.
"""
import os
import sys

sys.path.insert(0, ".")

from nautilus_trader.adapters.binance import BINANCE
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import PolymarketDataClientConfig
from nautilus_trader.adapters.polymarket import PolymarketExecClientConfig
from nautilus_trader.adapters.polymarket import PolymarketLiveDataClientFactory
from nautilus_trader.adapters.polymarket import PolymarketLiveExecClientFactory
from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import LiveRiskEngineConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from strategy_maker import MakerStrategyConfig, MakerTradingStrategy
from strategy_price import PriceStrategyConfig, PriceTradingStrategy
from strategy_trading import TradingStrategy, TradingStrategyConfig

LIVE_ORDERS = os.environ.get("POLY_LIVE_ORDERS", "0") == "1"
MODEL_KIND = os.environ.get("NAUT_MODEL", "maker")  # "maker" (Exp 10) | "price" | "hybrid"
SIGNATURE_TYPE = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1"))
STAKE_USD = float(os.environ.get("NAUT_STAKE_USD", "10"))

updown_provider = PolymarketInstrumentProviderConfig(
    event_slug_builder="slug_builder_5m:build_btc_updown_5m_slugs",
)

config_node = TradingNodeConfig(
    trader_id=TraderId("NAUTCANARY-001"),
    logging=LoggingConfig(
        log_level="INFO",
        log_directory="logs",
        log_file_name="naut_canary",
        log_level_file="INFO",
        log_file_max_size=50_000_000,
        log_file_max_backup_count=5,
    ),
    risk_engine=LiveRiskEngineConfig(
        max_order_submit_rate="5/00:01:00",           # ≤5 orders/minute, hard stop
        max_notional_per_order={"USDC.e": 15},        # independent of strategy stake
    ),
    exec_engine=LiveExecEngineConfig(reconciliation=True),
    data_clients={
        BINANCE: BinanceDataClientConfig(
            account_type=BinanceAccountType.SPOT,
            us=True,
            instrument_provider=InstrumentProviderConfig(
                load_ids=frozenset(["BTCUSDT.BINANCE"]),
            ),
        ),
        POLYMARKET: PolymarketDataClientConfig(
            instrument_config=updown_provider,
            compute_effective_deltas=True,
            # Rolling 5m markets: keep loading upcoming epochs (Phase 2 gap fix)
            update_instruments_interval_mins=2,
            subscribe_new_markets=True,
            signature_type=SIGNATURE_TYPE,
        ),
    },
    exec_clients={
        POLYMARKET: PolymarketExecClientConfig(
            instrument_config=updown_provider,
            signature_type=SIGNATURE_TYPE,
        ),
    } if LIVE_ORDERS else {},
    timeout_connection=30.0,
    timeout_disconnection=10.0,
    timeout_post_stop=2.0,
)

node = TradingNode(config=config_node)
if MODEL_KIND == "maker":
    node.trader.add_strategy(MakerTradingStrategy(MakerStrategyConfig(
        dry_run=not LIVE_ORDERS,
        stake_usd=STAKE_USD,
    )))
elif MODEL_KIND == "price":
    node.trader.add_strategy(PriceTradingStrategy(PriceStrategyConfig(
        dry_run=not LIVE_ORDERS,
        stake_usd=STAKE_USD,
    )))
else:
    node.trader.add_strategy(TradingStrategy(TradingStrategyConfig(
        dry_run=not LIVE_ORDERS,
        stake_usd=STAKE_USD,
    )))
node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
node.build()

if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
