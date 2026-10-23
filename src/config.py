import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Binance WS (use BINANCE_WS_URL env var to override)
    BINANCE_WS_URL = os.getenv(
        "BINANCE_WS_URL",
        "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@trade"
    )

    # Polymarket
    POLY_HOST = os.getenv("POLY_HOST", "https://clob.polymarket.com")
    POLY_CHAIN_ID = int(os.getenv("POLY_CHAIN_ID", "137"))
    POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "0x" + "0" * 64)
    POLY_FUNDER = os.getenv("POLY_FUNDER", "0x" + "0" * 40)

    # Engine
    OPERATION_MODE = os.getenv("OPERATION_MODE", "DATA_COLLECTION")  # DATA_COLLECTION | PAPER | LIVE
    _DEFAULT_PRED_FREQ = {"DATA_COLLECTION": 100, "PAPER": 2000, "LIVE": 2000}
    PREDICTION_FREQUENCY_MS = int(os.getenv(
        "PREDICTION_FREQUENCY_MS",
        str(_DEFAULT_PRED_FREQ.get(OPERATION_MODE, 100))
    ))

    # Feature Engineering
    WINDOW_SIZE_SECONDS = 60

    # Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "catboost_model.cbm")
    DATASET_PATH = os.getenv("DATASET_PATH", "dataset.csv")

    # Model version: "v1" (per-tick) or "v2" (epoch-aligned)
    MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

    # Live trading
    # Wallet signature type: 0=EOA (key controls funds directly),
    # 1=Polymarket proxy (email/embedded wallet), 2=Gnosis Safe (MetaMask-created)
    POLY_SIGNATURE_TYPE = int(os.getenv("POLY_SIGNATURE_TYPE", "1"))
    # Hard spend caps enforced in the Executor, independent of model/EV sizing
    MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", "50"))
    MAX_DAILY_SPEND_USD = float(os.getenv("MAX_DAILY_SPEND_USD", "200"))

    # Model conviction gate: skip trades when |prob_up - 0.5| < this value.
    # Default 0.08 per Exp 1+3 (EXPERIMENTS.md): with the isotonic calibrator,
    # V2 trades only at prob_up <= 0.42, the high-conviction plateau where test
    # win rate is 75.2% (137/1683 epochs, ~23 trades/day).
    MIN_MODEL_CONFIDENCE = float(os.getenv("MIN_MODEL_CONFIDENCE", "0.08"))
