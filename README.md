# PolyCrypto

An ML research and trading platform for [Polymarket](https://polymarket.com)'s 5-minute Bitcoin up/down markets, built to answer one question honestly: **is there a tradeable edge?**

It streams Binance and Polymarket data in real time, predicts each 5-minute move with a calibrated CatBoost model, sizes positions with the Kelly criterion, and executes through a risk-controlled NautilusTrader harness with a dry-run mode. After 17 documented experiments the answer is **no edge net of trading costs**, reached without deploying any capital. Every experiment's question, method, result and decision is in [EXPERIMENTS.md](EXPERIMENTS.md).

> **Disclaimer:** research and educational project, not financial advice. `LIVE` mode signs and submits real orders with real money. No strategy in this repo passed the forward-test gate, and none was traded with real capital. Use at your own risk.

## Results at a glance

| | |
|---|---|
| Training data | 364,676 five-minute epochs rebuilt from 3.5 years of Binance trade archives, 17× the self-collected dataset (Exp 8) |
| Model | CatBoost + isotonic calibration; out-of-time test AUC 0.77, log-loss 0.571 (Exp 11) |
| Forward tests | A +28% result on June prices (Exp 5) did not survive forward testing; no taker or maker configuration produced a profit with a 95% confidence interval clear of zero (Exp 7–12) |
| Why | In a 3.9-day forward test, the win rate matched the entry price in every price bucket: the market price already *is* the probability (Exp 9) |
| Leakage caught | A "+8.5-point edge" turned out to be using 60 seconds of future data (Exp 14) |
| Settlement change | Polymarket's new TWAP settlement rule derived from data, 94.4% agreement with actual resolutions (Exp 11) |
| Capital lost | $0 |

Why an AUC of 0.77 is not an edge: the model predicts 90 seconds into each epoch, after part of the move has already happened, and the market has already repriced for it. The tradeable question is whether the model beats the price at that moment. It doesn't.

## How it works

```
Binance trades + order book ──► feature engine ──► CatBoost ──► isotonic calibration ──► P(UP)
                                                                                           │
Polymarket order book ──────────────────────────► EV gate + Kelly sizing ◄─────────────────┘
                                                          │
                                  risk manager (drawdown, tilt, spend caps)
                                                          │
                        paper engine  ·  Polymarket CLOB  ·  NautilusTrader dry run
```

A new market opens every 5 minutes. The V2 predictor waits until 90 seconds into each one, predicts once, and the EV gate only trades when the model disagrees with the Polymarket price by enough to pay for the spread. A background task discovers each market from its predictable slug (`btc-updown-5m-{timestamp}`), settles the previous one, and allows at most one trade per epoch.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | The bot: WebSocket ingestion (`ingestion/`); features, labelers, trainers, predictors and calibration (`ml/`); EV and Kelly sizing, risk manager, paper engine, executor and market discovery (`trading/`); terminal dashboard (`monitoring/`); `main.py`, `backtest.py` |
| `nautilus_spike/` | NautilusTrader harness: signal, taker and maker strategies, dry-run and canary runners, forward-test reports (Exp 5–12) |
| `exp8/` | 3.5-year Binance archive pipeline, price-only features, TWAP label test (Exp 8, 11) |
| `exp13/`–`exp16/` | Order-flow, wallet-skill, longshot-bias and cross-market studies |
| `weather/` | Weather-market pilot (Exp 17) |
| `tests/`, `src/tests/` | 87 tests |
| `EXPERIMENTS.md` | The decision log |

Datasets, trained models and logs are not committed (tens of GB); the scripts below regenerate them.

## Quick start

Python 3.11+.

```bash
git clone <repo-url> && cd PolyCrypto
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest                                   # 85 pass; 2 skip until a trained model exists

# Smoke-test the trading pipeline on synthetic data (no API keys, no downloads)
python src/tests/generate_test_data.py
python src/backtest.py --dataset test_dataset_labeled.csv
```

The synthetic backtest only checks that the pipeline runs end to end. Its PnL is meaningless.

### Collect data, train, paper trade (V2 model)

```bash
cp .env.example .env                     # starts in OPERATION_MODE=DATA_COLLECTION

python src/main.py                       # 1. stream Binance features to dataset.csv
python src/ml/label_dataset_v3.py        # 2. label 5-minute epochs (streaming, constant memory)
python src/ml/train_model_v2.py --output catboost_model_v2.cbm   # 3. train + calibrate

# 4. in .env set OPERATION_MODE=PAPER, MODEL_VERSION=v2, MODEL_PATH=catboost_model_v2.cbm
python src/main.py                       #    paper-trade against live Polymarket books
python src/monitoring/dashboard.py       #    read-only dashboard, in a second terminal
```

Training uses a chronological train / calibration / test split. Configurations are chosen on the calibration set and the test set is scored once.

### Multi-year archive model (Exp 8, 11)

Download monthly `BTCUSDT-aggTrades-YYYY-MM.zip` files from Binance's public data archive ([data.binance.vision](https://data.binance.vision)) into `exp8/archive/`, then:

```bash
python exp8/build_multiyear.py           # month-at-a-time, memory-bounded
python exp8/train_full.py
```

### NautilusTrader harness

The harness runs in its own Python 3.13 environment and needs a trained model from one of the steps above.

```bash
cd nautilus_spike
python3.13 -m venv venv313 && venv313/bin/pip install -r requirements.txt
./run_dryrun.sh                          # logs would-be orders; never sends one
venv313/bin/python dryrun_report.py      # score the latest run against real resolutions
```

`run_canary.sh` is the capped live runner ($10 stakes, keys read from `.env`). Every recorded run was a dry run: no strategy passed the gate, so no live order was ever sent.

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OPERATION_MODE` | `DATA_COLLECTION` | `DATA_COLLECTION`, `PAPER` or `LIVE` |
| `MODEL_VERSION` | `v1` | `v1` per-tick model or `v2` epoch-aligned model |
| `MODEL_PATH` | `catboost_model.cbm` | Trained model (its calibrator is loaded from `<name>_calibrator.pkl`) |
| `DATASET_PATH` | `dataset.csv` | Feature CSV written in `DATA_COLLECTION` mode |
| `MIN_MODEL_CONFIDENCE` | `0.08` | Skip trades when P(UP) is within this distance of 0.5 |
| `MAX_SINGLE_TRADE_USD` | `50` | Hard per-order cap in the executor |
| `MAX_DAILY_SPEND_USD` | `200` | Hard daily spend cap in the executor |
| `STARTING_BANKROLL` | `1000` | Bankroll for the risk manager and paper engine |
| `PREDICTION_FREQUENCY_MS` | `100` collecting, `2000` trading | Prediction cadence |
| `BINANCE_WS_URL` | Binance Futures stream | Override, e.g. for Binance US |
| `POLY_PRIVATE_KEY`, `POLY_FUNDER`, `POLY_SIGNATURE_TYPE` | none, none, `1` | Wallet credentials, `LIVE` only. Never commit `.env`. |

## Lessons

- **Score against prices you can actually trade at.** Paper trading showed +580% in two days on fills that weren't real. Real prices gave +28%, and forward tests gave less than zero (Exp 5, 7, 9).
- **Choose models on a calibration set, never the test set.** The original trainer selected on the test set, which helped turn a 0.52 AUC into a reported 0.575 (Exp 4).
- **When a result looks too good, check what the inputs could have known.** The best-looking signal of the project was a look-ahead leak (Exp 14).
- **A slightly better forecaster is not an edge once you pay the spread**, and posting resting orders with a weak signal gets you adversely selected (Exp 9, 10, 12).

## Testing

`pytest` runs 87 tests: feature engine, EV and Kelly maths, risk manager and paper settlement (`src/tests/`), plus dashboard rendering and live-readiness checks (`tests/`). Two live-readiness tests skip until a trained V2 model and calibrator are present.
