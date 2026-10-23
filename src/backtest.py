"""
Backtester
==========
Replays a labeled dataset through the full EV + Risk + Paper Engine pipeline.

Usage:
  python src/backtest.py --dataset test_dataset_labeled.csv --bankroll 1000
  python src/backtest.py --dataset dataset_labeled.csv --model catboost_model.cbm

Prerequisites:
  1. Run bot in DATA_COLLECTION mode to produce dataset.csv
  2. Run `python src/ml/label_dataset.py` to produce dataset_labeled.csv
  3. (Optional) Train catboost_model.cbm
"""
import csv
import sys
import os
import argparse

import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.features import FeatureEngine
from trading.ev_calculator import EVCalculator
from trading.risk_manager import RiskManager
from trading.paper_engine import PaperEngine

FEATURE_COLS = FeatureEngine.FEATURE_NAMES


def load_labeled_dataset(path: str) -> list:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["timestamp"] = float(r["timestamp"])
                r["last_price"] = float(r["last_price"])
                r["target_resolution"] = int(r["target_resolution"])
                for col in FEATURE_COLS:
                    r[col] = float(r.get(col, 0))
                rows.append(r)
            except (ValueError, KeyError):
                continue
    return rows


def group_into_epochs(rows: list, epoch_seconds: int = 300) -> list:
    if not rows:
        return []

    epochs = []
    epoch_start = rows[0]["timestamp"]
    current_epoch = []

    for row in rows:
        if row["timestamp"] - epoch_start >= epoch_seconds:
            if current_epoch:
                epochs.append(current_epoch)
            current_epoch = [row]
            epoch_start = row["timestamp"]
        else:
            current_epoch.append(row)

    if current_epoch:
        epochs.append(current_epoch)
    return epochs


def simulate_prediction(features: dict, model=None) -> float:
    """
    Simulates ML prediction. Uses CatBoost model if provided,
    otherwise a heuristic with enough signal to validate the pipeline.
    """
    if model:
        model_input = [features.get(col, 0) for col in FEATURE_COLS]
        return float(model.predict_proba([model_input])[0][1])

    # Heuristic combining multiple features for directional signal
    obi = features.get("obi", 0)
    cvd = features.get("cvd", 0)
    toxicity = features.get("toxicity", 0.5)
    obi_grad = features.get("obi_gradient", 0)
    cvd_grad = features.get("cvd_gradient", 0)
    vwap_dist = features.get("vwap_distance", 0)
    w_obi = features.get("weighted_obi", 0)

    # Weighted combination — each contributes directional signal
    signal = (
        obi * 0.20 +
        w_obi * 0.15 +
        max(-0.20, min(0.20, cvd * 0.004)) +
        (0.5 - toxicity) * 0.25 +
        obi_grad * 0.08 +
        max(-0.10, min(0.10, cvd_grad * 0.003)) +
        max(-0.10, min(0.10, vwap_dist * 0.001))
    )

    prob = 0.5 + signal
    return max(0.05, min(0.95, prob))


def build_synthetic_lob() -> tuple:
    """
    Simulates a Polymarket LOB centered at ~0.50 (efficient market).
    Our edge comes from the predictor disagreeing with this fair price.
    """
    market_mid = 0.50
    spread = 0.04
    best_ask = round(market_mid + spread / 2, 4)  # 0.52
    best_bid = round(market_mid - spread / 2, 4)  # 0.48

    poly_bids = {best_bid: 500.0}
    poly_asks = {best_ask: 500.0}
    return poly_bids, poly_asks


def run_backtest(dataset_path: str, bankroll: float, model_path: str = None,
                 kelly_frac: float = 0.25, min_ev: float = 0.02):
    logger.info(f"Loading dataset from {dataset_path}...")
    rows = load_labeled_dataset(dataset_path)
    if not rows:
        logger.error("No valid rows found.")
        return None

    logger.info(f"Loaded {len(rows)} rows. Grouping into 5-min epochs...")
    epochs = group_into_epochs(rows)
    logger.info(f"Found {len(epochs)} epochs to simulate.")

    # Load model if provided
    model = None
    if model_path and os.path.exists(model_path):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info(f"Loaded CatBoost model from {model_path}")

    # Initialize — relax tilt protection for backtesting (it's a live-trading guard)
    ev_calc = EVCalculator(kelly_fraction=kelly_frac, min_ev_threshold=min_ev)
    risk_mgr = RiskManager(starting_bankroll=bankroll)
    risk_mgr.max_consecutive_losses = 999  # Disable tilt in backtest
    paper = PaperEngine(start_balance=bankroll)

    # Tracking
    wins = 0
    losses = 0
    total_bets = 0
    epochs_with_trades = 0
    equity_curve = [bankroll]
    trade_log = []

    for epoch_idx, epoch_rows in enumerate(epochs):
        epoch_traded = False
        # Use first row's label: it looks 300s ahead from epoch start, matching settlement
        ground_truth = epoch_rows[0]["target_resolution"]
        winning_side = "UP" if ground_truth == 1 else "DOWN"

        for row in epoch_rows:
            features = {col: row.get(col, 0) for col in FEATURE_COLS}
            features["last_price"] = row["last_price"]

            prob_up = simulate_prediction(features, model)
            poly_bids, poly_asks = build_synthetic_lob()

            ev_result = ev_calc.evaluate_ev(
                prob_up=prob_up,
                poly_bids=poly_bids,
                poly_asks=poly_asks,
                fee_rate=0.0,
                bankroll=risk_mgr.current_bankroll,
            )

            action = ev_result["action"]
            if action != "HOLD" and ev_result["bet_size_usd"] > 0:
                bet_size = ev_result["bet_size_usd"]
                price = ev_result["target_price"]

                if risk_mgr.validate_trade(bet_size):
                    success = paper.execute_order(action, price, bet_size)
                    if success:
                        risk_mgr.register_position(bet_size)
                        epoch_traded = True
                        total_bets += 1

                        trade_log.append({
                            "epoch": epoch_idx,
                            "action": action,
                            "price": price,
                            "size": round(bet_size, 4),
                            "prob_up": round(prob_up, 4),
                            "ev": round(ev_result["ev"], 6),
                            "kelly_f": round(ev_result.get("kelly_f", 0), 6),
                            "balance": round(paper.balance, 2),
                        })

        # Settle epoch
        if epoch_traded:
            epochs_with_trades += 1

            epoch_pnl = paper.settle_market(winning_side)

            if epoch_pnl > 0:
                wins += 1
                risk_mgr.register_result(True)
            else:
                losses += 1
                risk_mgr.register_result(False)
            risk_mgr.clear_exposure()
            risk_mgr.update_bankroll(paper.balance)

        equity_curve.append(paper.balance)

    # --- Final Report ---
    total_settled = wins + losses
    win_rate = (wins / total_settled * 100) if total_settled > 0 else 0
    final_pnl = paper.balance - bankroll
    roi = (final_pnl / bankroll) * 100

    # Risk metrics
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[returns != 0]
    # Annualized Sharpe: 288 five-minute epochs per day, 365 trading days
    periods_per_year = 288 * 365
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 1 and np.std(returns) > 0 else 0
    max_dd = _max_drawdown(equity)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Dataset:              {dataset_path}")
    logger.info(f"  Model:                {'CatBoost' if model else 'Heuristic'}")
    logger.info(f"  Kelly Fraction:       {kelly_frac}")
    logger.info(f"  Min EV Threshold:     {min_ev}")
    logger.info("-" * 60)
    logger.info(f"  Total Epochs:         {len(epochs)}")
    logger.info(f"  Epochs with Trades:   {epochs_with_trades}")
    logger.info(f"  Total Bets Placed:    {total_bets}")
    logger.info(f"  Wins:                 {wins}")
    logger.info(f"  Losses:               {losses}")
    logger.info(f"  Win Rate:             {win_rate:.1f}%")
    logger.info("-" * 60)
    logger.info(f"  Starting Bankroll:    ${bankroll:.2f}")
    logger.info(f"  Final Balance:        ${paper.balance:.2f}")
    logger.info(f"  Net PnL:              ${final_pnl:.2f}")
    logger.info(f"  ROI:                  {roi:+.2f}%")
    logger.info(f"  Sharpe Ratio:         {sharpe:.3f}")
    logger.info(f"  Max Drawdown:         {max_dd:.2f}%")
    logger.info("=" * 60)

    # Profitability verdict
    if roi > 0 and win_rate > 50:
        logger.info("  VERDICT: PROFITABLE - positive ROI with >50% win rate")
    elif roi > 0:
        logger.info("  VERDICT: PROFITABLE - positive ROI (bet sizing compensates for <50% WR)")
    else:
        logger.warning("  VERDICT: NOT PROFITABLE - review strategy parameters")

    # Save trade log
    if trade_log:
        log_path = "backtest_trades.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trade_log[0].keys())
            writer.writeheader()
            writer.writerows(trade_log)
        logger.info(f"  Trade log saved to {log_path}")

    return {
        "win_rate": win_rate,
        "final_balance": paper.balance,
        "pnl": final_pnl,
        "roi": roi,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as a percentage."""
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    parser = argparse.ArgumentParser(description="Backtest BTC 15M trading strategy")
    parser.add_argument("--dataset", default="test_dataset_labeled.csv")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--model", default=None, help="Path to CatBoost model")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    parser.add_argument("--min-ev", type=float, default=0.02, help="Min EV threshold")
    args = parser.parse_args()

    run_backtest(args.dataset, args.bankroll, args.model, args.kelly, args.min_ev)


if __name__ == "__main__":
    main()
