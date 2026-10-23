import sys
import os

# Force unbuffered output on Windows
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import asyncio
import signal
import time

from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from ingestion.binance_ws import BinanceWebsocket
from ingestion.polymarket_ws import PolymarketWebsocket
from ml.features import FeatureEngine
from ml.predictor import Predictor
from ml.predictor_v2 import PredictorV2
from trading.ev_calculator import EVCalculator
from trading.risk_manager import RiskManager
from trading.executor import Executor
from trading.paper_engine import PaperEngine
from trading.market_manager import MarketManager
from trading.trade_logger import TradeLogger
from monitoring.status_emitter import StatusEmitter

STARTING_BANKROLL = float(os.getenv("STARTING_BANKROLL", "1000.0"))


class Bot:
    def __init__(self):
        self.mode = Config.OPERATION_MODE
        self.pred_freq = Config.PREDICTION_FREQUENCY_MS / 1000.0

        # Modules
        self.binance_ws = BinanceWebsocket(Config.BINANCE_WS_URL)
        self.poly_ws = None
        self.market_manager = MarketManager(Config.POLY_HOST)
        self.feature_engine = FeatureEngine(window_seconds=Config.WINDOW_SIZE_SECONDS)
        if Config.MODEL_VERSION == "v2":
            logger.info("Using PredictorV2 (epoch-aligned, model={})", Config.MODEL_PATH)
            self.predictor = PredictorV2(self.mode, Config.MODEL_PATH, Config.DATASET_PATH)
        else:
            logger.info("Using Predictor v1 (per-tick, model={})", Config.MODEL_PATH)
            self.predictor = Predictor(self.mode, Config.MODEL_PATH, Config.DATASET_PATH)
        self.ev_calculator = EVCalculator(
            min_model_confidence=Config.MIN_MODEL_CONFIDENCE,
            model_version=Config.MODEL_VERSION,
        )
        self.risk_manager = RiskManager(starting_bankroll=STARTING_BANKROLL)
        self.executor = Executor(self.mode, Config.POLY_HOST, Config.POLY_CHAIN_ID,
                                  Config.POLY_PRIVATE_KEY, Config.POLY_FUNDER,
                                  signature_type=Config.POLY_SIGNATURE_TYPE,
                                  max_single_trade_usd=Config.MAX_SINGLE_TRADE_USD,
                                  max_daily_spend_usd=Config.MAX_DAILY_SPEND_USD)
        self.paper_engine = PaperEngine(start_balance=STARTING_BANKROLL)
        self.trade_logger = TradeLogger(log_path="trades.csv", starting_bankroll=STARTING_BANKROLL)
        self.status = StatusEmitter()
        self._last_wallet_usdc = None

        # Runtime
        self.is_running = False
        self.last_prediction_time = 0.0
        self.epoch_entry_price = 0.0
        self._loop = None
        self._poly_ws_task = None
        self._prediction_in_flight = False
        self._epoch_traded = False
        self._lag_skip_count = 0
        self._confidence_skip_count = 0
        self._last_status_time = 0.0

    def setup_callbacks(self):
        self.binance_ws.set_callbacks(
            depth_cb=self._on_binance_depth,
            trade_cb=self._on_binance_trade
        )

    def _on_binance_trade(self, price, qty, is_buyer_maker):
        self.feature_engine.add_trade(price, qty, is_buyer_maker)

    def _on_binance_depth(self, bids, asks):
        now = time.time()
        if now - self.last_prediction_time >= self.pred_freq:
            self.last_prediction_time = now

            if self.binance_ws.last_lag_ms > 150:
                self._lag_skip_count += 1
                return

            # Skip if a previous prediction cycle is still running
            if self._prediction_in_flight:
                return

            # Schedule coroutine safely on the event loop
            if self.is_running and self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self._prediction_cycle(bids, asks), self._loop)

    def _check_lob_fillable(self, action: str, target_price: float):
        """Check Polymarket LOB for fillability. Returns (fill_price, liq_usd) or None."""
        if action == "BUY_UP":
            if not self.poly_ws.asks:
                return None
            best_ask = min(self.poly_ws.asks.keys())
            if best_ask > target_price + 0.0001:
                logger.debug(f"LOB: skipped {action} — best ask {best_ask:.4f} > target {target_price:.4f}")
                return None
            # Realistic fill: use best ask (what you'd actually pay), not target
            fill_price = best_ask
            liq_tokens = sum(sz for px, sz in self.poly_ws.asks.items() if px <= target_price)
        else:
            if not self.poly_ws.bids:
                return None
            best_bid = max(self.poly_ws.bids.keys())
            no_ask = round(1.0 - best_bid, 4)
            if no_ask > target_price + 0.0001:
                logger.debug(f"LOB: skipped {action} — best NO ask {no_ask:.4f} > target {target_price:.4f}")
                return None
            fill_price = no_ask
            liq_tokens = sum(sz for px, sz in self.poly_ws.bids.items() if px >= 1.0 - target_price)

        available_liq_usd = liq_tokens * fill_price if liq_tokens > 0 else 0.0
        if available_liq_usd < 1.0:
            logger.debug(f"LOB: skipped {action} — insufficient liquidity (${available_liq_usd:.2f})")
            return None

        return fill_price, available_liq_usd

    async def _prediction_cycle(self, bids, asks):
        if not self.is_running:
            return
        self._prediction_in_flight = True
        try:
            features = self.feature_engine.compute_features(bids, asks)
            prob_up = await self.predictor.evaluate(features)

            if self.mode == "DATA_COLLECTION":
                return

            if not self.poly_ws or not self.poly_ws.bids:
                return

            fee_rate = self.market_manager.get_fee_rate()

            ev_result = self.ev_calculator.evaluate_ev(
                prob_up=prob_up,
                poly_bids=self.poly_ws.bids,
                poly_asks=self.poly_ws.asks,
                fee_rate=fee_rate,
                bankroll=self.risk_manager.current_bankroll
            )

            action = ev_result["action"]
            self.status.update(**self._signal_snapshot(prob_up, ev_result))
            if action == "HOLD":
                self._confidence_skip_count += 1
                return

            # Limit to 1 trade per epoch to avoid compounding risk
            if self._epoch_traded:
                return

            target_price = ev_result["target_price"]
            bet_size = ev_result["bet_size_usd"]
            ev = ev_result["ev"]

            if not self.risk_manager.validate_trade(bet_size):
                return

            logger.debug(f"Signal: {action} @ {target_price:.4f} | ${bet_size:.2f} | EV={ev:.4f} | P(UP)={prob_up:.3f}")

            market_id = self.market_manager.current_market_id or "unknown"

            # Verify fillability on Polymarket LOB (shared by paper + live)
            if not self.poly_ws:
                return

            lob_check = self._check_lob_fillable(action, target_price)
            if lob_check is None:
                return
            fill_price, available_liq_usd = lob_check

            if self.mode == "PAPER":
                success = self.paper_engine.execute_order(action, fill_price, bet_size, available_liq_usd)
                if success:
                    self._epoch_traded = True
                    self.risk_manager.register_position(bet_size)
                    self.trade_logger.log_trade(
                        market_id=market_id, action=action,
                        price=fill_price, size_usd=bet_size,
                        prob_up=prob_up, ev=ev,
                        balance_after=self.paper_engine.balance
                    )
                    logger.info(self.paper_engine.summary())

            elif self.mode == "LIVE":
                if action == "BUY_UP":
                    token_id = self.market_manager.current_up_token or ""
                else:
                    token_id = self.market_manager.current_down_token or ""
                success = self.executor.execute(
                    action, token_id, fill_price, bet_size, available_liq_usd
                )
                if success:
                    self._epoch_traded = True
                    # Mirror the real fill into the ledger so settlement and the
                    # risk circuit-breakers track actual positions and PnL.
                    if not self.paper_engine.execute_order(action, fill_price, bet_size, available_liq_usd):
                        logger.warning(
                            "LIVE fill succeeded but ledger rejected it — "
                            "ledger balance is now out of sync with the wallet."
                        )
                    self.risk_manager.register_position(bet_size)
                    self.trade_logger.log_trade(
                        market_id=market_id, action=action,
                        price=fill_price, size_usd=bet_size,
                        prob_up=prob_up, ev=ev,
                        balance_after=self.paper_engine.balance
                    )

        except Exception as e:
            logger.error(f"Prediction cycle error: {e}")
        finally:
            self._prediction_in_flight = False

    async def _market_rotation_loop(self):
        while self.is_running:
            try:
                discovered = self.market_manager.poll_active_market()
                if discovered:
                    await self._handle_market_rotation()
            except Exception as e:
                logger.error(f"Market rotation error: {e}")

            # Periodic status log every 60s so silent epochs aren't mysterious
            now = time.time()
            if now - self._last_status_time >= 60:
                self._last_status_time = now
                lag_ms = self.binance_ws.last_lag_ms if hasattr(self.binance_ws, 'last_lag_ms') else 0
                logger.info(
                    f"Status: lag_skips={self._lag_skip_count} | conf_skips={self._confidence_skip_count} | "
                    f"lag={lag_ms:.0f}ms | traded={self._epoch_traded} | "
                    f"balance=${self.paper_engine.balance:.2f}"
                )
                self._lag_skip_count = 0
                self._confidence_skip_count = 0
                self.status.update(**self._risk_snapshot())

            # Poll faster near epoch boundaries to minimize wasted time
            time_left = self.market_manager.current_epoch_end - time.time()
            if time_left <= 0:
                # Epoch ended or no market — poll every 3s to catch the next one fast
                await asyncio.sleep(3)
            elif time_left < 15:
                # Epoch about to end — poll every 2s to detect new market immediately
                await asyncio.sleep(2)
            else:
                # Mid-epoch — no need to poll frequently
                await asyncio.sleep(30)

    def _gate_state(self, prob_up: float) -> str:
        """Human-readable reason the bot is/isn't eligible to trade right now."""
        now = time.time()
        epoch_start = int(now) - (int(now) % 300)
        warmup_n = len(getattr(self.predictor, "_completed_epochs", []) or [])
        if Config.MODEL_VERSION == "v2":
            if warmup_n < 5:
                return f"warmup {warmup_n}/5"
            if now < epoch_start + 90:
                return f"wait-90s ({int(epoch_start + 90 - now)}s)"
        if abs(prob_up - 0.5) < Config.MIN_MODEL_CONFIDENCE:
            return "low-conviction"
        return "eligible"

    def _signal_snapshot(self, prob_up: float, ev_result: dict) -> dict:
        best_bid = max(self.poly_ws.bids.keys()) if self.poly_ws and self.poly_ws.bids else None
        best_ask = min(self.poly_ws.asks.keys()) if self.poly_ws and self.poly_ws.asks else None
        return {
            "mode": self.mode,
            "model_version": Config.MODEL_VERSION,
            "prob_up": prob_up,
            "gate_state": self._gate_state(prob_up),
            "best_action": ev_result["action"],
            "ev": ev_result["ev"],
            "target_price": ev_result["target_price"],
            "up_bid": best_bid,
            "up_ask": best_ask,
            "epoch_end": self.market_manager.current_epoch_end,
            "epoch_traded": self._epoch_traded,
            "bankroll": self.risk_manager.current_bankroll,
        }

    def _risk_snapshot(self) -> dict:
        rm = self.risk_manager
        drawdown_pct = 0.0
        if rm.window_start_bankroll > 0:
            drawdown_pct = max(0.0, (1 - rm.current_bankroll / rm.window_start_bankroll) * 100)
        spend_24h = sum(a for _, a in self.executor._spend_log)
        return {
            "mode": self.mode,
            "binance_lag_ms": getattr(self.binance_ws, "last_lag_ms", 0),
            "poly_connected": bool(self.poly_ws and self.poly_ws.bids),
            "exposure": rm.active_exposure,
            "exposure_cap": rm.current_bankroll * rm.max_epoch_exposure_pct,
            "drawdown_pct": drawdown_pct,
            "drawdown_limit_pct": rm.daily_drawdown_limit * 100,
            "consec_losses": rm.consecutive_losses,
            "tilt_threshold": rm.max_consecutive_losses,
            "spend_24h": spend_24h,
            "spend_cap": self.executor.max_daily_spend_usd,
            "bankroll": rm.current_bankroll,
            "wallet_usdc": self._last_wallet_usdc,
            "epoch_end": self.market_manager.current_epoch_end,
        }

    def _resolve_winner(self) -> tuple:
        """Settlement source of truth: Polymarket's own resolution for the market
        we just left, falling back to the Binance BTC heuristic if unresolved.
        Returns (winner, source) where source is "resolution" or "heuristic"."""
        winner = self.market_manager.get_market_winner(
            self.market_manager.previous_condition_id
        )
        if winner:
            logger.info(f"Settlement source: Polymarket resolution -> {winner}")
            return winner, "resolution"

        current_btc = self.market_manager.get_current_btc_price()
        if current_btc and self.epoch_entry_price > 0:
            winner = "UP" if current_btc > self.epoch_entry_price else "DOWN"
        else:
            winner = "DOWN"
        logger.warning(
            f"Settlement source: BTC heuristic fallback -> {winner} "
            f"(${self.epoch_entry_price:.0f} -> ${current_btc or 0:.0f})"
        )
        return winner, "heuristic"

    async def _handle_market_rotation(self):
        self._epoch_traded = False
        self.risk_manager.tick_tilt_cooldown()
        old_market_id = (self.market_manager.previous_market_id
                         or self.market_manager.current_market_id or "unknown")

        # Settle previous positions
        if self.risk_manager.active_exposure > 0:
            # get_market_winner blocks on HTTP retries — keep the event loop free
            winner, settle_source = await asyncio.to_thread(self._resolve_winner)

            # Capture positions BEFORE settlement zeroes them
            pre_settle_positions = {k: v for k, v in self.paper_engine.positions.items()}

            epoch_pnl = self.paper_engine.settle_market(winner)
            is_win = epoch_pnl > 0
            self.risk_manager.register_result(is_win)
            self.risk_manager.clear_exposure()
            self.risk_manager.update_bankroll(self.paper_engine.balance)

            self.trade_logger.log_settlement(
                market_id=old_market_id,
                winning_side=winner,
                positions=pre_settle_positions,
                balance_after=self.paper_engine.balance,
                source=settle_source
            )

            stats = self.trade_logger.get_stats()
            logger.info(
                f"Settled: {winner} | epoch PnL ${epoch_pnl:+.2f} | "
                f"WR: {stats['win_rate_pct']:.0f}% | {self.paper_engine.summary()}"
            )

            # LIVE: reconcile ledger against the real wallet. Ledger stays the
            # source of truth for risk (unredeemed winnings lag on-chain), but
            # a large drift means something is wrong — halt new trades.
            if self.mode == "LIVE":
                real_balance = await asyncio.to_thread(self.executor.get_usdc_balance)
                if real_balance is not None:
                    self._last_wallet_usdc = real_balance
                    drift = real_balance - self.paper_engine.balance
                    logger.info(
                        f"Wallet USDC ${real_balance:.2f} | ledger ${self.paper_engine.balance:.2f} | "
                        f"drift ${drift:+.2f}"
                    )
                    if drift < -max(10.0, 0.25 * self.paper_engine.balance):
                        logger.error(
                            "Wallet is far below ledger — halting live trading. "
                            "Check fills/redemptions on polymarket.com before restarting."
                        )
                        self.is_running = False

            self.status.update(force=True, **self._risk_snapshot())

        # Record entry price for next epoch
        btc_now = self.market_manager.get_current_btc_price()
        self.epoch_entry_price = btc_now if btc_now else 0.0

        # Rotate Polymarket WS
        if self.poly_ws:
            self.poly_ws.stop()
        if self._poly_ws_task and not self._poly_ws_task.done():
            self._poly_ws_task.cancel()
        self.poly_ws = PolymarketWebsocket(self.market_manager.current_up_token)
        self._poly_ws_task = asyncio.create_task(self.poly_ws.run())
        self._poly_ws_task.add_done_callback(self._on_poly_ws_done)

    @staticmethod
    def _on_poly_ws_done(task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Polymarket WS task crashed: {exc}")

    async def _live_preflight(self) -> bool:
        """Verifies auth, funds, and allowances before any live order can fire.
        Seeds all accounting from the real wallet balance."""
        if not self.executor.client:
            logger.error("CLOB client failed to initialize — check POLY_PRIVATE_KEY / POLY_SIGNATURE_TYPE.")
            return False

        balance = await asyncio.to_thread(self.executor.get_usdc_balance)
        if balance is None:
            logger.error("Could not read USDC balance from the CLOB — auth or network problem.")
            return False
        if balance < 10.0:
            logger.error(f"Wallet USDC ${balance:.2f} is below the $10 minimum for LIVE trading.")
            return False

        # Real wallet balance replaces STARTING_BANKROLL for every accounting surface
        self.paper_engine.start_balance = balance
        self.paper_engine.balance = balance
        self.risk_manager.seed_bankroll(balance)
        logger.info(
            f"LIVE preflight OK | wallet USDC ${balance:.2f} | "
            f"caps ${Config.MAX_SINGLE_TRADE_USD:.0f}/trade ${Config.MAX_DAILY_SPEND_USD:.0f}/24h"
        )
        return True

    async def run(self):
        self.is_running = True
        self._loop = asyncio.get_running_loop()

        if self.mode == "LIVE" and not await self._live_preflight():
            logger.error("LIVE preflight failed — refusing to start. Fix the issues above or use PAPER mode.")
            self.is_running = False
            return

        logger.info(f"Starting Bot | Mode: {self.mode} | PredFreq: {self.pred_freq}s | "
                    f"Bankroll: ${self.risk_manager.current_bankroll:.0f}")
        self.setup_callbacks()

        self._tasks = [
            asyncio.create_task(self.binance_ws.run()),
            asyncio.create_task(self._market_rotation_loop()),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        logger.info("Shutting down gracefully...")
        self.is_running = False
        self.binance_ws.stop()
        if self.poly_ws:
            self.poly_ws.stop()
        for t in getattr(self, "_tasks", []):
            t.cancel()
        self.predictor.shutdown()
        logger.info("Shutdown complete.")


def main():
    # Persistent log file: settlement sources, fills, and preflight results must
    # be auditable after the fact (stdout alone is lost on terminal close).
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="1 day",
               retention="30 days", level="DEBUG", enqueue=True)

    bot = Bot()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}")
        bot.is_running = False
        if bot._loop and bot._loop.is_running():
            bot._loop.call_soon_threadsafe(lambda: bot._loop.create_task(bot.shutdown()))

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
