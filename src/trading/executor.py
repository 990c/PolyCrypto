import time
from collections import deque
from typing import Optional

from loguru import logger


class Executor:
    """Routes live orders to the Polymarket CLOB. All non-LIVE modes are no-ops.

    Spend limits here are enforced independently of the model/EV pipeline so a
    bad prediction or sizing bug upstream cannot drain the wallet.
    """

    MIN_FILL_USD = 1.0
    USDC_DECIMALS = 1e6
    SPEND_WINDOW_SECS = 86400

    def __init__(self, mode: str, host: str, chain_id: int, private_key: str, funder: str,
                 signature_type: int = 1,
                 max_single_trade_usd: float = 50.0,
                 max_daily_spend_usd: float = 200.0):
        self.mode = mode
        self.client = None
        self.signer_address: Optional[str] = None
        self.max_single_trade_usd = max_single_trade_usd
        self.max_daily_spend_usd = max_daily_spend_usd
        self._spend_log: deque = deque()  # (timestamp, usd) tuples within the last 24h

        if self.mode != "LIVE":
            logger.info(f"{self.mode} mode: Executor will not route live orders.")
            return

        try:
            from eth_account import Account
            from py_clob_client.client import ClobClient

            self.signer_address = Account.from_key(private_key).address
            # Funder == signer means a plain EOA wallet; proxy signature types
            # (1=Polymarket proxy, 2=Gnosis Safe) only apply when they differ.
            if funder.lower() == self.signer_address.lower():
                signature_type = 0

            self.client = ClobClient(
                host,
                key=private_key,
                chain_id=chain_id,
                signature_type=signature_type,
                funder=funder,
            )
            # L2 auth: order posting requires derived API credentials, not the raw key
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            logger.info(
                f"LIVE mode: ClobClient ready | signer={self.signer_address[:10]}... "
                f"funder={funder[:10]}... | sig_type={signature_type} | "
                f"caps: ${max_single_trade_usd:.0f}/trade, ${max_daily_spend_usd:.0f}/24h"
            )
        except ImportError:
            logger.error("py_clob_client / eth_account not installed. LIVE mode unavailable.")
        except Exception as e:
            logger.error(f"ClobClient init failed: {e}")

    def check_spend_limits(self, usd_amount: float) -> bool:
        """Hard caps independent of model output. Returns True if the spend is allowed."""
        if usd_amount > self.max_single_trade_usd:
            logger.warning(
                f"Spend guard: ${usd_amount:.2f} exceeds per-trade cap ${self.max_single_trade_usd:.2f}"
            )
            return False

        now = time.time()
        while self._spend_log and now - self._spend_log[0][0] > self.SPEND_WINDOW_SECS:
            self._spend_log.popleft()
        spent_24h = sum(amount for _, amount in self._spend_log)
        if spent_24h + usd_amount > self.max_daily_spend_usd:
            logger.warning(
                f"Spend guard: ${spent_24h:.2f} spent in 24h + ${usd_amount:.2f} "
                f"exceeds daily cap ${self.max_daily_spend_usd:.2f}"
            )
            return False
        return True

    def record_spend(self, usd_amount: float) -> None:
        self._spend_log.append((time.time(), usd_amount))

    def get_usdc_balance(self) -> Optional[float]:
        """Reads the funder wallet's USDC collateral balance from the CLOB. None on failure."""
        if not self.client:
            return None
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            resp = self.client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            if not isinstance(resp, dict) or resp.get("balance") is None:
                logger.warning(f"Unexpected balance response: {resp}")
                return None
            balance = float(resp["balance"]) / self.USDC_DECIMALS
            allowances = resp.get("allowances")
            if allowances is not None:
                # Zero allowance means the exchange contract can't move USDC -> orders will fail
                if all(float(v or 0) == 0 for v in allowances.values()):
                    logger.warning(
                        "USDC allowance is 0 for all exchange contracts. "
                        "Approve USDC on Polymarket (or run client.update_balance_allowance) before trading."
                    )
            return balance
        except Exception as e:
            logger.error(f"Balance query failed: {e}")
            return None

    def execute(self, action: str, token_id: str, limit_price: float,
                size_usd: float, available_liq_usd: float = 0.0) -> bool:
        """
        Routes an order to Polymarket as Fill-or-Kill (FOK).
        - BUY_UP: BUY the YES/UP token at limit_price
        - BUY_DOWN: BUY the NO/DOWN token at limit_price

        limit_price is the actual fill price from the LOB (best ask/bid).
        available_liq_usd is the LOB liquidity verified by the caller.
        The order is capped to available liquidity so it fills completely.
        FOK ensures no orphaned orders — it either fills fully or cancels.
        """
        if self.mode != "LIVE" or not self.client:
            return False
        if not token_id:
            logger.warning(f"LIVE: skipped {action} — missing token_id")
            return False

        # Cap to available liquidity
        fill_size = min(size_usd, available_liq_usd) if available_liq_usd > 0 else size_usd
        if fill_size < self.MIN_FILL_USD:
            logger.debug(f"LIVE: skipped {action} — fill size too small (${fill_size:.2f})")
            return False

        if not self.check_spend_limits(fill_size):
            return False

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            price = round(limit_price, 4)
            if price <= 0:
                return False
            tokens = round(fill_size / price, 2)
            if tokens <= 0:
                return False

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=tokens,
                side=BUY,
            )
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order, orderType=OrderType.FOK)

            # FOK: check if order was actually filled
            success = resp.get("success", False) if isinstance(resp, dict) else False
            if success:
                self.record_spend(fill_size)
                logger.info(f"LIVE FOK filled: {action} {tokens:.2f} tokens @ {price:.4f} | resp: {resp}")
            else:
                logger.warning(f"LIVE FOK rejected: {action} @ {price:.4f} | resp: {resp}")
            return success

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False
