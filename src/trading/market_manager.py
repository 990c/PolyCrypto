import re
import time
import requests
from loguru import logger
from typing import Optional


class MarketManager:
    """
    Discovers and tracks the currently active Polymarket 5-minute BTC UP/DOWN market.
    Uses CLOB API for token data, Gamma API for market discovery.
    """
    GAMMA_API = "https://gamma-api.polymarket.com"

    # Patterns that indicate a 5-minute BTC market (case-insensitive)
    MARKET_PATTERNS = [
        re.compile(r"bitcoin.*up.*down", re.IGNORECASE),
        re.compile(r"btc.*up.*down", re.IGNORECASE),
        re.compile(r"btc.*5.?min", re.IGNORECASE),
        re.compile(r"bitcoin.*5.?min", re.IGNORECASE),
        re.compile(r"will.*btc.*go.*(up|down).*next.*5", re.IGNORECASE),
        re.compile(r"btc.*price.*5.?m", re.IGNORECASE),
    ]

    # Slug pattern for direct CLOB lookup
    SLUG_PATTERN = re.compile(r"btc-updown-5m-\d+")

    def __init__(self, host: str):
        self.host = host
        self.current_market_id: Optional[str] = None
        self.current_up_token: Optional[str] = None
        self.current_down_token: Optional[str] = None
        self.current_condition_id: Optional[str] = None
        self.current_epoch_end: float = 0.0  # unix timestamp when current market resolves
        # The market we just rotated away from — needed to settle its positions,
        # since adoption of the new market overwrites all current_* fields.
        self.previous_market_id: Optional[str] = None
        self.previous_condition_id: Optional[str] = None
        self.last_poll_time: float = 0.0
        self._fee_cache: Optional[float] = None
        self._fee_cache_time: float = 0.0
        self._last_fallback_time: float = 0.0

    def _matches_btc_5m(self, text: str) -> bool:
        return any(p.search(text) for p in self.MARKET_PATTERNS)

    def _extract_tokens(self, market_data: dict) -> tuple:
        """Extract UP/DOWN token IDs from market data. Works with both Gamma and CLOB formats."""
        tokens = market_data.get("tokens", [])
        up_token = None
        down_token = None

        for t in tokens:
            outcome = (t.get("outcome", "") or "").upper()
            tid = t.get("token_id", "")
            if outcome in ("YES", "UP"):
                up_token = tid
            elif outcome in ("NO", "DOWN"):
                down_token = tid

        # Fallback: first token = UP, second = DOWN
        if not up_token and tokens:
            up_token = tokens[0].get("token_id")
        if not down_token and len(tokens) > 1:
            down_token = tokens[1].get("token_id")

        return up_token, down_token

    def _fetch_from_clob(self, condition_id: str) -> Optional[dict]:
        """Fetch full market data (including tokens) from CLOB API."""
        try:
            resp = requests.get(f"{self.host}/markets/{condition_id}", timeout=5)
            if resp.ok:
                return resp.json()
        except Exception as e:
            logger.warning(f"CLOB lookup failed for {condition_id[:20]}: {e}")
        return None

    def poll_active_market(self) -> bool:
        """
        Discovers the active 5M BTC UP/DOWN market.
        Strategy: predictable slug lookup (primary), then fallback to question matching.
        Returns True if a NEW market was discovered.

        Sticks with the current market until its epoch ends to avoid flip-flopping
        between overlapping markets.
        """
        now = time.time()
        if self.current_condition_id and now < self.current_epoch_end:
            return False

        try:
            if self._poll_by_timestamp_slug():
                return True
            # Fallback at most every 30s to avoid DNS/log spam
            now = time.time()
            if now - self._last_fallback_time < 30:
                return False
            self._last_fallback_time = now
            return self._poll_gamma_by_question()
        except Exception as e:
            logger.error(f"Market discovery error: {e}")
            return False

    def _poll_by_timestamp_slug(self) -> bool:
        """
        BTC 5-min markets have predictable slugs: btc-updown-5m-{unix_ts}
        where unix_ts is the 300-second boundary for the epoch start.
        Picks the epoch that is currently open (started but not yet ended).
        """
        now = int(time.time())
        base = now - (now % 300)

        # Try current epoch first, then next (if current just ended)
        for epoch_start in [base, base + 300]:
            epoch_end = epoch_start + 300
            # Skip epochs that already ended
            if epoch_end <= now:
                continue

            slug = f"btc-updown-5m-{epoch_start}"
            try:
                resp = requests.get(f"{self.GAMMA_API}/events", params={"slug": slug}, timeout=5)
                if not resp.ok:
                    continue
                events = resp.json()
                if not events:
                    continue

                event = events[0]
                if event.get("closed", False):
                    continue

                markets = event.get("markets", [])
                if markets:
                    if self._try_adopt_market(markets[0], epoch_end=epoch_end):
                        return True
            except requests.exceptions.RequestException:
                continue
        return False

    def _poll_gamma_by_question(self) -> bool:
        """Fallback: search Gamma markets by question text matching."""
        try:
            resp = requests.get(f"{self.GAMMA_API}/markets", params={
                "active": "true",
                "closed": "false",
                "limit": 100,
            }, timeout=5)
            if not resp.ok:
                return False

            markets = resp.json()
            for m in markets:
                question = m.get("question", "") or ""
                slug = m.get("market_slug", "") or m.get("slug", "") or ""
                if self._matches_btc_5m(question) or self.SLUG_PATTERN.search(slug):
                    # Estimate epoch end from slug timestamp if available
                    slug_match = self.SLUG_PATTERN.search(slug)
                    if slug_match:
                        epoch_start = int(slug_match.group().split("-")[-1])
                        epoch_end = epoch_start + 300
                    else:
                        epoch_end = time.time() + 300
                    return self._try_adopt_market(m, epoch_end=epoch_end)
        except requests.exceptions.RequestException as e:
            logger.debug(f"Gamma question search failed: {e}")
        return False

    def _try_adopt_market(self, market: dict, epoch_end: float = 0.0) -> bool:
        """Adopt a discovered market — fetch tokens from CLOB if needed."""
        condition_id = market.get("conditionId") or market.get("condition_id", "")
        if not condition_id or condition_id == self.current_condition_id:
            return False

        # Try tokens from Gamma first
        up_token, down_token = self._extract_tokens(market)

        # If Gamma didn't have tokens, fetch from CLOB API
        if not up_token:
            clob_data = self._fetch_from_clob(condition_id)
            if clob_data:
                up_token, down_token = self._extract_tokens(clob_data)

        if not up_token:
            question = market.get("question", "?")
            logger.warning(f"Market found but no tokens: {question}")
            return False

        self.previous_condition_id = self.current_condition_id
        self.previous_market_id = self.current_market_id
        self.current_condition_id = condition_id
        self.current_market_id = market.get("id", condition_id)
        self.current_up_token = up_token
        self.current_down_token = down_token
        self.current_epoch_end = epoch_end
        self.last_poll_time = time.time()
        self._fee_cache = None

        question = market.get("question", "")
        logger.info(f"New 5M market: {question} | UP: {up_token[:20]}... | DOWN: {down_token[:20] if down_token else 'N/A'}...")

        return True

    def get_fee_rate(self) -> float:
        """Fetches fee config from CLOB, cached for 60s."""
        now = time.time()
        if self._fee_cache is not None and (now - self._fee_cache_time) < 60:
            return self._fee_cache

        try:
            if self.current_condition_id:
                clob_data = self._fetch_from_clob(self.current_condition_id)
                if clob_data:
                    # CLOB returns fee in basis points (e.g. 1000 = 10%)
                    raw_fee = float(clob_data.get("taker_base_fee", 0))
                    self._fee_cache = raw_fee / 10000.0 if raw_fee > 1 else raw_fee
                    self._fee_cache_time = now
                    return self._fee_cache
        except Exception:
            pass
        return 0.0

    @staticmethod
    def parse_winner(market_data: dict) -> Optional[str]:
        """Extracts the winning side ("UP"/"DOWN") from CLOB market data, or None if unresolved."""
        for t in market_data.get("tokens", []):
            if t.get("winner"):
                outcome = (t.get("outcome", "") or "").upper()
                if outcome in ("YES", "UP"):
                    return "UP"
                if outcome in ("NO", "DOWN"):
                    return "DOWN"
        return None

    def get_market_winner(self, condition_id: Optional[str],
                          attempts: int = 3, delay_secs: float = 2.0) -> Optional[str]:
        """
        Queries the CLOB for the market's actual resolution.
        Retries briefly because 5-minute markets resolve a few seconds after close.
        Returns "UP", "DOWN", or None if unresolved/unreachable.
        """
        if not condition_id:
            return None
        for attempt in range(attempts):
            data = self._fetch_from_clob(condition_id)
            if data:
                winner = self.parse_winner(data)
                if winner:
                    return winner
            if attempt < attempts - 1:
                time.sleep(delay_secs)
        return None

    def get_current_btc_price(self) -> Optional[float]:
        """Quick BTC price from Binance REST for settlement."""
        try:
            # Try futures first, fall back to spot
            for url in [
                "https://fapi.binance.com/fapi/v1/ticker/price",
                "https://api.binance.us/api/v3/ticker/price",
            ]:
                resp = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=3)
                if resp.ok:
                    return float(resp.json().get("price", 0))
        except Exception:
            pass
        return None
