import asyncio
import websockets
import orjson
from loguru import logger
from typing import Dict, Optional, Callable

class PolymarketWebsocket:
    def __init__(self, token_id: str):
        self.url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.token_id = token_id
        self.lob_callback: Optional[Callable] = None
        self.is_running = False
        
        # Local limit order book cache mapping price -> size
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        
    def set_callback(self, lob_cb: Callable):
        self.lob_callback = lob_cb
        
    async def run(self):
        self.is_running = True
        consecutive_failures = 0
        while self.is_running:
            try:
                async with websockets.connect(self.url, ping_interval=10, ping_timeout=5) as ws:
                    logger.info(f"Connected to Polymarket WS for token {self.token_id}")
                    consecutive_failures = 0
                    
                    # Send subscription message
                    sub_msg = {
                        "assets_ids": [self.token_id],
                        "type": "market"
                    }
                    await ws.send(orjson.dumps(sub_msg).decode('utf-8'))
                    
                    await self._listen(ws)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Polymarket WS Connection closed.")
            except Exception as e:
                logger.error(f"Polymarket WS error: {e}")
            
            if self.is_running:
                consecutive_failures += 1
                backoff = min(1.0 * (2 ** consecutive_failures), 30.0)
                logger.info(f"Reconnecting Poly WS in {backoff} seconds...")
                await asyncio.sleep(backoff)
                
    async def _listen(self, ws):
        async for msg in ws:
            try:
                data = orjson.loads(msg)
            except Exception:
                continue
                
            # Polymarket WS payload structure evaluation
            if isinstance(data, list):
                for event in data:
                    self._update_lob(event)
            elif isinstance(data, dict):
                self._update_lob(data)

            if self.lob_callback:
                self.lob_callback(self.bids, self.asks)
                
    MAX_LOB_LEVELS = 200  # Prevent unbounded memory growth

    def _update_lob(self, event: dict):
        if "bids" in event and event["bids"]:
            for bid in event["bids"]:
                price = float(bid.get("price", 0))
                size = float(bid.get("size", 0))
                if size == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = size
            self._trim_lob(self.bids, keep_top=True)

        if "asks" in event and event["asks"]:
            for ask in event["asks"]:
                price = float(ask.get("price", 0))
                size = float(ask.get("size", 0))
                if size == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = size
            self._trim_lob(self.asks, keep_top=False)

    def _trim_lob(self, book: dict, keep_top: bool):
        """Evict stale levels beyond MAX_LOB_LEVELS, keeping best prices."""
        if len(book) <= self.MAX_LOB_LEVELS:
            return
        sorted_prices = sorted(book.keys(), reverse=keep_top)
        for price in sorted_prices[self.MAX_LOB_LEVELS:]:
            del book[price]

    def stop(self):
        self.is_running = False
        self.bids.clear()
        self.asks.clear()
