import asyncio
import websockets
import orjson
import time
from loguru import logger
from typing import Callable, Optional, List, Tuple

class BinanceWebsocket:
    def __init__(self, url: str):
        self.url = url
        self.depth_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.is_running = False
        
        # Local limit order book cache (depth 20)
        self.bids: List[Tuple[float, float]] = []
        self.asks: List[Tuple[float, float]] = []
        
        self.last_lag_ms = 0
        
    def set_callbacks(self, depth_cb: Callable, trade_cb: Callable):
        self.depth_callback = depth_cb
        self.trade_callback = trade_cb
        
    async def run(self):
        self.is_running = True
        consecutive_failures = 0
        while self.is_running:
            try:
                # Ping interval and timeout for fast failover
                async with websockets.connect(self.url, ping_interval=10, ping_timeout=5) as ws:
                    logger.info(f"Connected to Binance WebSocket: {self.url}")
                    consecutive_failures = 0
                    self.last_lag_ms = 0  # Reset stale lag on reconnect
                    await self._listen(ws)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Binance WS Connection closed.")
            except Exception as e:
                logger.error(f"Binance WS encountered an error: {e}")
            
            if self.is_running:
                consecutive_failures += 1
                backoff = min(1.0 * (2 ** consecutive_failures), 30.0)
                logger.info(f"Reconnecting in {backoff} seconds...")
                await asyncio.sleep(backoff)
                
    async def _listen(self, ws):
        async for msg in ws:
            try:
                data = orjson.loads(msg)
            except Exception:
                continue
            
            if "stream" not in data or "data" not in data:
                continue
                
            stream = data["stream"]
            payload = data["data"]
            
            # Calculate lag via Binance Server Time (E)
            if "E" in payload:
                server_time = payload["E"]
                local_time = int(time.time() * 1000)
                self.last_lag_ms = local_time - server_time
                if self.last_lag_ms > 150:
                    logger.warning(f"Binance WS Lag anomaly detected: {self.last_lag_ms}ms cutoff")
            
            if "depth" in stream:
                # Futures uses "b"/"a", Spot uses "bids"/"asks"
                self.bids = payload.get("b") or payload.get("bids", [])
                self.asks = payload.get("a") or payload.get("asks", [])
                if self.depth_callback:
                    # In asyncio, making sync callbacks relies on them being very fast
                    self.depth_callback(self.bids, self.asks)
                    
            elif "trade" in stream:
                # p = price, q = qty, m = is_buyer_maker (True = sell taker, False = buy taker)
                price = float(payload.get("p", 0))
                qty = float(payload.get("q", 0))
                is_buyer_maker = payload.get("m", False)
                if self.trade_callback:
                    self.trade_callback(price, qty, is_buyer_maker)

    def stop(self):
        self.is_running = False
