"""Slug builder for Polymarket BTC 5-minute UpDown markets.

Referenced by dotted path from PolymarketInstrumentProviderConfig
(event_slug_builder="slug_builder_5m:build_btc_updown_5m_slugs").
"""
import time

EPOCH_SECONDS = 300


def build_btc_updown_5m_slugs() -> list[str]:
    """Slugs for the current epoch plus the next few (pattern btc-updown-5m-{ts})."""
    now = int(time.time())
    base = now - (now % EPOCH_SECONDS)
    return [f"btc-updown-5m-{base + i * EPOCH_SECONDS}" for i in range(4)]
