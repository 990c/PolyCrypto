#!/bin/zsh
# Overnight DRY-RUN of the Nautilus trading node (no orders are ever sent).
# Records everything to nautilus_spike/logs/naut_canary_*.log
cd "$(dirname "$0")"
export POLYMARKET_PK="0x1111111111111111111111111111111111111111111111111111111111111111"
export POLYMARKET_API_KEY=dummy POLYMARKET_API_SECRET=ZHVtbXk= POLYMARKET_PASSPHRASE=dummy
export POLYMARKET_FUNDER=0x0000000000000000000000000000000000000000
export BINANCE_API_KEY=dummy BINANCE_API_SECRET=dummy
# POLY_LIVE_ORDERS is deliberately NOT set -> dry_run stays on
exec ./venv313/bin/python run_trading_node.py
