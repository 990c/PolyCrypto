#!/bin/zsh
# LIVE canary runner — reads secrets from ../.env (never hardcode keys here).
# Caps: $10 stake, $15/order RiskEngine, 5 orders/min. Ctrl-C to stop.
cd "$(dirname "$0")"
set -a
grep -E '^(POLY_PRIVATE_KEY|POLY_FUNDER|POLY_SIGNATURE_TYPE|POLYMARKET_API_KEY|POLYMARKET_API_SECRET|POLYMARKET_PASSPHRASE)=' ../.env | sed 's/^POLY_PRIVATE_KEY=/POLYMARKET_PK=/; s/^POLY_FUNDER=/POLYMARKET_FUNDER=/; s/^POLY_SIGNATURE_TYPE=/POLYMARKET_SIGNATURE_TYPE=/' > /tmp/naut_canary_env.$$
source /tmp/naut_canary_env.$$ && rm -f /tmp/naut_canary_env.$$
set +a
: "${POLYMARKET_SIGNATURE_TYPE:=1}"
export POLYMARKET_SIGNATURE_TYPE
export BINANCE_API_KEY=dummy BINANCE_API_SECRET=dummy
if [[ -z "$POLYMARKET_PK" || -z "$POLYMARKET_FUNDER" ]]; then
  echo "POLY_PRIVATE_KEY / POLY_FUNDER missing from ../.env — aborting."; exit 1
fi
# API creds: derive once with venv313/bin/python ../nautilus_spike/… create_api_key.py
# and export POLYMARKET_API_KEY/SECRET/PASSPHRASE here, or leave for the adapter.
export POLY_LIVE_ORDERS=1
export NAUT_STAKE_USD=10
echo ">>> LIVE ORDERS ENABLED — canary caps in force. Ctrl-C to stop. <<<"
exec ./venv313/bin/python run_trading_node.py
