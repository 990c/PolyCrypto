"""Dry-run performance report.

Parses `would submit` lines from a naut_canary log, resolves each epoch's
actual winner from Polymarket, and computes the PnL those orders would have
produced IF every FOK filled at the logged price.

Caveat printed with results: 100% fill at the observed ask is the optimistic
bound — measuring the real fill rate is the live canary's job.

Usage: ./venv313/bin/python dryrun_report.py [logfile]
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
WINNER_CACHE = os.path.join(HERE, "winner_cache.json")
LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})[\d.]*Z.*would submit "
    r"(BUY_UP|BUY_DOWN) ([\d.]+) @ ([\d.]+) "
    r"\(\$([\d.]+), prob_up=([\d.]+), ev=([\d.]+)\)"
)


def parse_ts(iso: str) -> float:
    return datetime.fromisoformat(iso + "+00:00").timestamp()


def get_winner(epoch_start: int, cache: dict) -> str | None:
    key = str(epoch_start)
    if key in cache:
        return cache[key]
    winner = None
    try:
        r = requests.get("https://gamma-api.polymarket.com/events",
                         params={"slug": f"btc-updown-5m-{epoch_start}"}, timeout=10)
        evs = r.json() if r.ok else []
        cond = evs[0]["markets"][0].get("conditionId") if evs else None
        if cond:
            r2 = requests.get(f"https://clob.polymarket.com/markets/{cond}", timeout=10)
            if r2.ok:
                for t in r2.json().get("tokens", []):
                    if t.get("winner"):
                        o = (t.get("outcome") or "").upper()
                        winner = "UP" if o in ("YES", "UP") else "DOWN"
        time.sleep(0.12)
    except requests.RequestException:
        pass
    if winner is not None:
        cache[key] = winner
    return winner


def main():
    logs = sorted(
        (os.path.join(HERE, "logs", f) for f in os.listdir(os.path.join(HERE, "logs"))
         if f.startswith("naut_canary_")),
        key=os.path.getmtime,
    )
    path = sys.argv[1] if len(sys.argv) > 1 else logs[-1]
    print(f"log: {path}")

    orders = []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for line in open(path, errors="replace"):
        if "would submit" not in line:
            continue
        m = LINE_RE.match(ansi.sub("", line))
        if not m:
            continue
        ts = parse_ts(m.group(1))
        orders.append({
            "epoch": int(ts) - (int(ts) % 300),
            "side": "UP" if m.group(2) == "BUY_UP" else "DOWN",
            "qty": float(m.group(3)), "price": float(m.group(4)),
            "stake": float(m.group(5)), "prob_up": float(m.group(6)),
            "ev": float(m.group(7)),
        })
    print(f"would-submit orders: {len(orders)}")

    cache = json.load(open(WINNER_CACHE)) if os.path.exists(WINNER_CACHE) else {}
    resolved = []
    for i, o in enumerate(orders):
        w = get_winner(o["epoch"], cache)
        if i % 40 == 0:
            json.dump(cache, open(WINNER_CACHE, "w"))
            print(f"  resolving {i}/{len(orders)}...", flush=True)
        if w is None:
            continue
        win = o["side"] == w
        o["win"] = win
        o["pnl"] = o["stake"] * (1 - o["price"]) / o["price"] if win else -o["stake"]
        resolved.append(o)
    json.dump(cache, open(WINNER_CACHE, "w"))

    n = len(resolved)
    if n == 0:
        print("no resolvable orders")
        return
    wins = sum(o["win"] for o in resolved)
    pnl = sum(o["pnl"] for o in resolved)
    staked = sum(o["stake"] for o in resolved)
    print("\n=== DRY-RUN PERFORMANCE (assumes 100% fill at logged ask) ===")
    print(f"resolved: {n}/{len(orders)} | WR {wins / n * 100:.1f}% | "
          f"PnL ${pnl:+.2f} on ${staked:.0f} staked ({pnl / staked * 100:+.1f}%/$)")
    for side in ("UP", "DOWN"):
        g = [o for o in resolved if o["side"] == side]
        if g:
            gw = sum(o["win"] for o in g)
            print(f"  {side:4s}: n={len(g):3d} WR {gw / len(g) * 100:5.1f}% "
                  f"avg price {sum(o['price'] for o in g) / len(g):.3f} "
                  f"PnL ${sum(o['pnl'] for o in g):+.2f}")
    print("\n  by entry price:")
    for lo, hi in [(0.0, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 1.0)]:
        g = [o for o in resolved if lo <= o["price"] < hi]
        if g:
            gw = sum(o["win"] for o in g)
            be = sum(o["price"] for o in g) / len(g)
            print(f"    [{lo:.2f},{hi:.2f}): n={len(g):3d} WR {gw / len(g) * 100:5.1f}% "
                  f"(breakeven {be * 100:.0f}%) PnL ${sum(o['pnl'] for o in g):+.2f}")
    print("\nCaveat: real FOK orders may not fill at these prices — the live canary")
    print("measures the fill rate, which scales this PnL down proportionally.")


if __name__ == "__main__":
    main()
