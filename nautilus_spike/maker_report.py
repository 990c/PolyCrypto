"""Maker dry-run performance report (Exp 10).

Reconstructs each `MAKER would-quote` against the epoch's REAL trade prints:
  - conservative fill: a print traded strictly THROUGH our bid (definitely filled)
  - optimistic fill:  a print traded AT our bid (queue position unknown)
Fill windows are truncated by `MAKER would-cancel` lines (adverse-move guard).
Winners come from the shared winner_cache. Both PnL bounds are reported.

Usage: ./venv313/bin/python maker_report.py [logfile]
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
WINNER_CACHE = os.path.join(HERE, "winner_cache.json")
TRADES_CACHE = os.path.join(HERE, "maker_trades_cache.json")
QUOTE_RE = re.compile(
    r"MAKER would-quote epoch=(\d+) side=(UP|DOWN) qty=([\d.]+) @([\d.]+) "
    r"window=\[t\+90,t\+(\d+)\]")
CANCEL_RE = re.compile(r"MAKER would-cancel epoch=(\d+) at t\+(\d+)s")
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


def http_get(url, params):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.ok:
                time.sleep(0.12)
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def epoch_prints(epoch: int, cache: dict):
    """UP-equivalent (rel_ts, price) prints in rel window [85, 295]."""
    key = str(epoch)
    if key in cache:
        return cache[key]
    evs = http_get(f"{GAMMA}/events", {"slug": f"btc-updown-5m-{epoch}"})
    cond = evs[0]["markets"][0].get("conditionId") if evs else None
    out = []
    if cond:
        for offset in range(0, 4000, 500):
            trades = http_get(f"{DATA_API}/trades",
                              {"market": cond, "limit": 500, "offset": offset})
            if not trades:
                break
            min_rel = None
            for t in trades:
                try:
                    rel = float(t["timestamp"]) - epoch
                    px = float(t["price"])
                    oc = (t.get("outcome") or "").upper()
                except (KeyError, ValueError, TypeError):
                    continue
                min_rel = rel if min_rel is None else min(min_rel, rel)
                if 85 <= rel <= 295 and oc in ("UP", "DOWN"):
                    up_px = px if oc == "UP" else round(1 - px, 4)
                    out.append([rel, up_px])
            if (min_rel is not None and min_rel < 85) or len(trades) < 500:
                break
    if out:  # never cache empty results — network failures would poison the cache
        cache[key] = out
    return out


def get_winner(epoch: int, wcache: dict):
    key = str(epoch)
    if key in wcache:
        return wcache[key]
    winner = None
    evs = http_get(f"{GAMMA}/events", {"slug": f"btc-updown-5m-{epoch}"})
    cond = evs[0]["markets"][0].get("conditionId") if evs else None
    if cond:
        data = http_get(f"https://clob.polymarket.com/markets/{cond}", {})
        for t in (data or {}).get("tokens", []):
            if t.get("winner"):
                winner = "UP" if (t.get("outcome") or "").upper() in ("YES", "UP") else "DOWN"
    if winner is not None:
        wcache[key] = winner
    return winner


def main():
    logs = sorted(
        (os.path.join(HERE, "logs", f) for f in os.listdir(os.path.join(HERE, "logs"))
         if f.startswith("naut_canary_")), key=os.path.getmtime)
    path = sys.argv[1] if len(sys.argv) > 1 else logs[-1]
    print(f"log: {path}")

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    quotes, cancels = [], {}
    for line in open(path, errors="replace"):
        clean = ansi.sub("", line)
        m = QUOTE_RE.search(clean)
        if m:
            quotes.append({"epoch": int(m.group(1)), "side": m.group(2),
                           "qty": float(m.group(3)), "price": float(m.group(4)),
                           "end": int(m.group(5))})
        m2 = CANCEL_RE.search(clean)
        if m2:
            e = int(m2.group(1))
            cancels[e] = min(cancels.get(e, 999), int(m2.group(2)))
    print(f"quotes: {len(quotes)} | cancels: {len(cancels)}")

    wcache = json.load(open(WINNER_CACHE)) if os.path.exists(WINNER_CACHE) else {}
    tcache = json.load(open(TRADES_CACHE)) if os.path.exists(TRADES_CACHE) else {}

    rows = []
    stake = 10.0
    for i, q in enumerate(quotes):
        if i % 30 == 0:
            json.dump(tcache, open(TRADES_CACHE, "w"))
            json.dump(wcache, open(WINNER_CACHE, "w"))
            print(f"  resolving {i}/{len(quotes)}...", flush=True)
        winner = get_winner(q["epoch"], wcache)
        prints = epoch_prints(q["epoch"], tcache)
        if winner is None:
            continue
        end = min(q["end"], cancels.get(q["epoch"], 999))
        # our bid in UP-equivalent terms
        bid_up_equiv = q["price"] if q["side"] == "UP" else round(1 - q["price"], 4)
        window = [p for rel, p in prints if 90 <= rel <= end]
        if q["side"] == "UP":
            through = any(p < bid_up_equiv for p in window)
            at = any(p <= bid_up_equiv for p in window)
        else:  # buying DOWN at price q means UP-equivalent selling above 1-q
            through = any(p > bid_up_equiv for p in window)
            at = any(p >= bid_up_equiv for p in window)
        win = q["side"] == winner
        pnl = stake * (1 - q["price"]) / q["price"] if win else -stake
        rows.append({**q, "win": win, "pnl": pnl, "through": through, "at": at,
                     "cancelled": q["epoch"] in cancels})
    json.dump(tcache, open(TRADES_CACHE, "w"))
    json.dump(wcache, open(WINNER_CACHE, "w"))

    n = len(rows)
    if n == 0:
        print("no resolvable quotes")
        return
    for bound, key in (("CONSERVATIVE (traded through)", "through"),
                       ("OPTIMISTIC (traded at)", "at")):
        fills = [r for r in rows if r[key]]
        nf = len(fills)
        print(f"\n=== {bound} ===")
        if not nf:
            print("  no fills")
            continue
        w = sum(r["win"] for r in fills)
        pnl = sum(r["pnl"] for r in fills)
        print(f"  fill rate {nf}/{n} ({nf/n*100:.0f}%) | WR {w/nf*100:.1f}% | "
              f"PnL ${pnl:+.2f} on ${nf*stake:.0f} filled (${pnl/max(nf*stake,1)*100:+.1f}%/$)")
        for side in ("UP", "DOWN"):
            g = [r for r in fills if r["side"] == side]
            if g:
                gw = sum(r["win"] for r in g)
                print(f"    {side}: n={len(g)} WR {gw/len(g)*100:.1f}% "
                      f"avg px {sum(r['price'] for r in g)/len(g):.3f} "
                      f"PnL ${sum(r['pnl'] for r in g):+.2f}")
        cx = [r for r in fills if r["cancelled"]]
        print(f"    fills on later-cancelled quotes: {len(cx)} "
              f"(PnL ${sum(r['pnl'] for r in cx):+.2f})")
    nc = sum(1 for r in rows if r["cancelled"])
    print(f"\nadverse-cancels: {nc}/{n} quotes")


if __name__ == "__main__":
    main()
