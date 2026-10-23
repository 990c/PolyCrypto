"""Determine the exact post-TWAP label definition empirically.

Candidates (per the ambiguous rules text), tested against actual Polymarket
resolutions for Aug 7 (first day under TWAP rules):
  A  old:        close >= open
  B  twap-vs-open: mean(px[270..300)) >= open print
  C  twap-vs-trailing: mean(px[270..300)) >= mean(px[-30..0))  (prev epoch tail)

Prices from the Binance global archive (proxy for Chainlink aggregate).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
WCACHE = os.path.join(REPO_ROOT, "nautilus_spike/winner_cache.json")
EPOCH = 300


def get_winner(es: int, cache: dict):
    k = str(es)
    if k in cache:
        return cache[k]
    winner = None
    try:
        r = requests.get("https://gamma-api.polymarket.com/events",
                         params={"slug": f"btc-updown-5m-{es}"}, timeout=10)
        evs = r.json() if r.ok else []
        cond = evs[0]["markets"][0].get("conditionId") if evs else None
        if cond:
            r2 = requests.get(f"https://clob.polymarket.com/markets/{cond}", timeout=10)
            for t in (r2.json() if r2.ok else {}).get("tokens", []):
                if t.get("winner"):
                    winner = "UP" if (t.get("outcome") or "").upper() in ("YES", "UP") else "DOWN"
        time.sleep(0.1)
    except requests.RequestException:
        pass
    if winner is not None:
        cache[k] = winner
    return winner


def main():
    z = os.path.join(HERE, "archive", "BTCUSDT-aggTrades-2026-08-07.zip")
    probe = pd.read_csv(z, nrows=1, header=None)
    has_header = not str(probe.iloc[0, 0]).lstrip("-").isdigit()
    df = pd.read_csv(z, header=0 if has_header else None, usecols=[1, 5])
    df.columns = ["p", "t"]
    unit = 1e6 if df["t"].iloc[0] > 1e14 else 1e3
    secs = (df["t"] / unit).astype("int64")
    grid = pd.DataFrame({"s": secs, "p": df["p"]}).groupby("s")["p"].last().to_dict()
    lo, hi = min(grid), max(grid)
    print(f"grid: {len(grid)} seconds")

    def px_range(a, b):
        v = [grid[s] for s in range(a, b) if s in grid]
        return v

    cache = json.load(open(WCACHE)) if os.path.exists(WCACHE) else {}
    stats = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}
    n = 0
    start = lo - (lo % EPOCH) + EPOCH
    es = start
    while es + EPOCH <= hi:
        full = px_range(es, es + EPOCH)
        tail = px_range(es + EPOCH - 30, es + EPOCH)
        prev_tail = px_range(es - 30, es)
        if len(full) < 200 or len(tail) < 15 or len(prev_tail) < 15:
            es += EPOCH
            continue
        w = get_winner(es, cache)
        if w is None:
            es += EPOCH
            continue
        actual = 1 if w == "UP" else 0
        o, c = full[0], full[-1]
        labels = {
            "A": 1 if c >= o else 0,
            "B": 1 if np.mean(tail) >= o else 0,
            "C": 1 if np.mean(tail) >= np.mean(prev_tail) else 0,
        }
        for k, v in labels.items():
            stats[k][0] += int(v == actual)
            stats[k][1] += 1
        n += 1
        if n % 40 == 0:
            json.dump(cache, open(WCACHE, "w"))
            print(f"  {n} epochs tested...", flush=True)
        es += EPOCH
    json.dump(cache, open(WCACHE, "w"))

    print(f"\nepochs tested: {n} (Aug 7, first day under TWAP rules)")
    for k, name in (("A", "close >= open (old rule)"),
                    ("B", "TWAP(last30) >= open print"),
                    ("C", "TWAP(last30) >= TWAP(trailing 30 at start)")):
        m, t = stats[k]
        print(f"  {k} {name:<42}: {m}/{t} = {m/t*100:.1f}%")


if __name__ == "__main__":
    main()
