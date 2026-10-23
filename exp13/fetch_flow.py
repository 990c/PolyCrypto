"""Exp 13: fetch full Polymarket trade flow for btc-updown-5m epochs.

Writes JSONL (one record per trade) plus a done-set for resumability.
Each record: epoch, ts_rel, wallet, side, outcome, size, price.
"""
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "flow.jsonl")
DONE = os.path.join(HERE, "flow_done.json")
GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
EPOCH = 300


def http_get(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.5 * (i + 1))
    return None


def fetch_epoch(es, out_f):
    evs = http_get(f"{GAMMA}/events", {"slug": f"btc-updown-5m-{es}"})
    if not evs:
        return 0, None
    m = evs[0]["markets"][0]
    cond = m.get("conditionId")
    if not cond:
        return 0, None
    # winner from clob
    winner = None
    data = http_get(f"https://clob.polymarket.com/markets/{cond}", {})
    for t in (data or {}).get("tokens", []):
        if t.get("winner"):
            winner = "UP" if (t.get("outcome") or "").upper() in ("YES", "UP") else "DOWN"
    n = 0
    for offset in range(0, 8000, 500):
        trades = http_get(f"{DATA}/trades", {"market": cond, "limit": 500, "offset": offset})
        if not trades:
            break
        oldest = None
        for t in trades:
            try:
                ts = float(t["timestamp"])
                rec = {
                    "e": es, "t": round(ts - es, 1),
                    "w": t["proxyWallet"], "s": t["side"],
                    "o": (t.get("outcome") or "").upper(),
                    "sz": float(t["size"]), "p": float(t["price"]),
                }
            except (KeyError, ValueError, TypeError):
                continue
            oldest = ts if oldest is None else min(oldest, ts)
            if -60 <= rec["t"] <= 320:
                out_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
        if len(trades) < 500 or (oldest is not None and oldest < es - 60):
            break
        time.sleep(0.05)
    return n, winner


def main():
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    back = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    done = json.load(open(DONE)) if os.path.exists(DONE) else {}
    now = int(time.time())
    base = now - (now % EPOCH) - back * EPOCH
    t0 = time.time()
    total = 0
    with open(OUT, "a") as f:
        for i in range(n_epochs):
            es = base - i * EPOCH
            if str(es) in done:
                continue
            n, w = fetch_epoch(es, f)
            if n:
                done[str(es)] = w
                total += n
            if i % 25 == 0:
                json.dump(done, open(DONE, "w"))
                f.flush()
                print(f"  {i}/{n_epochs} epochs | {total} trades | {time.time()-t0:.0f}s", flush=True)
    json.dump(done, open(DONE, "w"))
    print(f"done: {len(done)} epochs cached, {total} new trades, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
