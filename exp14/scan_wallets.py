"""Per-address PnL across the full 172-day 5m-BTC history (both maker & taker roles).

Binary markets with minting make net flow structurally zero, so the only
testable edge in this data is participant skill. For each address we track,
per period: mark-to-resolution PnL, notional volume, trade count, epochs
touched, and mean trade time within the epoch (for followability).
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY_DATA_DIR = os.environ.get("POLY_DATA_DIR", os.path.join(REPO_ROOT, "..", "poly_data"))

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(POLY_DATA_DIR, "processed/trades.csv")
MAP = json.load(open(os.path.join(HERE, "market_map.json")))
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
COLS = ["timestamp", "market_id", "maker", "taker", "nonusdc_side",
        "taker_direction", "price", "token_amount"]

ep_of = {k: v[0] for k, v in MAP.items()}
win_of = {k: v[1] for k, v in MAP.items() if v[1]}
eps_all = sorted(set(ep_of.values()))
SPLIT = eps_all[len(eps_all) // 2]          # period boundary (median epoch)
print(f"period split at {pd.to_datetime(SPLIT, unit='s')}")


def main():
    t0 = time.time()
    # wallet -> [p1_pnl, p1_vol, p1_n, p2_pnl, p2_vol, p2_n, sum_rel, n_rel]
    acc = defaultdict(lambda: [0.0] * 8)
    ep_seen = defaultdict(lambda: [set(), set()])
    px90 = {}
    n_rows = 0

    for chunk in pd.read_csv(SRC, usecols=COLS, chunksize=2_000_000,
                             nrows=NROWS or None, low_memory=False):
        chunk = chunk[chunk.market_id.isin(win_of)]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk.timestamp, errors="coerce").astype("int64") // 10**9
        ep = chunk.market_id.map(ep_of).astype("int64")
        rel = (ts - ep).astype(float)
        is_up = (chunk.nonusdc_side == "token1").values
        buy = (chunk.taker_direction == "BUY").values
        px = chunk.price.astype(float).values
        tok = chunk.token_amount.astype(float).values
        wins_up = (chunk.market_id.map(win_of) == "UP").values
        tok_wins = np.where(is_up, wins_up, ~wins_up).astype(float)
        tpnl = np.where(buy, tok * (tok_wins - px), tok * (px - tok_wins))
        notional = tok * px
        per2 = (ep.values >= SPLIT)
        keep = (rel.values >= -30) & (rel.values <= 320)

        for w, p, n, r, k, sec in zip(chunk.taker.values, tpnl, notional,
                                      rel.values, keep, per2):
            if not k:
                continue
            a = acc[w]
            o = 3 if sec else 0
            a[o] += p; a[o + 1] += n; a[o + 2] += 1
            a[6] += r; a[7] += 1
        for w, p, n, r, k, sec in zip(chunk.maker.values, tpnl, notional,
                                      rel.values, keep, per2):
            if not k:
                continue
            a = acc[w]
            o = 3 if sec else 0
            a[o] -= p; a[o + 1] += n; a[o + 2] += 1
            a[6] += r; a[7] += 1
        for w, e, sec, k in zip(chunk.taker.values, ep.values, per2, keep):
            if k:
                ep_seen[w][1 if sec else 0].add(int(e))
        for w, e, sec, k in zip(chunk.maker.values, ep.values, per2, keep):
            if k:
                ep_seen[w][1 if sec else 0].add(int(e))

        pre = (rel.values >= 0) & (rel.values <= 90)
        if pre.any():
            up_px = np.where(is_up, px, 1.0 - px)
            df = pd.DataFrame({"e": ep.values[pre], "r": rel.values[pre],
                               "p": up_px[pre]})
            idx = df.groupby("e")["r"].idxmax()
            for e, r, p in zip(df.loc[idx, "e"], df.loc[idx, "r"], df.loc[idx, "p"]):
                cur = px90.get(e)
                if cur is None or r >= cur[0]:
                    px90[e] = (r, p)
        n_rows += len(chunk)
        if n_rows % 20_000_000 < 2_000_000:
            print(f"  {n_rows/1e6:.0f}M rows | {len(acc)} wallets | "
                  f"{time.time()-t0:.0f}s", flush=True)

    rows = []
    for w, a in acc.items():
        rows.append({"w": w, "p1_pnl": a[0], "p1_vol": a[1], "p1_n": int(a[2]),
                     "p2_pnl": a[3], "p2_vol": a[4], "p2_n": int(a[5]),
                     "mean_rel": a[6] / max(a[7], 1),
                     "p1_eps": len(ep_seen[w][0]), "p2_eps": len(ep_seen[w][1])})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "wallets.csv"), index=False)
    pd.DataFrame([{"epoch": e, "px90_t": v[0], "px90": v[1],
                   "winner": win_of.get(next((k for k, x in ep_of.items() if x == e), ""), None)}
                  for e, v in list(px90.items())[:0]]).to_csv(
        os.path.join(HERE, "_unused.csv"), index=False)
    json.dump({str(k): [round(v[0], 1), round(v[1], 4)] for k, v in px90.items()},
              open(os.path.join(HERE, "px90.json"), "w"))
    print(f"done: {len(rows)} wallets, {len(px90)} epochs priced, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
