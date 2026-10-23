"""Stream poly_data trades.csv (60GB) -> per-epoch flow features + per-wallet PnL.

Taker = aggressor (has intent). Maker PnL is the mirror image.
Up is always token1 (validated). Up-equivalent price/flow conventions:
  taker BUY token1  -> +Up demand   |  taker SELL token1 -> -Up demand
  taker BUY token2  -> -Up demand   |  taker SELL token2 -> +Up demand
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
WINDOWS = [(0, 90), (90, 180), (180, 300)]


def main():
    t0 = time.time()
    ep_of = {k: v[0] for k, v in MAP.items()}
    win_of = {k: v[1] for k, v in MAP.items() if v[1]}
    # per-epoch accumulators
    flow = defaultdict(lambda: defaultdict(float))
    last_px = {}          # (epoch, window_end) -> (t, up_price)
    tak = defaultdict(lambda: [0.0, 0.0, 0])   # wallet -> pnl, vol, n
    mak = defaultdict(lambda: [0.0, 0.0, 0])
    tak_ep = defaultdict(set)
    n_rows = 0

    reader = pd.read_csv(SRC, usecols=COLS, chunksize=2_000_000,
                         nrows=NROWS or None, low_memory=False)
    for chunk in reader:
        chunk = chunk[chunk.market_id.isin(win_of)]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk.timestamp, errors="coerce").astype("int64") // 10**9
        ep = chunk.market_id.map(ep_of)
        rel = (ts - ep).astype(float)
        is_up_tok = (chunk.nonusdc_side == "token1")
        buy = (chunk.taker_direction == "BUY")
        px = chunk.price.astype(float)
        tokamt = chunk.token_amount.astype(float)
        up_px = np.where(is_up_tok, px, 1.0 - px)
        # taker's Up-equivalent demand
        sign = np.where(is_up_tok == buy, 1.0, -1.0)
        signed = sign * tokamt
        wins_up = chunk.market_id.map(win_of) == "UP"
        tok_wins = np.where(is_up_tok, wins_up, ~wins_up).astype(float)
        tpnl = np.where(buy, tokamt * (tok_wins - px), tokamt * (px - tok_wins))
        notional = tokamt * px

        d = pd.DataFrame({"e": ep.values, "rel": rel.values, "up_px": up_px,
                          "signed": signed, "notional": notional.values,
                          "tpnl": tpnl, "mk": chunk.maker.values,
                          "tk": chunk.taker.values, "tok": tokamt.values})
        d = d[(d.rel >= -30) & (d.rel <= 320)]
        if d.empty:
            continue
        for lo, hi in WINDOWS:
            w = d[(d.rel >= lo) & (d.rel < hi)]
            if w.empty:
                continue
            g = w.groupby("e")
            for e, v in g["signed"].sum().items():
                flow[e][f"f{lo}_{hi}"] += v
            for e, v in g["notional"].sum().items():
                flow[e][f"v{lo}_{hi}"] += v
            for e, v in g["tok"].sum().items():
                flow[e][f"q{lo}_{hi}"] += v
            big = w[w.notional >= 100]
            if not big.empty:
                for e, v in big.groupby("e")["signed"].sum().items():
                    flow[e][f"b{lo}_{hi}"] += v
        # decision price: latest trade at rel<=90
        pre = d[(d.rel >= 0) & (d.rel <= 90)]
        if not pre.empty:
            idx = pre.groupby("e")["rel"].idxmax()
            for e, r in zip(pre.loc[idx, "e"], pre.loc[idx].itertuples()):
                cur = last_px.get(e)
                if cur is None or r.rel >= cur[0]:
                    last_px[e] = (r.rel, r.up_px)
        # wallet aggregates
        for w_, p_, n_ in zip(d.tk.values, d.tpnl.values, d.notional.values):
            s = tak[w_]; s[0] += p_; s[1] += n_; s[2] += 1
        for w_, p_, n_ in zip(d.mk.values, d.tpnl.values, d.notional.values):
            s = mak[w_]; s[0] -= p_; s[1] += n_; s[2] += 1
        for w_, e_ in zip(d.tk.values, d.e.values):
            tak_ep[w_].add(int(e_))
        n_rows += len(chunk)
        print(f"  {n_rows/1e6:.1f}M matched rows | {len(flow)} epochs | "
              f"{len(tak)} takers | {time.time()-t0:.0f}s", flush=True)

    # write outputs
    rows = []
    for e, f in flow.items():
        r = {"epoch": e, "winner": win_of.get(e if isinstance(e, str) else e)}
        r.update(f)
        px = last_px.get(e)
        r["px90"] = px[1] if px else np.nan
        r["px90_t"] = px[0] if px else np.nan
        rows.append(r)
    fdf = pd.DataFrame(rows)
    inv = {v[0]: k for k, v in MAP.items()}
    fdf["winner"] = fdf.epoch.map(lambda e: MAP.get(inv.get(e, ""), [None, None])[1])
    fdf.to_csv(os.path.join(HERE, "epoch_flow.csv"), index=False)
    wdf = pd.DataFrame([{"w": w, "pnl": v[0], "vol": v[1], "n": v[2],
                         "n_epochs": len(tak_ep.get(w, ()))} for w, v in tak.items()])
    wdf.to_csv(os.path.join(HERE, "taker_wallets.csv"), index=False)
    mdf = pd.DataFrame([{"w": w, "pnl": v[0], "vol": v[1], "n": v[2]} for w, v in mak.items()])
    mdf.to_csv(os.path.join(HERE, "maker_wallets.csv"), index=False)
    json.dump({str(k): sorted(v) for k, v in tak_ep.items() if len(v) >= 20},
              open(os.path.join(HERE, "taker_epochs.json"), "w"))
    print(f"done: {len(fdf)} epochs, {len(wdf)} takers, {len(mdf)} makers, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
