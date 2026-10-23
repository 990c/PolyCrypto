"""Is elite-wallet positioning a FOLLOWABLE signal?

Elite set is defined on period 1 ONLY. Second pass over trades.csv accumulates,
for each period-2 epoch, the elite cohort's net Up-equivalent position built by
t+90 and t+150. Test: does it predict the outcome BEYOND the t+90 market price?
"""
import json, os, sys, time
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY_DATA_DIR = os.environ.get("POLY_DATA_DIR", os.path.join(REPO_ROOT, "..", "poly_data"))

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(POLY_DATA_DIR, "processed/trades.csv")
MAP = json.load(open(f"{HERE}/market_map.json"))
ep_of = {k: v[0] for k, v in MAP.items()}
win_of = {k: v[1] for k, v in MAP.items() if v[1]}
eps_all = sorted(set(ep_of.values())); SPLIT = eps_all[len(eps_all)//2]

W = pd.read_csv(f"{HERE}/wallets.csv")
W["roi1"] = W.p1_pnl / W.p1_vol.replace(0, np.nan)
elite = set(W[(W.roi1 > 0.02) & (W.p1_vol >= 20000) & (W.p1_n >= 200)].w)
# "followable" subset: decent ROI, EARLY traders, not mega-volume MMs
foll = set(W[(W.roi1 > 0.03) & (W.p1_vol >= 5000) & (W.p1_n >= 100)
             & (W.p1_n <= 20000) & (W.mean_rel <= 150)].w)
print(f"elite p1 cohort: {len(elite)} | followable subset: {len(foll)}")
json.dump({"elite": sorted(elite), "foll": sorted(foll)}, open(f"{HERE}/cohorts.json", "w"))

COLS = ["timestamp","market_id","maker","taker","nonusdc_side","taker_direction","price","token_amount"]
p2_markets = {k for k, v in MAP.items() if v[1] and v[0] >= SPLIT}
acc = {}   # epoch -> [elite_90, elite_150, foll_90, foll_150]
if os.path.exists(f"{HERE}/acc_ckpt.json"):
    acc = {int(k): v for k, v in json.load(open(f"{HERE}/acc_ckpt.json")).items()}
    print(f"resumed checkpoint: {len(acc)} epochs")
nchunk = 0
t0 = time.time(); nrow = 0
for chunk in pd.read_csv(SRC, usecols=COLS, chunksize=4_000_000, low_memory=False):
    chunk = chunk[chunk.market_id.isin(p2_markets)]
    if chunk.empty:
        continue
    in_e = chunk.taker.isin(elite) | chunk.maker.isin(elite)
    in_f = chunk.taker.isin(foll) | chunk.maker.isin(foll)
    chunk = chunk[in_e | in_f]
    if chunk.empty:
        continue
    ts = pd.to_datetime(chunk.timestamp, errors="coerce").astype("int64")//10**9
    ep = chunk.market_id.map(ep_of).astype("int64")
    rel = (ts - ep).astype(float).values
    is_up = (chunk.nonusdc_side == "token1").values
    buy_t = (chunk.taker_direction == "BUY").values
    tok = chunk.token_amount.astype(float).values
    # taker's Up-equivalent signed position; maker takes the opposite
    sgn_t = np.where(is_up == buy_t, 1.0, -1.0) * tok
    te = chunk.taker.isin(elite).values; me = chunk.maker.isin(elite).values
    tf = chunk.taker.isin(foll).values;  mf = chunk.maker.isin(foll).values
    e_sig = np.where(te, sgn_t, 0.0) + np.where(me, -sgn_t, 0.0)
    f_sig = np.where(tf, sgn_t, 0.0) + np.where(mf, -sgn_t, 0.0)
    df = pd.DataFrame({"e": ep.values, "rel": rel, "es": e_sig, "fs": f_sig})
    for lim, i0, i1 in ((90, 0, 2), (150, 1, 3)):
        w = df[(df.rel >= 0) & (df.rel <= lim)]
        if w.empty:
            continue
        for e, v in w.groupby("e")["es"].sum().items():
            acc.setdefault(e, [0.0]*4)[i0] += v
        for e, v in w.groupby("e")["fs"].sum().items():
            acc.setdefault(e, [0.0]*4)[i1] += v
    nrow += len(chunk)
    nchunk += 1
    if nchunk % 5 == 0:
        json.dump({str(k): v for k, v in acc.items()}, open(f"{HERE}/acc_ckpt.json", "w"))
        print(f"  {nrow/1e6:.1f}M cohort rows | {len(acc)} epochs | "
              f"{time.time()-t0:.0f}s [ckpt]", flush=True)

px90 = {int(k): v for k, v in json.load(open(f"{HERE}/px90.json")).items()}
inv = {v[0]: k for k, v in MAP.items()}
rows = []
for e, a in acc.items():
    p = px90.get(e)
    cid = inv.get(e)
    if not p or not cid or not win_of.get(cid):
        continue
    rows.append({"e": e, "px90": p[1], "y": 1 if win_of[cid] == "UP" else 0,
                 "e90": a[0], "e150": a[1], "f90": a[2], "f150": a[3]})
d = pd.DataFrame(rows)
d.to_csv(f"{HERE}/elite_signal.csv", index=False)
print(f"\nperiod-2 epochs with data: {len(d)} | {time.time()-t0:.0f}s")
