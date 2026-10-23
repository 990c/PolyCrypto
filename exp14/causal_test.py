"""Causal (leak-free) test: at each horizon T, compare price_T vs price_T + elite_T.

Both measured at the SAME instant, so nothing from the future enters.
"""
import json, os, time
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY_DATA_DIR = os.environ.get("POLY_DATA_DIR", os.path.join(REPO_ROOT, "..", "poly_data"))

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(POLY_DATA_DIR, "processed/trades.csv")
MAP = json.load(open(f"{HERE}/market_map.json"))
ep_of = {k: v[0] for k, v in MAP.items()}
win_of = {k: v[1] for k, v in MAP.items() if v[1]}
eps = sorted(set(ep_of.values())); SPLIT = eps[len(eps)//2]
elite = set(json.load(open(f"{HERE}/cohorts.json"))["elite"])
p2 = {k for k, v in MAP.items() if v[1] and v[0] >= SPLIT}
HOR = [90, 150, 210]
COLS = ["timestamp","market_id","maker","taker","nonusdc_side","taker_direction","price","token_amount"]

pos = {}   # epoch -> {T: elite net}
prc = {}   # epoch -> {T: (rel, up_px)}
t0 = time.time(); n = 0
for chunk in pd.read_csv(SRC, usecols=COLS, chunksize=4_000_000, low_memory=False):
    chunk = chunk[chunk.market_id.isin(p2)]
    if chunk.empty:
        continue
    ts = pd.to_datetime(chunk.timestamp, errors="coerce").astype("int64")//10**9
    ep = chunk.market_id.map(ep_of).astype("int64").values
    rel = (ts.values - ep).astype(float)
    is_up = (chunk.nonusdc_side == "token1").values
    buy = (chunk.taker_direction == "BUY").values
    px = chunk.price.astype(float).values
    up_px = np.where(is_up, px, 1.0 - px)
    tok = chunk.token_amount.astype(float).values
    sgn = np.where(is_up == buy, 1.0, -1.0) * tok
    te = chunk.taker.isin(elite).values; me = chunk.maker.isin(elite).values
    esig = np.where(te, sgn, 0.0) + np.where(me, -sgn, 0.0)
    df = pd.DataFrame({"e": ep, "rel": rel, "up": up_px, "es": esig})
    for T in HOR:
        w = df[(df.rel >= 0) & (df.rel <= T)]
        if w.empty:
            continue
        for e, v in w.groupby("e")["es"].sum().items():
            pos.setdefault(e, {}).setdefault(T, 0.0)
            pos[e][T] += v
        idx = w.groupby("e")["rel"].idxmax()
        for e, r, p in zip(w.loc[idx, "e"], w.loc[idx, "rel"], w.loc[idx, "up"]):
            cur = prc.setdefault(e, {}).get(T)
            if cur is None or r >= cur[0]:
                prc[e][T] = (r, p)
    n += len(chunk)
print(f"scan done: {len(pos)} epochs | {time.time()-t0:.0f}s")

inv = {v[0]: k for k, v in MAP.items()}
rows = []
for e, ph in prc.items():
    cid = inv.get(e)
    if not cid or not win_of.get(cid):
        continue
    r = {"e": e, "y": 1 if win_of[cid] == "UP" else 0}
    ok = True
    for T in HOR:
        if T not in ph or T not in pos.get(e, {}):
            ok = False; break
        r[f"px{T}"] = ph[T][1]
        r[f"es{T}"] = pos[e][T]
    if ok:
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(f"{HERE}/causal.csv", index=False)

from sklearn.linear_model import LogisticRegression
def ll(y, p):
    p = np.clip(p, 1e-6, 1-1e-6); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
print(f"\nepochs usable: {len(d):,}")
print(f"{'horizon':>8} {'price alone':>13} {'price+elite':>13} {'Δ log-loss':>12} {'corr(elite,resid)':>19}")
for T in HOR:
    dd = d[(d[f"px{T}"] > 0.02) & (d[f"px{T}"] < 0.98)]
    y = dd.y.values; px = dd[f"px{T}"].values
    s = dd[f"es{T}"].values
    sn = np.sign(s)*np.log1p(np.abs(s))/10.0
    lo = np.log(px/(1-px))
    h = len(dd)//2
    m1 = LogisticRegression(max_iter=1000).fit(lo[:h].reshape(-1,1), y[:h])
    p1 = m1.predict_proba(lo[h:].reshape(-1,1))[:,1]
    X = np.column_stack([lo, sn])
    m2 = LogisticRegression(max_iter=1000).fit(X[:h], y[:h])
    p2_ = m2.predict_proba(X[h:])[:,1]
    c = np.corrcoef(sn, y - px)[0,1]
    print(f"{T:>8} {ll(y[h:],p1):>13.4f} {ll(y[h:],p2_):>13.4f} "
          f"{ll(y[h:],p2_)-ll(y[h:],p1):>+12.4f} {c:>+19.4f}")
