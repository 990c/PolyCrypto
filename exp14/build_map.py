"""Build condition_id -> (epoch_start, winner, up_is_token1) from poly_data markets.csv."""
import os
import json, re
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY_DATA_DIR = os.environ.get("POLY_DATA_DIR", os.path.join(REPO_ROOT, "..", "poly_data"))

MK = os.path.join(POLY_DATA_DIR, "data/markets.csv")
OUT = os.path.join(REPO_ROOT, "exp14/market_map.json")

df = pd.read_csv(MK, usecols=["condition_id", "market_slug", "tokens"], low_memory=False)
b = df[df.market_slug.astype(str).str.startswith("btc-updown-5m", na=False)].copy()
b["epoch"] = b.market_slug.str.extract(r"btc-updown-5m-(\d+)")[0].astype("Int64")

out = {}
bad = 0
for cid, ep, tk in zip(b.condition_id, b.epoch, b.tokens):
    try:
        toks = json.loads(tk)
        if len(toks) != 2 or pd.isna(ep):
            bad += 1; continue
        outcomes = [(t.get("outcome") or "").upper() for t in toks]
        winners = [bool(t.get("winner")) for t in toks]
        if "UP" not in outcomes:
            bad += 1; continue
        up_idx = outcomes.index("UP")
        w = None
        if winners[0] != winners[1]:
            w = "UP" if winners[up_idx] else "DOWN"
        out[cid] = [int(ep), w, 1 if up_idx == 0 else 2]
    except (json.JSONDecodeError, TypeError, ValueError):
        bad += 1
json.dump(out, open(OUT, "w"))
res = sum(1 for v in out.values() if v[1])
up1 = sum(1 for v in out.values() if v[2] == 1)
print(f"markets mapped: {len(out)} (bad {bad}) | resolved: {res} | Up==token1 in {up1} ({up1/len(out)*100:.1f}%)")
eps = [v[0] for v in out.values()]
print(f"epoch span: {pd.to_datetime(min(eps),unit='s')} -> {pd.to_datetime(max(eps),unit='s')}")
