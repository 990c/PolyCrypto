"""Exp 15: does Polymarket underprice near-certain outcomes late in the epoch?

At t+210 the TWAP reference (trailing 30s at epoch start) is known and BTC's
path so far is known; only ~90s of drift remains. Compare the market's t+210
price against the empirical outcome frequency conditional on BTC's position.
"""
import os
import glob, json
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HERE = REPO_ROOT
d = pd.read_csv(f"{HERE}/exp14/causal.csv")
print(f"poly epochs: {len(d):,}")

grid = {}
for z in sorted(glob.glob(f"{HERE}/exp8/archive/BTCUSDT-aggTrades-2026-0[56].zip")):
    probe = pd.read_csv(z, nrows=1, header=None)
    hh = 0 if not str(probe.iloc[0, 0]).lstrip("-").isdigit() else None
    for ch in pd.read_csv(z, header=hh, usecols=[1, 5], chunksize=8_000_000):
        ch.columns = ["p", "t"]
        u = 1e6 if ch["t"].iloc[0] > 1e14 else 1e3
        g = pd.DataFrame({"s": (ch["t"]/u).astype("int64"), "p": ch["p"]}).groupby("s")["p"].last()
        grid.update(g.to_dict())
print(f"btc grid seconds: {len(grid):,}")

rows = []
for e, y, px210 in zip(d.e, d.y, d.px210):
    ref = [grid[s] for s in range(e-30, e) if s in grid]
    cur = [grid[s] for s in range(e+200, e+211) if s in grid]
    if len(ref) < 15 or not cur:
        continue
    r = float(np.mean(ref)); c = cur[-1]
    rows.append({"e": e, "y": y, "px210": px210, "lead_bps": (c-r)/r*1e4})
m = pd.DataFrame(rows)
print(f"epochs with BTC coverage: {len(m):,}\n")

print("BTC lead at t+210 vs actual UP rate vs market price:")
print(f"{'lead (bps)':>16} {'n':>6} {'actual UP':>10} {'mkt px210':>10} {'gap':>8}")
bins = [(-99,-15),(-15,-8),(-8,-3),(-3,3),(3,8),(8,15),(15,99)]
for lo, hi in bins:
    g = m[(m.lead_bps >= lo) & (m.lead_bps < hi)]
    if len(g) < 50:
        continue
    print(f"{lo:>7}..{hi:<7} {len(g):>6} {g.y.mean()*100:>9.1f}% {g.px210.mean()*100:>9.1f}% "
          f"{(g.y.mean()-g.px210.mean())*100:>+7.1f}pp")

print("\nnear-certain cases (|lead| >= 15bps):")
for side, g in (("UP-certain", m[m.lead_bps >= 15]), ("DOWN-certain", m[m.lead_bps <= -15])):
    if len(g) < 30:
        continue
    if side == "UP-certain":
        wr, cost = g.y.mean(), g.px210.mean()
    else:
        wr, cost = 1 - g.y.mean(), 1 - g.px210.mean()
    ev = wr*(1-cost) - (1-wr)*cost
    print(f"  {side}: n={len(g):,} | win rate {wr*100:.1f}% | avg cost {cost:.3f} "
          f"| EV/$1 {ev:+.4f}")
