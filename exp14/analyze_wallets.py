"""Exp 14: is there persistent, followable participant skill in 5m BTC markets?

Identify on period 1 (Feb 12 - May 8), validate OUT-OF-SAMPLE on period 2
(May 8 - Aug 2). Never evaluate on the identification sample.
"""
import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W = pd.read_csv(os.path.join(REPO_ROOT, "exp14/wallets.csv"))
print(f"wallets: {len(W):,} | total trades: {(W.p1_n+W.p2_n).sum():,.0f}")
print(f"aggregate PnL (zero-sum check): ${(W.p1_pnl+W.p2_pnl).sum():+,.0f}")

both = W[(W.p1_n >= 50) & (W.p2_n >= 50) & (W.p1_vol >= 1000) & (W.p2_vol >= 1000)].copy()
both["roi1"] = both.p1_pnl / both.p1_vol
both["roi2"] = both.p2_pnl / both.p2_vol
print(f"\nactive in BOTH periods (>=50 trades & >=$1k vol each): {len(both):,}")
print(f"  period1 volume ${both.p1_vol.sum():,.0f} | period2 ${both.p2_vol.sum():,.0f}")

r = np.corrcoef(both.roi1, both.roi2)[0, 1]
rs = both[["roi1", "roi2"]].corr(method="spearman").iloc[0, 1]
print(f"\nPERSISTENCE: corr(ROI p1, ROI p2) = {r:+.4f} | spearman {rs:+.4f}")

q = both.copy()
q["dec"] = pd.qcut(q.roi1, 10, labels=False, duplicates="drop")
print(f"\n{'p1 decile':>9} {'n':>6} {'p1 ROI':>9} {'p2 ROI (OOS)':>13} {'p2 PnL':>12}")
for d in sorted(q.dec.dropna().unique()):
    g = q[q.dec == d]
    print(f"{int(d):>9} {len(g):>6} {g.p1_pnl.sum()/g.p1_vol.sum()*100:>8.2f}% "
          f"{g.p2_pnl.sum()/g.p2_vol.sum()*100:>12.2f}% ${g.p2_pnl.sum():>11,.0f}")

# strictest test: wallets that were BOTH profitable and high-volume in p1
elite = both[(both.roi1 > 0.02) & (both.p1_vol >= 20000) & (both.p1_n >= 200)]
print(f"\nELITE p1 filter (ROI>2%, vol>=$20k, >=200 trades): n={len(elite)}")
if len(elite):
    print(f"  p1: ROI {elite.p1_pnl.sum()/elite.p1_vol.sum()*100:+.2f}% (${elite.p1_pnl.sum():+,.0f})")
    print(f"  p2 OUT-OF-SAMPLE: ROI {elite.p2_pnl.sum()/elite.p2_vol.sum()*100:+.2f}% "
          f"(${elite.p2_pnl.sum():+,.0f})")
    wins = (elite.p2_pnl > 0).mean()
    print(f"  fraction still profitable in p2: {wins*100:.1f}% (chance = ~50%)")
    boot = [np.random.default_rng(i).choice(elite.p2_pnl.values, len(elite)).sum()
            for i in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap 95% CI on p2 PnL: [${lo:+,.0f}, ${hi:+,.0f}]")
    print(f"\n  top 10 elite wallets by p1 ROI -> their p2 result:")
    for _, x in elite.nlargest(10, "roi1").iterrows():
        print(f"    {x.w[:12]}.. p1 ROI {x.roi1*100:+7.2f}% (${x.p1_pnl:+9,.0f}) -> "
              f"p2 ROI {x.roi2*100:+7.2f}% (${x.p2_pnl:+9,.0f}) | mean t+{x.mean_rel:.0f}s")

# who actually makes money overall, and is it followable?
tot = W.assign(pnl=W.p1_pnl + W.p2_pnl, vol=W.p1_vol + W.p2_vol, n=W.p1_n + W.p2_n)
big = tot[(tot.n >= 500) & (tot.vol >= 50000)].nlargest(12, "pnl")
print(f"\nTOP 12 ALL-TIME EARNERS (>=500 trades, >=$50k volume):")
print(f"  {'wallet':>14} {'PnL':>12} {'ROI':>8} {'trades':>8} {'mean t':>8}")
for _, x in big.iterrows():
    print(f"  {x.w[:12]+'..':>14} ${x.pnl:>10,.0f} {x.pnl/x.vol*100:>7.2f}% "
          f"{int(x.n):>8,} t+{x.mean_rel:>4.0f}s")
