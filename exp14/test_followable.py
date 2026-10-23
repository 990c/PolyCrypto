"""Decisive test: does elite-cohort positioning predict beyond the market price?

Cohort defined on period 1 ONLY. Within period 2, fit any combination weights on
the FIRST half and score once on the SECOND half.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(f"{HERE}/elite_signal.csv").sort_values("e").reset_index(drop=True)
d = d[(d.px90 > 0.02) & (d.px90 < 0.98)].dropna(subset=["px90", "y"])
print(f"period-2 epochs: {len(d):,} | UP rate {d.y.mean():.3f}")

def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

# scale-free versions of the cohort signal
for c in ["e90", "e150", "f90", "f150"]:
    s = d[c].values
    d[c + "_n"] = np.sign(s) * np.log1p(np.abs(s)) / 10.0
y = d.y.values
px = d.px90.values
lo = np.log(px / (1 - px))
print(f"\nbaseline  price@t+90 alone: log-loss {ll(y, px):.4f}")

print("\ncorrelation with residual (outcome − price):")
resid = y - px
for c in ["e90", "e150", "f90", "f150"]:
    print(f"   {c:>5}: corr {np.corrcoef(d[c + '_n'], resid)[0,1]:+.4f}")

h = len(d) // 2
feats = {
    "price only": [lo],
    "price + elite@90": [lo, d.e90_n.values],
    "price + elite@90,150": [lo, d.e90_n.values, d.e150_n.values],
    "price + followable@90": [lo, d.f90_n.values],
    "price + all cohort sigs": [lo, d.e90_n.values, d.e150_n.values,
                                d.f90_n.values, d.f150_n.values],
}
print(f"\nfit on first {h:,} period-2 epochs, score on last {len(d)-h:,}:")
base = None
for name, cols in feats.items():
    X = np.column_stack(cols)
    m = LogisticRegression(max_iter=1000).fit(X[:h], y[:h])
    p = m.predict_proba(X[h:])[:, 1]
    L = ll(y[h:], p)
    if base is None:
        base = L
    coefs = " ".join(f"{c:+.3f}" for c in m.coef_[0])
    print(f"   {name:<24} log-loss {L:.4f}  (Δ vs price {L-base:+.4f})  coefs [{coefs}]")

# economic view: does the signal move the probability enough to beat a ~2c spread?
best = np.column_stack([lo, d.e90_n.values, d.e150_n.values])
m = LogisticRegression(max_iter=1000).fit(best[:h], y[:h])
p = m.predict_proba(best[h:])[:, 1]
edge = p - px[h:]
print(f"\nmodel-vs-market probability shift on holdout: "
      f"mean |shift| {np.mean(np.abs(edge))*100:.2f}c | p95 {np.percentile(np.abs(edge),95)*100:.2f}c")
big = np.abs(edge) > 0.03
if big.sum() > 30:
    side_up = edge[big] > 0
    yy = y[h:][big]
    wr = np.mean(np.where(side_up, yy == 1, yy == 0))
    cost = np.mean(np.where(side_up, px[h:][big], 1 - px[h:][big]))
    print(f"   epochs with >3c disagreement: n={int(big.sum())} | win rate {wr*100:.1f}% "
          f"| avg cost {cost:.3f} (breakeven {cost*100:.1f}%) → edge {(wr-cost)*100:+.1f}pp")
else:
    print(f"   epochs with >3c disagreement: only {int(big.sum())} — signal never moves the price materially")
