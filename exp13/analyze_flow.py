"""Exp 13 analysis: does Polymarket's own order flow contain a tradeable edge?

A. Diagnostic — does the API report one side or both sides of each fill?
B. Wallet edge persistence: identify on period 1, validate OUT-OF-SAMPLE on period 2.
C. Flow signal beyond price: does net flow add information over the t+90 price?
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DECIDE_T = 90.0


def load():
    done = json.load(open(os.path.join(HERE, "flow_done.json")))
    winners = {int(k): v for k, v in done.items() if v in ("UP", "DOWN")}
    trades = defaultdict(list)
    with open(os.path.join(HERE, "flow.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["e"] in winners:
                trades[r["e"]].append(r)
    return winners, trades


def up_price(r):
    return r["p"] if r["o"] == "UP" else 1.0 - r["p"]


def up_signed_size(r):
    """+size = net UP demand, -size = net DOWN demand."""
    buy = r["s"] == "BUY"
    up = r["o"] == "UP"
    return r["sz"] if (buy == up) else -r["sz"]


def trade_pnl(r, winner):
    """Mark-to-resolution PnL of this trade for the wallet that made it."""
    won = 1.0 if r["o"] == winner else 0.0
    if r["s"] == "BUY":
        return r["sz"] * (won - r["p"])
    return r["sz"] * (r["p"] - won)


def main():
    winners, trades = load()
    epochs = sorted(trades)
    print(f"epochs: {len(epochs)} | trades: {sum(len(v) for v in trades.values())}")

    # ---- A. one-sided or two-sided feed? ----
    net = [sum(up_signed_size(r) for r in trades[e]) for e in epochs]
    tot = [sum(r["sz"] for r in trades[e]) for e in epochs]
    ratio = np.mean([abs(n) / t for n, t in zip(net, tot) if t > 0])
    pnl_sum = np.mean([sum(trade_pnl(r, winners[e]) for r in trades[e]) for e in epochs])
    print(f"\nA. feed diagnostic: mean |net flow|/volume = {ratio:.3f} "
          f"| mean epoch PnL across all recorded trades = ${pnl_sum:+.1f}")
    print("   (≈0 net flow and ≈0 PnL ⇒ both sides recorded; large ⇒ taker-side only)")

    # ---- B. wallet edge persistence, out-of-sample ----
    half = len(epochs) // 2
    p1, p2 = set(epochs[:half]), set(epochs[half:])
    stats = defaultdict(lambda: {1: [0.0, 0.0, 0], 2: [0.0, 0.0, 0]})  # pnl, volume, trades
    for e in epochs:
        per = 1 if e in p1 else 2
        w_ = winners[e]
        for r in trades[e]:
            s = stats[r["w"]][per]
            s[0] += trade_pnl(r, w_)
            s[1] += r["sz"] * r["p"]
            s[2] += 1
    print(f"\nB. wallet persistence: {len(stats)} distinct wallets | "
          f"period1 {len(p1)} epochs, period2 {len(p2)} epochs")

    MIN_TRADES, MIN_VOL = 50, 2000.0
    cands = [(w, v) for w, v in stats.items()
             if v[1][2] >= MIN_TRADES and v[1][1] >= MIN_VOL and v[2][2] >= 20]
    print(f"   wallets active in BOTH periods (p1: ≥{MIN_TRADES} trades & ≥${MIN_VOL:.0f} vol, "
          f"p2: ≥20 trades): {len(cands)}")
    if cands:
        ranked = sorted(cands, key=lambda kv: -(kv[1][1][0] / max(kv[1][1][1], 1)))
        top = ranked[:max(5, len(ranked) // 10)]
        bot = ranked[-max(5, len(ranked) // 10):]
        for label, grp in (("TOP decile by period-1 ROI", top), ("BOTTOM decile", bot)):
            r1 = sum(v[1][0] for _, v in grp) / max(sum(v[1][1] for _, v in grp), 1)
            r2 = sum(v[2][0] for _, v in grp) / max(sum(v[2][1] for _, v in grp), 1)
            print(f"   {label}: n={len(grp)} | period1 ROI {r1*100:+.2f}% "
                  f"→ period2 ROI (OUT-OF-SAMPLE) {r2*100:+.2f}%")
        roi1 = np.array([v[1][0] / max(v[1][1], 1) for _, v in cands])
        roi2 = np.array([v[2][0] / max(v[2][1], 1) for _, v in cands])
        if len(roi1) > 3:
            print(f"   correlation of wallet ROI period1 vs period2: {np.corrcoef(roi1, roi2)[0,1]:+.3f}")

    # ---- C. flow beyond price ----
    rows = []
    for e in epochs:
        pre = [r for r in trades[e] if 0 <= r["t"] <= DECIDE_T]
        if len(pre) < 20:
            continue
        px = up_price(sorted(pre, key=lambda r: r["t"])[-1])
        if not (0.02 < px < 0.98):
            continue
        flow = sum(up_signed_size(r) for r in pre)
        vol = sum(r["sz"] for r in pre)
        big = sum(up_signed_size(r) for r in pre if r["sz"] * r["p"] >= 100)
        rows.append((px, flow / max(vol, 1), big / max(vol, 1), 1 if winners[e] == "UP" else 0))
    if len(rows) < 100:
        print(f"\nC. insufficient epochs ({len(rows)}) for the flow test")
        return
    arr = np.array(rows)
    px, fl, bg, y = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    print(f"\nC. flow-beyond-price test: n={len(rows)} epochs (decision at t+{DECIDE_T:.0f}s)")
    print(f"   market price alone: log-loss "
          f"{-np.mean(y*np.log(np.clip(px,1e-6,1))+(1-y)*np.log(np.clip(1-px,1e-6,1))):.4f}")
    from sklearn.linear_model import LogisticRegression
    lo = np.log(np.clip(px, 1e-6, 1 - 1e-6) / (1 - np.clip(px, 1e-6, 1 - 1e-6)))
    for name, X in (("price only", lo.reshape(-1, 1)),
                    ("price + netflow", np.column_stack([lo, fl])),
                    ("price + netflow + bigflow", np.column_stack([lo, fl, bg]))):
        h = len(y) // 2
        m = LogisticRegression(max_iter=1000).fit(X[:h], y[:h])
        p = m.predict_proba(X[h:])[:, 1]
        ll = -np.mean(y[h:]*np.log(np.clip(p,1e-6,1)) + (1-y[h:])*np.log(np.clip(1-p,1e-6,1)))
        coef = " ".join(f"{c:+.3f}" for c in m.coef_[0])
        print(f"   {name:<26} out-of-sample log-loss {ll:.4f} | coefs [{coef}]")
    # does flow predict the residual at all?
    resid = y - px
    print(f"   corr(netflow, outcome−price) = {np.corrcoef(fl, resid)[0,1]:+.3f} | "
          f"corr(bigflow, outcome−price) = {np.corrcoef(bg, resid)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
