"""Phase 3 analysis: model vs market at t+90s, and EV-gated strategy PnL at REAL prices."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 0.08
MIN_EV = 0.03
STAKE = 10.0


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    rows = json.load(open(os.path.join(HERE, "phase3_joined.json")))
    n = len(rows)
    p = np.array([r["prob_up"] for r in rows])
    c = np.array([r["price_up_t90"] for r in rows])
    y = np.array([r["label"] for r in rows])
    print(f"joined epochs: {n} | UP rate: {y.mean():.4f}")

    eps = 1e-6
    ll_market = -np.mean(y * np.log(np.clip(c, eps, 1)) + (1 - y) * np.log(np.clip(1 - c, eps, 1)))
    ll_model = -np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    print(f"\nlog-loss as UP-predictor:  market@t90 {ll_market:.4f}  |  model {ll_model:.4f}")

    print("\nmarket efficiency at t+90 (price bucket -> actual UP rate):")
    for lo_b in np.arange(0.0, 1.0, 0.2):
        m = (c >= lo_b) & (c < lo_b + 0.2)
        if m.sum() >= 10:
            print(f"  price [{lo_b:.1f},{lo_b + 0.2:.1f}): n={int(m.sum()):4d} "
                  f"actual UP {y[m].mean():.3f} vs implied {c[m].mean():.3f}")

    print("\nmodel vs market disagreement value:")
    dis = p - c
    for lo_b, hi_b in [(-1, -0.15), (-0.15, -0.05), (-0.05, 0.05), (0.05, 0.15), (0.15, 1)]:
        m = (dis >= lo_b) & (dis < hi_b)
        if m.sum() >= 10:
            print(f"  model-market [{lo_b:+.2f},{hi_b:+.2f}): n={int(m.sum()):4d} "
                  f"actual UP {y[m].mean():.3f} | model {p[m].mean():.3f} | market {c[m].mean():.3f}")

    # ---- EV-gated strategy at real prices ----
    print(f"\nstrategy sim (gate {GATE}, min EV {MIN_EV}, ${STAKE} flat stakes, real t+90 prices):")
    trades = []
    for pi, ci, yi in zip(p, c, y):
        if abs(pi - 0.5) < GATE:
            continue
        ev_up = pi * (1 - ci) - (1 - pi) * ci
        c_dn = 1 - ci
        ev_dn = (1 - pi) * (1 - c_dn) - pi * c_dn
        side, ev, cost = ("UP", ev_up, ci) if ev_up >= ev_dn else ("DOWN", ev_dn, c_dn)
        if ev < MIN_EV or not (0.05 <= cost <= 0.95):
            continue
        win = (yi == 1) if side == "UP" else (yi == 0)
        pnl = STAKE * (1 - cost) / cost if win else -STAKE
        trades.append({"side": side, "cost": cost, "win": win, "pnl": pnl, "ev": ev})

    if not trades:
        print("  no trades passed the gates")
        return
    nt = len(trades)
    w = sum(t["win"] for t in trades)
    pnl = sum(t["pnl"] for t in trades)
    lo95, hi95 = wilson(w, nt)
    print(f"  trades: {nt}/{n} epochs ({nt / n * 100:.0f}%) | WR {w / nt * 100:.1f}% "
          f"[{lo95 * 100:.1f},{hi95 * 100:.1f}] | PnL ${pnl:+.0f} on ${nt * STAKE:.0f} staked "
          f"({pnl / (nt * STAKE) * 100:+.1f}%/$)")
    for side in ("UP", "DOWN"):
        g = [t for t in trades if t["side"] == side]
        if g:
            gw = sum(t["win"] for t in g)
            print(f"    {side}: n={len(g)} WR {gw / len(g) * 100:.1f}% "
                  f"avg cost {np.mean([t['cost'] for t in g]):.3f} "
                  f"PnL ${sum(t['pnl'] for t in g):+.0f}")
    print("\n  by entry cost:")
    for lo_b, hi_b in [(0.05, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.95)]:
        g = [t for t in trades if lo_b <= t["cost"] < hi_b]
        if g:
            gw = sum(t["win"] for t in g)
            print(f"    cost [{lo_b:.2f},{hi_b:.2f}): n={len(g):3d} WR {gw / len(g) * 100:5.1f}% "
                  f"(breakeven {np.mean([t['cost'] for t in g]) * 100:.0f}%) "
                  f"PnL ${sum(t['pnl'] for t in g):+.0f}")


if __name__ == "__main__":
    main()
