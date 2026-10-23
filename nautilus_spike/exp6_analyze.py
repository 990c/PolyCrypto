"""Exp 6 analysis: model vs market vs blends at t+90.

All fitting (blend weight, stack coefficients) happens on the CAL region;
the TEST region is scored exactly once per candidate.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 0.08
MIN_EV = 0.03
STAKE = 10.0
EPS = 1e-6


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def simulate(p_star, c_market, y):
    """Production gates: conviction on p*, EV against the real market price."""
    trades = []
    for pi, ci, yi in zip(p_star, c_market, y):
        if abs(pi - 0.5) < GATE:
            continue
        ev_up = pi * (1 - ci) - (1 - pi) * ci
        c_dn = 1 - ci
        ev_dn = (1 - pi) * (1 - c_dn) - pi * c_dn
        side, ev, cost = ("UP", ev_up, ci) if ev_up >= ev_dn else ("DOWN", ev_dn, c_dn)
        if ev < MIN_EV or not (0.05 <= cost <= 0.95):
            continue
        win = (yi == 1) if side == "UP" else (yi == 0)
        trades.append(STAKE * (1 - cost) / cost if win else -STAKE)
    nt = len(trades)
    pnl = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    return {"n": nt, "wr": wins / nt if nt else 0.0, "pnl": pnl,
            "roi": pnl / (nt * STAKE) if nt else 0.0}


def main():
    rows = json.load(open(os.path.join(HERE, "exp6_joined.json")))
    cal = [r for r in rows if r["region"] == "cal"]
    test = [r for r in rows if r["region"] == "test"]
    print(f"cal: {len(cal)} | test: {len(test)}")

    def arrays(rs):
        return (np.array([r["prob_up"] for r in rs]),
                np.array([r["price_up_t90"] for r in rs]),
                np.array([r["label"] for r in rs]))

    p_cal, c_cal, y_cal = arrays(cal)
    p_te, c_te, y_te = arrays(test)

    # --- fit on CAL ---
    best_w, best_ll = None, np.inf
    for w in np.arange(0.0, 1.01, 0.05):
        ll = logloss(w * p_cal + (1 - w) * c_cal, y_cal)
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    print(f"linear blend fit on cal: w_model={best_w:.2f} (cal logloss {best_ll:.4f})")

    from sklearn.linear_model import LogisticRegression
    stack = LogisticRegression(C=10.0, max_iter=1000)
    stack.fit(np.column_stack([logit(p_cal), logit(c_cal)]), y_cal)
    print(f"logit stack fit on cal: coef model={stack.coef_[0][0]:.3f} "
          f"market={stack.coef_[0][1]:.3f} intercept={stack.intercept_[0]:.3f}")

    # --- evaluate ONCE on TEST ---
    candidates = {
        "model only": p_te,
        "market only": c_te,
        f"blend w={best_w:.2f}": best_w * p_te + (1 - best_w) * c_te,
        "logit stack": stack.predict_proba(np.column_stack([logit(p_te), logit(c_te)]))[:, 1],
    }
    print(f"\n{'candidate':<18} {'logloss':>8} {'trades':>7} {'WR':>7} {'PnL':>8} {'ROI/$':>7}")
    for name, p_star in candidates.items():
        ll = logloss(p_star, y_te)
        sim = simulate(p_star, c_te, y_te)
        print(f"{name:<18} {ll:>8.4f} {sim['n']:>7} {sim['wr'] * 100:>6.1f}% "
              f"{sim['pnl']:>+8.0f} {sim['roi'] * 100:>+6.1f}%")


if __name__ == "__main__":
    main()
