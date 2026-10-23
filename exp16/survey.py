"""Exp 16: cross-market efficiency survey.

For each family, sample recent resolved markets and measure the SAME yardstick
we calibrated on BTC 5m (Exp 15): favorite-longshot bias at a comparable point
in each market's life (70% through), plus liquidity and trade counts.
"""
import json, os, sys, time
from collections import defaultdict

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CACHE = os.path.join(HERE, "survey_cache.json")
REF_FRAC = 0.70


def get(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.4 * (i + 1))
    return None


def sample_family(asset, horizon, secs, n_markets, cache):
    key = f"{asset}-{horizon}"
    rows = cache.setdefault(key, {})
    now = int(time.time())
    base = now - (now % secs) - 2 * secs
    got = 0
    for i in range(n_markets * 3):
        if got >= n_markets:
            break
        es = base - i * secs
        if str(es) in rows:
            got += 1
            continue
        slug = f"{asset}-updown-{horizon}-{es}"
        evs = get(f"{GAMMA}/events", {"slug": slug})
        if not evs:
            continue
        m = evs[0].get("markets", [{}])[0]
        cond = m.get("conditionId")
        if not cond:
            continue
        md = get(f"{CLOB}/markets/{cond}", {})
        winner = None
        for t in (md or {}).get("tokens", []):
            if t.get("winner"):
                winner = "UP" if (t.get("outcome") or "").upper() in ("YES", "UP") else "DOWN"
        if winner is None:
            continue
        prints, vol = [], 0.0
        for off in range(0, 3000, 500):
            tr = get(f"{DATA}/trades", {"market": cond, "limit": 500, "offset": off})
            if not tr:
                break
            oldest = None
            for t in tr:
                try:
                    rel = float(t["timestamp"]) - es
                    p = float(t["price"]); oc = (t.get("outcome") or "").upper()
                    sz = float(t["size"])
                except (KeyError, ValueError, TypeError):
                    continue
                oldest = rel if oldest is None else min(oldest, rel)
                if 0 <= rel <= secs:
                    prints.append((rel, p if oc == "UP" else round(1 - p, 4)))
                    vol += sz * p
            if len(tr) < 500 or (oldest is not None and oldest < 0):
                break
        if len(prints) < 5:
            continue
        ref_t = secs * REF_FRAC
        before = [p for r, p in prints if r <= ref_t]
        if not before:
            continue
        rows[str(es)] = {"y": 1 if winner == "UP" else 0, "px": before[-1],
                         "n": len(prints), "vol": round(vol, 1)}
        got += 1
        if got % 40 == 0:
            json.dump(cache, open(CACHE, "w"))
            print(f"    {key}: {got}/{n_markets}", flush=True)
    json.dump(cache, open(CACHE, "w"))
    return rows


def bias_table(rows, label):
    d = [(v["px"], v["y"]) for v in rows.values() if 0.005 < v["px"] < 0.995]
    if len(d) < 100:
        print(f"  {label}: only {len(d)} usable markets — skipped")
        return None
    px = np.array([x[0] for x in d]); y = np.array([x[1] for x in d])
    vols = np.array([v["vol"] for v in rows.values()])
    ntr = np.array([v["n"] for v in rows.values()])
    # favourite-side bias: fold to the favourite's perspective
    fav_p = np.where(px >= 0.5, px, 1 - px)
    fav_win = np.where(px >= 0.5, y == 1, y == 0)
    out = {"n": len(d), "med_vol": float(np.median(vols)), "med_trades": float(np.median(ntr))}
    print(f"\n  {label}  (n={len(d)}, median vol ${np.median(vols):,.0f}, "
          f"median trades {np.median(ntr):.0f})")
    print(f"    {'fav price':>12} {'n':>6} {'actual':>8} {'implied':>8} {'bias':>8}")
    for lo, hi in [(0.5,0.65),(0.65,0.8),(0.8,0.92),(0.92,1.0)]:
        m = (fav_p >= lo) & (fav_p < hi)
        if m.sum() < 25:
            continue
        a = fav_win[m].mean(); im = fav_p[m].mean()
        print(f"    {lo:.2f}-{hi:.2f}  {int(m.sum()):>6} {a*100:>7.1f}% {im*100:>7.1f}% "
              f"{(a-im)*100:>+7.1f}pp")
        out[f"bias_{lo}"] = float(a - im)
    m = fav_p >= 0.8
    if m.sum() >= 25:
        out["bias_high"] = float(fav_win[m].mean() - fav_p[m].mean())
    return out


if __name__ == "__main__":
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    fams = [("btc","15m",900),("eth","5m",300),("sol","5m",300),("xrp","5m",300)]
    res = {}
    for a, h, s in fams:
        print(f"\nsampling {a}-{h} ...", flush=True)
        rows = sample_family(a, h, s, N, cache)
        res[f"{a}-{h}"] = bias_table(rows, f"{a.upper()} {h}")
    json.dump(res, open(os.path.join(HERE, "survey_results.json"), "w"), indent=1)
