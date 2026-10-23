"""Weather pilot analysis: market-implied vs leak-free day-ahead forecast.

Both sides observe the same information cutoff (24h before close).
Step 1 validates the station offset empirically (gridpoint != NOAA station);
Step 2 fits the forecast->bucket model on a TRAIN half and scores the HELD-OUT
half exactly once.
"""
import json, os
from math import erf, sqrt
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CITIES = {
 "nyc":(40.78,-73.97,"America/New_York"),"london":(51.48,-0.45,"Europe/London"),
 "moscow":(55.97,37.41,"Europe/Moscow"),"madrid":(40.47,-3.56,"Europe/Madrid"),
 "paris":(49.01,2.55,"Europe/Paris"),"tokyo":(35.55,139.78,"Asia/Tokyo"),
 "seoul":(37.57,126.98,"Asia/Seoul"),"beijing":(40.08,116.58,"Asia/Shanghai"),
 "los-angeles":(33.94,-118.41,"America/Los_Angeles"),"chicago":(41.98,-87.90,"America/Chicago"),
 "miami":(25.79,-80.29,"America/New_York"),"toronto":(43.68,-79.63,"America/Toronto")}
cdf = lambda x: 0.5*(1+erf(x/sqrt(2)))

def fc_cache():
    p = os.path.join(HERE,"fc_cache.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def get_fc(city, date, cache):
    k = f"{city}|{date}"
    if k in cache: return cache[k]
    la,lo,tz = CITIES[city]
    try:
        j = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast",
            params={"latitude":la,"longitude":lo,"timezone":tz,"start_date":date,"end_date":date,
                    "hourly":"temperature_2m_previous_day1"}, timeout=25).json()
        v = [x for x in j["hourly"].get("temperature_2m_previous_day1",[]) if x is not None]
        cache[k] = max(v) if v else None
    except Exception:
        cache[k] = None
    return cache[k]

def dist(point_c, los, his, unit, sd):
    """Bucket probabilities from N(point, sd). Bounds are in the market's unit."""
    if unit == "F":
        point = point_c*9/5+32
        s = sd*9/5
        half = 0.5
    else:
        point, s, half = point_c, sd, 0.5
    p = []
    for lo, hi in zip(los, his):
        a = -50 if lo < -900 else (lo-half-point)/s
        b = 50 if hi > 900 else (hi+half-point)/s
        p.append(max(cdf(b)-cdf(a), 1e-9))
    p = np.array(p); return p/p.sum()

def main():
    rows = json.load(open(os.path.join(HERE,"events.json")))
    cache = fc_cache()
    for r in rows:
        r["fc"] = get_fc(r["city"], r["date"], cache)
    json.dump(cache, open(os.path.join(HERE,"fc_cache.json"),"w"))
    rows = [r for r in rows if r.get("fc") is not None]
    print(f"events with market prices + forecast: {len(rows)}")
    if len(rows) < 60:
        print("too few for a verdict"); return

    # ---- Step 1: station offset per city (forecast -> winning bucket agreement) ----
    print("\nstation-offset calibration (per city, on TRAIN half):")
    rows.sort(key=lambda r: r["date"])
    h = len(rows)//2
    tr, te = rows[:h], rows[h:]
    off = {}
    for c in CITIES:
        g = [r for r in tr if r["city"]==c]
        if len(g) < 5: continue
        best, bo = -1, 0.0
        for cand in np.arange(-2.0, 2.01, 0.25):
            hit = 0
            for r in g:
                p = dist(r["fc"]+cand, r["los"], r["his"], r["unit"], 0.9)
                hit += int(int(np.argmax(p)) == r["win"])
            if hit > best: best, bo = hit, float(cand)
        off[c] = bo
        print(f"   {c:<13} offset {bo:+.2f}°C  (train modal-hit {best}/{len(g)})")

    # ---- Step 2: fit sd on train, score test once ----
    def ll(rs, sd, use_fc):
        tot = 0.0
        for r in rs:
            if use_fc:
                p = dist(r["fc"]+off.get(r["city"],0.0), r["los"], r["his"], r["unit"], sd)
            else:
                p = np.array(r["mkt"])
            tot += -np.log(max(p[r["win"]], 1e-9))
        return tot/len(rs)
    best_sd = min(np.arange(0.5, 3.01, 0.1), key=lambda s: ll(tr, s, True))
    print(f"\nfitted forecast sd on train: {best_sd:.1f}°C")
    m_ll, f_ll = ll(te, 0, False), ll(te, best_sd, True)
    print(f"\nHELD-OUT ({len(te)} events):")
    print(f"   market-implied  log-loss {m_ll:.4f}")
    print(f"   forecast        log-loss {f_ll:.4f}   (Δ {f_ll-m_ll:+.4f})")
    mh = np.mean([int(np.argmax(r["mkt"])==r["win"]) for r in te])
    fh = np.mean([int(np.argmax(dist(r["fc"]+off.get(r["city"],0.0),r["los"],r["his"],
                                     r["unit"],best_sd))==r["win"]) for r in te])
    print(f"   modal-bucket hit: market {mh*100:.1f}%  vs forecast {fh*100:.1f}%")

    # ---- economic view: biggest disagreements ----
    print("\n   largest edges (forecast prob - market prob on same bucket):")
    edges = []
    for r in te:
        p = dist(r["fc"]+off.get(r["city"],0.0), r["los"], r["his"], r["unit"], best_sd)
        mk = np.array(r["mkt"])
        for i in range(len(p)):
            edges.append((p[i]-mk[i], mk[i], int(i==r["win"])))
    edges.sort(key=lambda x: -x[0])
    for lo_e, hi_e, lab in ((0.15,9,">15pp"),(0.08,0.15,"8-15pp"),(0.03,0.08,"3-8pp")):
        g = [e for e in edges if lo_e <= e[0] < hi_e]
        if len(g) >= 20:
            wr = np.mean([e[2] for e in g]); cost = np.mean([e[1] for e in g])
            print(f"     {lab:<7} n={len(g):>4} | market px {cost:.3f} | actual {wr:.3f} "
                  f"| EV/$1 {(wr*(1-cost)-(1-wr)*cost)/max(cost,1e-6):+.3f}")

if __name__ == "__main__":
    main()
