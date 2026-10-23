"""Weather pilot: does a leak-free day-ahead forecast beat the market's implied
distribution on Polymarket daily high-temperature markets?

Decision time = 24h before market close. Forecast uses ONLY previous_day1
(what the model said a day earlier) — no analysis data, no look-ahead.
"""
import datetime as dt
import json, os, sys, time
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "pilot_cache.json")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MONTHS = {7:"july",8:"august",9:"september"}
CITIES = {
 "nyc":(40.78,-73.97,"America/New_York"), "london":(51.48,-0.45,"Europe/London"),
 "moscow":(55.97,37.41,"Europe/Moscow"),  "madrid":(40.47,-3.56,"Europe/Madrid"),
 "paris":(49.01,2.55,"Europe/Paris"),     "tokyo":(35.55,139.78,"Asia/Tokyo"),
 "seoul":(37.57,126.98,"Asia/Seoul"),     "beijing":(40.08,116.58,"Asia/Shanghai"),
 "los-angeles":(33.94,-118.41,"America/Los_Angeles"),
 "chicago":(41.98,-87.90,"America/Chicago"),
 "miami":(25.79,-80.29,"America/New_York"),
 "toronto":(43.68,-79.63,"America/Toronto"),
}

def get(url, params, tries=3, timeout=25):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.4*(i+1))
    return None

def parse_bucket(label):
    """Returns (lo, hi, unit) in the market's own unit.
    '67F or below'->(-inf,67,'F'); '72-73F'->(72,73,'F');
    '29C or below'->(-inf,29,'C'); '33C'->(33,33,'C'); '39C or higher'->(39,inf,'C')"""
    t = str(label).strip()
    unit = "F" if "F" in t.upper() and "C" not in t.upper().replace("°C","") else None
    if unit is None:
        unit = "F" if "°F" in t or t.upper().endswith("F") else "C"
    s2 = t.replace("°F", "").replace("°C", "").replace("°", "").replace("F", "").replace("C", "").strip()
    low = t.lower()
    try:
        if "below" in low:
            return (-999.0, float(s2.lower().split("or")[0].strip()), unit)
        if "higher" in low or "above" in low:
            return (float(s2.lower().split("or")[0].strip()), 999.0, unit)
        if "-" in s2:
            a, b = s2.split("-")[:2]
            return (float(a.strip()), float(b.strip()), unit)
        v = float(s2)
        return (v, v, unit)
    except ValueError:
        return None

def event_data(city, d, cache):
    key = f"{city}|{d}"
    if key in cache:
        return cache[key]
    slug = f"highest-temperature-in-{city}-on-{MONTHS.get(d.month,'august')}-{d.day}-{d.year}"
    evs = get(f"{GAMMA}/events", {"slug": slug})
    if not evs:
        cache[key] = None; return None
    e = evs[0]
    if not e.get("closed"):
        return None
    out = {"buckets": [], "end": e.get("endDate")}
    for m in e.get("markets", []):
        lab = m.get("groupItemTitle") or m.get("question")
        rng = parse_bucket(lab)
        cond = m.get("conditionId")
        if rng is None or not cond:
            continue
        md = get(f"{CLOB}/markets/{cond}", {})
        win, tok = None, None
        for t in (md or {}).get("tokens", []):
            if (t.get("outcome") or "").upper() == "YES":
                tok = t.get("token_id"); win = bool(t.get("winner"))
        if tok is None:
            continue
        out["buckets"].append({"lab": lab, "lo": rng[0], "hi": rng[1], "unit": rng[2],
                               "tok": tok, "win": bool(win)})
    if len(out["buckets"]) < 8 or not any(b["win"] for b in out["buckets"]):
        cache[key] = None; return None
    # price 24h before close for each bucket
    end_ts = int(dt.datetime.fromisoformat(out["end"].replace("Z","+00:00")).timestamp())
    dec = end_ts - 86400
    for b in out["buckets"]:
        ph = get(f"{CLOB}/prices-history",
                 {"market": b["tok"], "startTs": dec-7200, "endTs": dec, "fidelity": 10})
        pts = (ph or {}).get("history") or []
        b["px"] = float(pts[-1]["p"]) if pts else None
    cache[key] = out
    return out

def forecast_dist(city, d, buckets, bias=0.0, sd=1.0):
    lat, lon, tz = CITIES[city]
    j = get("https://previous-runs-api.open-meteo.com/v1/forecast",
            {"latitude":lat,"longitude":lon,"timezone":tz,
             "start_date":str(d),"end_date":str(d),
             "hourly":"temperature_2m_previous_day1"})
    if not j:
        return None, None
    vals = [v for v in j["hourly"].get("temperature_2m_previous_day1",[]) if v is not None]
    if not vals:
        return None, None
    point = max(vals) + bias
    probs = []
    for b in buckets:
        lo, hi = b["lo"], b["hi"]
        a = (lo - 0.5 - point)/sd if lo > -90 else -50
        z = (hi + 0.5 - point)/sd if hi < 90 else 50
        from math import erf, sqrt
        cdf = lambda x: 0.5*(1+erf(x/sqrt(2)))
        probs.append(max(cdf(z)-cdf(a), 1e-6))
    p = np.array(probs); p = p/p.sum()
    return p, point

if __name__ == "__main__":
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    today = dt.date.today()
    rows = []
    for city in CITIES:
        for k in range(2, 2+ndays):
            d = today - dt.timedelta(days=k)
            ev = event_data(city, d, cache)
            if not ev:
                continue
            bs = ev["buckets"]
            if any(b.get("px") is None for b in bs):
                continue
            mp = np.array([b["px"] for b in bs], dtype=float)
            if mp.sum() <= 0:
                continue
            mp = np.clip(mp, 1e-4, None); mp = mp/mp.sum()
            wi = [i for i,b in enumerate(bs) if b["win"]]
            if len(wi) != 1:
                continue
            rows.append({"city":city,"date":str(d),"win":wi[0],
                         "mkt":mp.tolist(),
                         "labs":[b["lab"] for b in bs],
                         "unit":bs[0]["unit"],
                         "los":[b["lo"] for b in bs],"his":[b["hi"] for b in bs]})
        json.dump(cache, open(CACHE,"w"))
        print(f"  {city}: {len(rows)} events so far", flush=True)
    json.dump(rows, open(os.path.join(HERE,"events.json"),"w"))
    print(f"collected {len(rows)} resolved events with prices")
