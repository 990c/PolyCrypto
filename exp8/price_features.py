"""Exp 8: price-pure epoch features, computable identically from
(a) Binance public aggTrades archive and (b) our collected dataset.csv.

Both sources are first reduced to a per-second last-price grid; everything
downstream sees only that grid, so train/serve semantics cannot diverge.
Scale-free by construction (returns/bps only — no volumes, no book state),
sidestepping the binance.com vs binance.us volume mismatch.
"""
from typing import Dict, List, Optional

import numpy as np

EPOCH_SECONDS = 300
CURR_WINDOW_SECONDS = 90
MIN_COVERAGE_SECONDS = 200  # epoch must have price data for >= this many seconds

PRICE_FEATURE_NAMES = [
    # previous epoch (t-1)
    "prev_ret_bps", "prev_range_bps", "prev_rv_bps", "prev_close_pos",
    # multi-epoch momentum/vol regime
    "ret_2ep", "ret_3ep", "ret_5ep",
    "up_ratio_3ep", "up_ratio_5ep", "streak_sign",
    "rv_mean_3ep", "rv_mean_5ep", "vol_expanding",
    "range_mean_3ep", "ret_accel", "prev_vs_2ep",
    # current epoch, first 90 seconds
    "curr_return_90s", "curr_range_90s", "curr_rv_90s", "curr_close_pos_90s",
]


def epoch_aggregate(grid: Dict[int, float], epoch_start: int) -> Optional[dict]:
    """Aggregate one epoch from the per-second grid. None if coverage too thin."""
    secs = [s for s in range(epoch_start, epoch_start + EPOCH_SECONDS) if s in grid]
    if len(secs) < MIN_COVERAGE_SECONDS:
        return None
    px = np.array([grid[s] for s in secs])
    o, c, hi, lo = px[0], px[-1], px.max(), px.min()
    if o <= 0:
        return None
    logret = np.diff(np.log(px))
    rng = hi - lo
    return {
        "epoch_start": epoch_start,
        "open": float(o), "close": float(c),
        "ret_bps": float((c - o) / o * 1e4),
        "range_bps": float(rng / o * 1e4),
        "rv_bps": float(np.std(logret) * 1e4),
        "close_pos": float((c - lo) / rng) if rng > 0 else 0.5,
        # trailing 30s TWAP at epoch end — the post-2026-08-07 resolution reference
        "twap_last30": float(np.mean([grid[s] for s in secs
                                      if s >= epoch_start + EPOCH_SECONDS - 30])),
        # first-90s slice (relative to epoch start, from the same grid)
        "_first90": [grid[s] for s in secs if s < epoch_start + CURR_WINDOW_SECONDS],
    }


def assemble_row(priors: List[dict], first90: List[float]) -> Optional[dict]:
    """The 20 price features from 5 prior-epoch aggregates (oldest first) and
    the current epoch's first-90s price series. Shared by the offline builders
    and the live strategy so feature semantics cannot drift."""
    if len(first90) < 60:  # need decent coverage of the decision window
        return None
    p90 = np.array(first90)
    o = p90[0]
    if o <= 0:
        return None
    hi90, lo90 = p90.max(), p90.min()
    rng90 = hi90 - lo90
    lr90 = np.diff(np.log(p90)) if len(p90) > 1 else np.array([0.0])

    rets = [p["ret_bps"] for p in priors]
    rvs = [p["rv_bps"] for p in priors]
    dirs = [1 if r > 0 else 0 for r in rets]
    streak = 0
    for d in reversed(dirs):
        if d == dirs[-1]:
            streak += 1
        else:
            break
    rv3 = float(np.mean(rvs[2:]))
    return {
        "prev_ret_bps": rets[4],
        "prev_range_bps": priors[4]["range_bps"],
        "prev_rv_bps": rvs[4],
        "prev_close_pos": priors[4]["close_pos"],
        "ret_2ep": rets[3] + rets[4],
        "ret_3ep": sum(rets[2:]),
        "ret_5ep": sum(rets),
        "up_ratio_3ep": sum(dirs[2:]) / 3.0,
        "up_ratio_5ep": sum(dirs) / 5.0,
        "streak_sign": streak if dirs[4] == 1 else -streak,
        "rv_mean_3ep": rv3,
        "rv_mean_5ep": float(np.mean(rvs)),
        "vol_expanding": 1.0 if rvs[4] > rv3 > 0 else 0.0,
        "range_mean_3ep": float(np.mean([p["range_bps"] for p in priors[2:]])),
        "ret_accel": rets[4] - rets[3],
        "prev_vs_2ep": rets[4] - (rets[3] + rets[4]),
        "curr_return_90s": float((p90[-1] - o) / o * 1e4),
        "curr_range_90s": float(rng90 / o * 1e4),
        "curr_rv_90s": float(np.std(lr90) * 1e4),
        "curr_close_pos_90s": float((p90[-1] - lo90) / rng90) if rng90 > 0 else 0.5,
    }


def build_rows(aggs: List[dict]) -> List[dict]:
    """Offline path: rows for every epoch with 5 contiguous priors."""
    by_start = {a["epoch_start"]: a for a in aggs}
    rows = []
    for a in aggs:
        es = a["epoch_start"]
        priors = [by_start.get(es - k * EPOCH_SECONDS) for k in range(5, 0, -1)]
        if any(p is None for p in priors):
            continue
        row = assemble_row(priors, a["_first90"])
        if row is None:
            continue
        row["epoch_start"] = es
        row["start_price"] = a["open"]
        row["end_price"] = a["close"]
        # TWAP resolution (Polymarket, effective 2026-08-07): last-30s TWAP vs
        # trailing-30s TWAP at range start (= previous epoch's tail); ties -> Up.
        # Empirically validated 94% raw / ~100% on decided epochs (EXPERIMENTS.md Exp 11).
        row["target_resolution"] = 1 if a["twap_last30"] >= priors[4]["twap_last30"] else 0
        rows.append(row)
    return rows
