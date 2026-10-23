"""Exp 8 Step 2: build the multi-year epoch dataset from all archive months.

Memory-bounded: one month's second-grid at a time (~2.6M entries); the trailing
partial epoch's seconds carry into the next month. Epoch aggregates (small)
accumulate across the whole span, then build_rows() runs once at the end.
"""
import csv
import glob
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from price_features import EPOCH_SECONDS, PRICE_FEATURE_NAMES, build_rows, epoch_aggregate

OUT = os.path.join(HERE, "epochs_multiyear.csv")


def month_grid(zip_path: str) -> dict:
    probe = pd.read_csv(zip_path, nrows=1, header=None)
    has_header = not str(probe.iloc[0, 0]).lstrip("-").isdigit()
    grid: dict = {}
    for chunk in pd.read_csv(zip_path, header=0 if has_header else None,
                             usecols=[1, 5], chunksize=10_000_000):
        chunk.columns = ["p", "t"]
        ts = chunk["t"].to_numpy()
        unit = 1e6 if ts[0] > 1e14 else 1e3
        secs = (ts / unit).astype("int64")
        g = pd.DataFrame({"s": secs, "p": chunk["p"].to_numpy()}).groupby("s")["p"].last()
        grid.update(g.to_dict())
    return grid


def main():
    t0 = time.time()
    zips = sorted(glob.glob(os.path.join(HERE, "archive", "BTCUSDT-aggTrades-20[0-9][0-9]-[0-9][0-9].zip")))
    print(f"{len(zips)} monthly zips")
    aggs: list = []
    carry: dict = {}
    for z in zips:
        grid = month_grid(z)
        grid.update(carry)
        lo, hi = min(grid), max(grid)
        start = lo - (lo % EPOCH_SECONDS)
        last_complete_start = (hi + 1) - ((hi + 1) % EPOCH_SECONDS) - EPOCH_SECONDS
        while start <= last_complete_start:
            a = epoch_aggregate(grid, start)
            if a is not None:
                aggs.append(a)
            start += EPOCH_SECONDS
        carry = {s: p for s, p in grid.items() if s >= last_complete_start + EPOCH_SECONDS}
        print(f"  {os.path.basename(z)}: total aggs={len(aggs)} | {time.time()-t0:.0f}s", flush=True)

    rows = build_rows(aggs)
    cols = (["epoch_start"] + PRICE_FEATURE_NAMES
            + ["start_price", "end_price", "target_resolution"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
