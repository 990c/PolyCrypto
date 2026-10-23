"""Exp 8: build price-pure epoch datasets from both sources.

  python build_datasets.py archive    -> epochs_archive.csv   (binance.com aggTrades)
  python build_datasets.py collected  -> epochs_collected.csv (our dataset.csv, binance.us)

Both paths reduce to a per-second last-price grid, then share price_features.
"""
import csv
import glob
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from price_features import PRICE_FEATURE_NAMES, build_rows, epoch_aggregate, EPOCH_SECONDS


def rows_to_csv(rows: list, path: str) -> None:
    cols = (["epoch_start"] + PRICE_FEATURE_NAMES
            + ["start_price", "end_price", "target_resolution"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} epoch rows -> {path}")


def grid_to_rows(grid: dict) -> list:
    if not grid:
        return []
    lo, hi = min(grid), max(grid)
    aggs = []
    start = lo - (lo % EPOCH_SECONDS)
    while start + EPOCH_SECONDS <= hi + 1:
        a = epoch_aggregate(grid, start)
        if a is not None:
            aggs.append(a)
        start += EPOCH_SECONDS
    return build_rows(aggs)


def build_archive() -> None:
    t0 = time.time()
    grid: dict = {}
    zips = sorted(glob.glob(os.path.join(HERE, "archive", "*.zip")))
    print(f"archive zips: {[os.path.basename(z) for z in zips]}")
    for z in zips:
        # sniff header row
        probe = pd.read_csv(z, nrows=1, header=None)
        has_header = not str(probe.iloc[0, 0]).lstrip("-").isdigit()
        reader = pd.read_csv(
            z, header=0 if has_header else None, usecols=[1, 5],
            names=None if has_header else ["p", "t"],
            chunksize=5_000_000,
        )
        for chunk in reader:
            chunk.columns = ["p", "t"]
            ts = chunk["t"].to_numpy()
            # ms vs us timestamps: normalize to seconds
            unit = 1e6 if ts[0] > 1e14 else 1e3
            secs = (ts / unit).astype("int64")
            px = chunk["p"].to_numpy()
            # last price per second: reversed unique keeps the final value
            df = pd.DataFrame({"s": secs, "p": px}).groupby("s")["p"].last()
            grid.update(df.to_dict())
        print(f"  {os.path.basename(z)} done | grid seconds: {len(grid)} | {time.time()-t0:.0f}s")
    rows = grid_to_rows(grid)
    rows_to_csv(rows, os.path.join(HERE, "epochs_archive.csv"))


def build_collected() -> None:
    t0 = time.time()
    grid: dict = {}
    src = os.path.join(REPO_ROOT, "dataset.csv")
    for chunk in pd.read_csv(src, usecols=["timestamp", "last_price"], chunksize=5_000_000, index_col=False):
        chunk = chunk[chunk["last_price"] > 0]
        secs = chunk["timestamp"].astype("int64")
        df = pd.DataFrame({"s": secs, "p": chunk["last_price"].to_numpy()}).groupby("s")["p"].last()
        grid.update(df.to_dict())
        print(f"  ...grid seconds: {len(grid)} | {time.time()-t0:.0f}s", flush=True)
    rows = grid_to_rows(grid)
    rows_to_csv(rows, os.path.join(HERE, "epochs_collected.csv"))


if __name__ == "__main__":
    {"archive": build_archive, "collected": build_collected}[sys.argv[1]]()
