"""Fetch 5-minute regular-session bars (raw prices) for the (symbol, day) pairs where a
daily bar touched both the target and the stop, so the order of events can be decided."""
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from alpaca import bars
from universe import DATA_DIR

OUT = DATA_DIR / "intraday"


def fetch_day(date, syms):
    out = OUT / f"{date}.parquet"
    if out.exists():
        return date, "cached"
    rows = []
    for i in range(0, len(syms), 100):
        chunk = syms[i : i + 100]
        for s, b in bars(chunk, "5Min", f"{date}T13:00:00Z", f"{date}T21:00:00Z", adjustment="raw"):
            rows.append((s, b["t"], b["h"], b["l"]))
    df = pd.DataFrame(rows, columns=["symbol", "t", "h", "l"])
    if len(df):
        ts = pd.to_datetime(df.t).dt.tz_convert("America/New_York")
        mins = ts.dt.hour * 60 + ts.dt.minute
        df = df[(mins >= 570) & (mins < 960)]  # 09:30 <= bar start < 16:00
    df.to_parquet(out, index=False)
    return date, len(df)


def main(workers=6):
    OUT.mkdir(exist_ok=True)
    need = pd.read_parquet(DATA_DIR / "intraday_needs.parquet")
    groups = need.groupby("date").symbol.apply(list)
    print(len(need), "pairs over", len(groups), "days", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_day, d, sorted(s)) for d, s in groups.items()]
        for n, f in enumerate(as_completed(futs)):
            d, r = f.result()
            if n % 100 == 0:
                print(n, "/", len(groups), d, r, flush=True)
    store = {}
    for p in sorted(OUT.glob("*.parquet")):
        df = pd.read_parquet(p)
        date = p.stem
        for s, g in df.groupby("symbol", sort=False):
            g = g.sort_values("t")
            store[(s, date)] = (g.h.values.astype(np.float64), g.l.values.astype(np.float64))
    pickle.dump(store, open(DATA_DIR / "intraday_bars.pkl", "wb"))
    print("stored", len(store), flush=True)


if __name__ == "__main__":
    main()
