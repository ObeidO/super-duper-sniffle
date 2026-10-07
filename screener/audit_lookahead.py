"""Look-ahead audit.

Truncation test: rebuild every feature using only data up to a cutoff date, with the
entry day's own bar blanked out, then compare the row for that entry day with the same row
from the full-history build. Any feature that used the entry-day bar or any later data
would differ.
"""
import json
import sys

import numpy as np
import pandas as pd

import features as F
from universe import DATA_DIR

RES = DATA_DIR.parent / "results"


def wide_subset(symbols, dates):
    sp = pd.read_parquet(DATA_DIR / "daily_split.parquet", filters=[("symbol", "in", symbols)])
    rw = pd.read_parquet(DATA_DIR / "daily_raw.parquet", columns=["symbol", "date", "c"], filters=[("symbol", "in", symbols)])
    sp = sp.drop_duplicates(["symbol", "date"])
    rw = rw.drop_duplicates(["symbol", "date"])
    W = {k: sp.pivot(index="date", columns="symbol", values=k).reindex(dates) for k in "ohlcv"}
    W["craw"] = rw.pivot(index="date", columns="symbol", values="c").reindex(dates)[W["c"].columns]
    return W


def build_rows(W, entry_dates):
    spy = pd.DataFrame({k: W[k]["SPY"] for k in "ohlc"})
    cols = [c for c in W["c"].columns if c != "SPY"]
    feats, liq = F.build(*(W[k][cols] for k in ("o", "h", "l", "c", "v", "craw")), spy)
    out = {n: fr.shift(1).loc[entry_dates] for n, fr in feats.items()}
    out["_liq_prev"] = liq.shift(1, fill_value=False).astype(float).loc[entry_dates]
    return out


def main(n_symbols=250, n_cutoffs=6, seed=3):
    rng = np.random.default_rng(seed)
    fe = pd.read_parquet(DATA_DIR / "features.parquet", columns=["symbol"])
    syms = sorted(rng.choice(fe.symbol.unique(), n_symbols, replace=False).tolist()) + ["SPY"]
    all_dates = sorted(pd.read_parquet(DATA_DIR / "daily_split.parquet", columns=["date"], filters=[("symbol", "==", "SPY")]).date)
    Wfull = wide_subset(syms, all_dates)
    cut_idx = sorted(rng.choice(np.arange(300, len(all_dates) - 1), n_cutoffs, replace=False))
    entry_dates = [all_dates[i + 1] for i in cut_idx]
    full = build_rows(Wfull, entry_dates)
    report = {"symbols_tested": n_symbols, "cutoffs": [all_dates[i] for i in cut_idx], "features": {}}
    worst = {}
    for i, ed in zip(cut_idx, entry_dates):
        # truncated history: data through the cutoff close, plus an EMPTY row for the entry day
        dates = all_dates[: i + 1] + [ed]
        Wt = {k: v.loc[dates].copy() for k, v in Wfull.items()}
        for k in Wt:
            Wt[k].iloc[-1] = np.nan  # entry-day bar is unknown before the open
        tr = build_rows(Wt, [ed])
        for name in full:
            a = full[name].loc[[ed]].values.astype(float)
            b = tr[name].values.astype(float)
            both_nan = (np.isnan(a) & np.isnan(b)) | (a == b)  # a == b also covers matching +/-inf
            diff = np.where(both_nan, 0, np.abs(a - b) / (np.abs(a) + 1e-9))
            diff = np.where(np.isnan(diff), 1.0, diff)  # one NaN, the other not -> mismatch
            worst[name] = max(worst.get(name, 0.0), float(np.nanmax(diff)))
    bad = {k: v for k, v in worst.items() if v > 1e-4}
    report["features"] = worst
    report["mismatches"] = bad
    report["passed"] = len(bad) == 0
    print("truncation test:", "PASSED" if not bad else f"FAILED {bad}")
    json.dump(report, open(RES / "audit_truncation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
