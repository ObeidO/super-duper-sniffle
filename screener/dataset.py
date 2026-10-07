"""Build the point-in-time research table: one row per (entry day, liquid stock).

Usage:
  python dataset.py features   # features + universe -> data/features.parquet
  python dataset.py needs      # list (symbol, day) pairs needing intraday bars
  python dataset.py labels     # outcomes for every stop/horizon config -> data/labels.parquet
"""
import pickle
import sys

import numpy as np
import pandas as pd

import features as F
import labels as LB
from universe import DATA_DIR


def load_wide(extra_dates=()):
    sp = pd.read_parquet(DATA_DIR / "daily_split.parquet").drop_duplicates(["symbol", "date"])
    rw = pd.read_parquet(DATA_DIR / "daily_raw.parquet", columns=["symbol", "date", "o", "c", "v"]).drop_duplicates(["symbol", "date"])
    dates = sorted(set(sp.loc[sp.symbol == "SPY", "date"]) | set(extra_dates))
    # only keep symbols that were ever liquid (raw price >= $5 and 20d $vol >= $20M)
    rw = rw.sort_values(["symbol", "date"])
    dv = (rw.c * rw.v).groupby(rw.symbol).transform(lambda s: s.rolling(20, min_periods=20).mean())
    ever = set(rw.loc[(rw.c >= F.MIN_PRICE) & (dv >= F.MIN_DOLLAR_VOL), "symbol"]) | {"SPY"}
    sp = sp[sp.symbol.isin(ever)]
    rw = rw[rw.symbol.isin(ever)]
    sp, rw = split_reused_tickers(sp, rw, dates)
    W = {}
    for k in "ohlcv":
        W[k] = sp.pivot(index="date", columns="symbol", values=k).reindex(dates)
    W["craw"] = rw.pivot(index="date", columns="symbol", values="c").reindex(dates)[W["c"].columns]
    W["oraw"] = rw.pivot(index="date", columns="symbol", values="o").reindex(dates)[W["c"].columns]
    return W


def split_reused_tickers(sp, rw, dates, gap=60):
    """A ticker that goes silent for 60+ sessions and comes back is treated as a new series
    (often a different company reusing the symbol), so its indicators restart from scratch.
    Earlier segments keep their history under 'SYM~1', 'SYM~2', ..."""
    pos = {d: i for i, d in enumerate(dates)}
    out = []
    for df in (sp, rw):
        df = df.sort_values(["symbol", "date"]).copy()
        ix = df.date.map(pos)
        brk = (ix.diff() > gap) & (df.symbol == df.symbol.shift())
        seg = brk.groupby(df.symbol).cumsum()
        last = seg.groupby(df.symbol).transform("max")
        df["symbol"] = np.where(seg < last, df.symbol + "~" + (seg + 1).astype(str), df.symbol)
        out.append(df)
    return out


def build_features(W):
    spy = pd.DataFrame({k: W[k]["SPY"] for k in "ohlc"})
    cols = [c for c in W["c"].columns if c != "SPY"]
    O, H, L, C, V, Cr = (W[k][cols] for k in ("o", "h", "l", "c", "v", "craw"))
    feats, liq = F.build(O, H, L, C, V, Cr, spy)
    # shift: row t sees only information up to the close of t-1
    liq_prev = liq.shift(1, fill_value=False).astype(bool)
    return feats, liq_prev, cols


def to_long(feats, mask, cols):
    idx_t, idx_j = np.nonzero(mask.values)
    dates = mask.index.values
    df = pd.DataFrame({"date": dates[idx_t], "symbol": np.array(cols)[idx_j],
                       "t": idx_t.astype(np.int32), "j": idx_j.astype(np.int32)})
    for name, fr in feats.items():
        df[name] = fr.shift(1).values[idx_t, idx_j].astype(np.float32)
    df["entry_weekday"] = pd.to_datetime(df.date).dt.dayofweek.astype(np.float32)
    F.FAMILIES["entry_weekday"] = "market"
    return df


def cmd_features():
    W = load_wide()
    feats, liq_prev, cols = build_features(W)
    O = W["o"][cols]
    mask = liq_prev & O.notna() & feats["atr_pct"].shift(1).notna()
    df = to_long(feats, mask, cols)
    df["entry_open"] = O.values[df.t, df.j]
    df["entry_open_raw"] = W["oraw"][cols].values[df.t, df.j]
    df["prev_close"] = W["c"][cols].shift(1).values[df.t, df.j]
    df.to_parquet(DATA_DIR / "features.parquet", index=False)
    pickle.dump(F.FAMILIES, open(DATA_DIR / "families.pkl", "wb"))
    print("rows", len(df), "days", df.date.nunique(), "symbols", df.symbol.nunique())
    print(df.groupby(df.date.str[:4]).symbol.count() / df.groupby(df.date.str[:4]).date.nunique())


class IntradayResolver:
    def __init__(self, W, cols):
        self.dates = W["c"].index.values
        self.cols = cols
        self.O = W["o"][cols].values
        self.Oraw = W["oraw"][cols].values
        self.bars = pickle.load(open(DATA_DIR / "intraday_bars.pkl", "rb"))
        self.stats = {"resolved": 0, "same_bar": 0, "missing": 0, "no_touch": 0}

    def __call__(self, j, d, T, S):
        key = (self.cols[j].split("~")[0], self.dates[d])
        b = self.bars.get(key)
        if b is None:
            self.stats["missing"] += 1
            return LB.LOSS  # conservative
        f = self.Oraw[d, j] / self.O[d, j]
        hi, lo = b
        ht = hi >= T * f
        hs = lo <= S * f
        it = np.argmax(ht) if ht.any() else 10**9
        is_ = np.argmax(hs) if hs.any() else 10**9
        if it == 10**9 and is_ == 10**9:
            self.stats["no_touch"] += 1
            return LB.LOSS
        self.stats["resolved"] += 1
        if it == is_:
            self.stats["same_bar"] += 1
            return LB.LOSS
        return LB.WIN if it < is_ else LB.LOSS


def cmd_labels(resolve):
    W = load_wide()
    cols = [c for c in W["c"].columns if c != "SPY"]
    df = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "t", "j"])
    O, H, L, C = (W[k][cols].values for k in "ohlc")
    res = IntradayResolver(W, cols) if resolve else None
    out = df[["date", "symbol"]].copy()
    needs = set()
    for stop, h in LB.configs():
        code, ratio, need = LB.simulate(O, H, L, C, df.t.values, df.j.values, stop, h, res)
        name = LB.config_name(stop, h)
        out[f"y_{name}"] = code
        out[f"r_{name}"] = LB.net_return(code, ratio).astype(np.float32)
        needs.update((cols[j].split("~")[0], W["c"].index[d]) for j, d, _, _ in need)
        print(name, "win%", round((code == LB.WIN).mean() * 100, 2), "ambig", len(need), flush=True)
    # did the stock touch +3% at any point during the entry day (no stop) - the plain base rate
    t, j = df.t.values, df.j.values
    out["touch3_day"] = (H[t, j] >= O[t, j] * 1.03).astype(np.int8)
    out["ret_oc"] = (C[t, j] / O[t, j] - 1).astype(np.float32)
    if resolve:
        out.to_parquet(DATA_DIR / "labels.parquet", index=False)
        print("resolver stats", res.stats)
    else:
        pd.DataFrame(sorted(needs), columns=["symbol", "date"]).to_parquet(DATA_DIR / "intraday_needs.parquet", index=False)
        print("intraday pairs needed", len(needs))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        cmd_features()
    elif cmd == "needs":
        cmd_labels(False)
    elif cmd == "labels":
        cmd_labels(True)
