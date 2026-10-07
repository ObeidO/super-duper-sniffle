"""Point-in-time features.

Every indicator is computed on bars up to and including day d, then shifted one
trading day forward so that the row for entry day t only sees data through the
close of t-1. The entry-day open is never used as a feature.
"""
import numpy as np
import pandas as pd
from numba import njit

MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 20e6

FAMILIES = {}  # feature name -> family


def reg(family, name, values, store):
    FAMILIES[name] = family
    store[name] = values


# ---------------------------------------------------------------- helpers
def sma(x, n):
    return x.rolling(n, min_periods=n).mean()


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(c, n):
    d = c.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---------------------------------------------------------------- swings (causal zigzag)
@njit(cache=True)
def _zigzag(h, l, c, th):
    """Causal zigzag on one symbol. th[i] is the reversal fraction at bar i.

    A swing high is only *confirmed* once price has fallen th below it, and the
    confirmation happens on that later bar, so nothing here peeks ahead.
    Returns, for each bar, the last 6 confirmed pivot prices (newest last) and the
    type of the newest pivot (+1 high, -1 low, 0 none).
    """
    n = len(c)
    piv = np.full((n, 6), np.nan)
    last_type = np.zeros(n)
    buf = np.full(6, np.nan)
    btype = 0.0
    direction = 0  # +1 up-leg (tracking a high), -1 down-leg (tracking a low)
    ext = np.nan
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(th[i]):
            piv[i, :] = buf
            last_type[i] = btype
            continue
        if direction == 0:
            direction = 1
            ext = h[i]
        elif direction == 1:
            if h[i] > ext:
                ext = h[i]
            elif l[i] <= ext * (1 - th[i]):
                buf[:5] = buf[1:]
                buf[5] = ext
                btype = 1.0
                direction = -1
                ext = l[i]
        else:
            if l[i] < ext:
                ext = l[i]
            elif h[i] >= ext * (1 + th[i]):
                buf[:5] = buf[1:]
                buf[5] = ext
                btype = -1.0
                direction = 1
                ext = h[i]
        piv[i, :] = buf
        last_type[i] = btype
    return piv, last_type


def swing_features(H, L, C, atrp):
    """Fibonacci and Elliott-wave-approximation features from causal swings."""
    th = np.clip(2.5 * atrp, 0.04, 0.20)
    shape = C.shape
    names = ["fib_retr", "fib_near", "fib_golden", "fib_room_up", "fib_ext_room",
             "ew_w3", "ew_w4", "ew_w5", "ew_abc"]
    out = {k: np.full(shape, np.nan, dtype=np.float32) for k in names}
    Hv, Lv, Cv, Tv = H.values, L.values, C.values, th.values
    levels = np.array([0.236, 0.382, 0.5, 0.618, 0.786])
    for j in range(shape[1]):
        piv, lt = _zigzag(Hv[:, j], Lv[:, j], Cv[:, j], Tv[:, j])
        c = Cv[:, j]
        p0, p1 = piv[:, 4], piv[:, 5]  # previous pivot, newest pivot
        hi = np.where(lt == 1, p1, p0)
        lo = np.where(lt == 1, p0, p1)
        rng = hi - lo
        ok = (rng > 0) & ~np.isnan(rng)
        # position of price inside the last swing (0 = swing low, 1 = swing high)
        pos = np.where(ok, (c - lo) / np.where(ok, rng, 1), np.nan)
        # retracement measured from the end of the last completed leg
        retr = np.where(lt == 1, 1 - pos, pos)
        out["fib_retr"][:, j] = retr
        lvl_px = lo[:, None] + rng[:, None] * levels[None, :]
        dist = np.abs(c[:, None] - lvl_px) / c[:, None]
        out["fib_near"][:, j] = np.nanmin(np.where(ok[:, None], dist, np.nan), axis=1)
        # golden zone: pulled back 50-61.8% of an up-swing (classic long setup)
        out["fib_golden"][:, j] = np.where(ok, ((lt == 1) & (retr >= 0.5) & (retr <= 0.618)).astype(float), np.nan)
        above = np.where(lvl_px > c[:, None] * 1.0005, lvl_px, np.nan)
        allabove = np.concatenate([above, np.where(ok, hi, np.nan)[:, None]], axis=1)
        room = (np.nanmin(np.where(np.isnan(allabove), np.inf, allabove), axis=1) / c) - 1
        out["fib_room_up"][:, j] = np.where(ok & np.isfinite(room), room, np.nan)
        ext = lo + 1.272 * rng
        out["fib_ext_room"][:, j] = np.where(ok, ext / c - 1, np.nan)

        # --- Elliott wave approximation (rule-based, not a true wave count) ---
        a, b, cc, d, e, f = (piv[:, k] for k in range(6))
        newest_low = lt == -1
        newest_high = lt == 1
        # pattern ending in a low: ... L(d) H(e) L(f)  -> wave1 = d->e, wave2 = e->f
        w1 = e - d
        w3_setup = newest_low & (f > d) & (w1 > 0) & ((e - f) / np.where(w1 > 0, w1, 1) <= 0.786) & ((e - f) / np.where(w1 > 0, w1, 1) >= 0.382) & (c > f)
        out["ew_w3"][:, j] = np.where(np.isnan(f), np.nan, w3_setup.astype(float))
        # pattern L(b) H(c) L(d) H(e) L(f) with wave3 (d->e) > wave1 (b->c), wave4 low f above wave1 high c
        w4 = newest_low & (cc > b) & (e > cc) & (d > b) & ((e - d) > (cc - b)) & (f > cc)
        out["ew_w4"][:, j] = np.where(np.isnan(b), np.nan, w4.astype(float))
        # pattern L(a) H(b) L(c) H(d) L(e) and now above d: possible wave 5 (late stage)
        w5 = newest_low & (b > a) & (d > b) & (cc > a) & (e > b) & ((d - cc) >= (b - a)) & (c > d)
        out["ew_w5"][:, j] = np.where(np.isnan(a), np.nan, w5.astype(float))
        # ABC correction: high(d) -> A low(e)?  use H(c) L(d) H(e) L(f): A=c->d, B=d->e (e<c), C=e->f ~ A
        A = cc - d
        Cl = e - f
        abc = newest_low & (e < cc) & (A > 0) & (Cl >= 0.618 * A) & (Cl <= 1.618 * A) & (f < d)
        out["ew_abc"][:, j] = np.where(np.isnan(cc), np.nan, abc.astype(float))
    return {k: pd.DataFrame(v, index=C.index, columns=C.columns) for k, v in out.items()}


# ---------------------------------------------------------------- candlesticks
def candles(O, H, L, C):
    body = (C - O)
    rng = (H - L).replace(0, np.nan)
    ab = body.abs()
    up_sh = H - np.maximum(O, C)
    lo_sh = np.minimum(O, C) - L
    bull, bear = body > 0, body < 0
    O1, C1, H1, L1 = O.shift(1), C.shift(1), H.shift(1), L.shift(1)
    O2, C2 = O.shift(2), C.shift(2)
    b1 = C1 - O1
    ab1 = b1.abs()
    trend_dn = C.shift(1) < C.shift(6)  # short-term downtrend before the pattern
    trend_up = C.shift(1) > C.shift(6)
    small = ab <= 0.1 * rng
    p = {}
    p["cdl_doji"] = small
    p["cdl_hammer"] = (lo_sh >= 2 * ab) & (up_sh <= 0.3 * ab.clip(lower=1e-9) + 0.1 * rng) & trend_dn & ~small
    p["cdl_inv_hammer"] = (up_sh >= 2 * ab) & (lo_sh <= 0.1 * rng) & trend_dn & ~small
    p["cdl_shooting_star"] = (up_sh >= 2 * ab) & (lo_sh <= 0.1 * rng) & trend_up & ~small
    p["cdl_hanging_man"] = (lo_sh >= 2 * ab) & (up_sh <= 0.1 * rng) & trend_up & ~small
    p["cdl_bull_engulf"] = bull & (b1 < 0) & (O <= C1) & (C >= O1) & (ab > ab1)
    p["cdl_bear_engulf"] = bear & (b1 > 0) & (O >= C1) & (C <= O1) & (ab > ab1)
    p["cdl_piercing"] = bull & (b1 < 0) & (O < L1) & (C > (O1 + C1) / 2) & (C < O1)
    p["cdl_dark_cloud"] = bear & (b1 > 0) & (O > H1) & (C < (O1 + C1) / 2) & (C > O1)
    star1 = (C1 - O1).abs() <= 0.3 * (H1 - L1)
    p["cdl_morning_star"] = (C2 < O2) & star1 & bull & (C > (O2 + C2) / 2)
    p["cdl_evening_star"] = (C2 > O2) & star1 & bear & (C < (O2 + C2) / 2)
    p["cdl_three_white"] = bull & (b1 > 0) & (C2 > O2) & (C > C1) & (C1 > C2) & (O > O1) & (O1 > O2)
    p["cdl_three_black"] = bear & (b1 < 0) & (C2 < O2) & (C < C1) & (C1 < C2) & (O < O1) & (O1 < O2)
    p["cdl_bull_harami"] = bull & (b1 < 0) & (O > C1) & (C < O1) & (ab < 0.6 * ab1)
    p["cdl_bear_harami"] = bear & (b1 > 0) & (O < C1) & (C > O1) & (ab < 0.6 * ab1)
    p["cdl_bull_marubozu"] = bull & (ab >= 0.9 * rng)
    p["cdl_bear_marubozu"] = bear & (ab >= 0.9 * rng)
    valid = C.notna() & C2.notna()
    return {k: v.astype(float).where(valid) for k, v in p.items()}


# ---------------------------------------------------------------- main builder
def build(O, H, L, C, V, Craw, spy):
    """O,H,L,C,V: split-adjusted wide frames (dates x symbols); Craw raw close; spy: DataFrame o,h,l,c.

    Returns dict name -> wide frame of features *as of the close of each date*
    (to be shifted by one day before use), plus the liquidity mask as of that close.
    """
    f = {}
    ret = C.pct_change(fill_method=None)
    lr = np.log(C).diff()
    pc = C.shift(1)
    tr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs()).where(C.notna())
    atr14 = wilder(tr, 14)
    atrp = atr14 / C

    # volatility
    reg("volatility", "atr_pct", atrp, f)
    reg("volatility", "rv20", lr.rolling(20, min_periods=20).std(), f)
    reg("volatility", "range20", sma((H - L) / C, 20), f)
    reg("volatility", "hit3_20", sma((H >= O * 1.03).astype(float).where(C.notna()), 20), f)
    reg("volatility", "hit3_60", sma((H >= O * 1.03).astype(float).where(C.notna()), 60), f)
    reg("volatility", "atr_ratio", wilder(tr, 5) / atr14, f)
    m20, s20 = sma(C, 20), C.rolling(20, min_periods=20).std()
    reg("volatility", "bb_pctb", (C - (m20 - 2 * s20)) / (4 * s20), f)
    reg("volatility", "bb_width", 4 * s20 / m20, f)
    bw = 4 * s20 / m20
    reg("volatility", "bb_squeeze", (bw <= bw.rolling(120, min_periods=60).quantile(0.1)).astype(float).where(bw.notna()), f)

    # trend
    for n in (10, 20, 50, 200):
        reg("trend", f"c_sma{n}", C / sma(C, n) - 1, f)
    s20, s50, s200 = sma(C, 20), sma(C, 50), sma(C, 200)
    reg("trend", "sma20_slope", s20 / s20.shift(5) - 1, f)
    reg("trend", "sma50_200", s50 / s200 - 1, f)
    upm, dnm = H.diff(), -L.diff()
    pdm = upm.where((upm > dnm) & (upm > 0), 0.0)
    mdm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    pdi = 100 * wilder(pdm.where(C.notna()), 14) / atr14
    mdi = 100 * wilder(mdm.where(C.notna()), 14) / atr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    reg("trend", "adx14", wilder(dx, 14), f)
    reg("trend", "di_diff", pdi - mdi, f)

    # momentum
    reg("momentum", "rsi2", rsi(C, 2), f)
    reg("momentum", "rsi14", rsi(C, 14), f)
    macd = ema(C, 12) - ema(C, 26)
    sig = ema(macd, 9)
    reg("momentum", "macd_hist", (macd - sig) / C, f)
    reg("momentum", "macd_cross_up", ((macd > sig) & (macd.shift(1) <= sig.shift(1))).astype(float).where(sig.notna()), f)
    ll14, hh14 = L.rolling(14, min_periods=14).min(), H.rolling(14, min_periods=14).max()
    k = 100 * (C - ll14) / (hh14 - ll14).replace(0, np.nan)
    reg("momentum", "stoch_k", k, f)
    reg("momentum", "stoch_d", sma(k, 3), f)
    for n in (1, 5, 20, 60):
        reg("momentum", f"roc{n}", C / C.shift(n) - 1, f)

    # volume
    v20 = sma(V, 20)
    reg("volume", "relvol", V / v20, f)
    reg("volume", "relvol5", sma(V, 5) / v20, f)
    obv = (np.sign(C.diff()) * V).fillna(0).cumsum().where(C.notna())
    reg("volume", "obv_slope", (obv - obv.shift(10)) / (10 * v20), f)
    upv = (V.where(ret > 0, 0.0)).rolling(20, min_periods=20).sum()
    dnv = (V.where(ret < 0, 0.0)).rolling(20, min_periods=20).sum()
    reg("volume", "updown_vol", np.log((upv + 1) / (dnv + 1)), f)
    dv20 = sma(C * V, 20)  # adjusted price x adjusted volume = true dollars traded
    reg("volume", "log_dollar_vol", np.log(dv20), f)

    # candlesticks
    for kname, v in candles(O, H, L, C).items():
        reg("candlestick", kname, v, f)

    # support / resistance
    for n in (20, 60, 252):
        reg("support_resistance", f"dist_hi{n}", C / H.rolling(n, min_periods=n).max() - 1, f)
        reg("support_resistance", f"dist_lo{n}", C / L.rolling(n, min_periods=n).min() - 1, f)
    reg("support_resistance", "breakout20", (C > H.shift(1).rolling(20, min_periods=20).max()).astype(float).where(C.notna()), f)
    lo60 = L.shift(1).rolling(60, min_periods=60).min()
    reg("support_resistance", "support_test", ((L <= lo60 * 1.01) & (C > lo60)).astype(float).where(lo60.notna()), f)
    # room to nearest prior high above today's close (resistance), over 252d of daily highs
    room = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    for n in (5, 10, 20, 60, 120, 252):
        hh = H.rolling(n, min_periods=n).max()
        cand = (hh / C - 1).where(hh > C * 1.001)
        room = room.where(room.notna() & (room <= cand.fillna(np.inf)), cand)
    reg("support_resistance", "resist_room", room, f)

    # gaps
    gap = O / C.shift(1) - 1
    reg("gaps", "gap_last", gap, f)
    reg("gaps", "gap_up_count20", (gap > 0.02).astype(float).where(gap.notna()).rolling(20, min_periods=20).sum(), f)
    reg("gaps", "gap_dn_count20", (gap < -0.02).astype(float).where(gap.notna()).rolling(20, min_periods=20).sum(), f)
    reg("gaps", "gap_up_held", ((gap > 0.02) & (C > O)).astype(float).where(gap.notna()), f)
    reg("gaps", "gap_dn_filled", ((gap < -0.02) & (H >= C.shift(1))).astype(float).where(gap.notna()), f)
    reg("gaps", "close_loc", (C - L) / (H - L).replace(0, np.nan), f)

    # strength vs SPY
    sc = spy["c"]
    for n in (5, 20, 60):
        reg("relative_strength", f"rs{n}", (C / C.shift(n)).sub(sc / sc.shift(n), axis=0), f)
    spy_r = sc.pct_change()
    reg("relative_strength", "rs_today", ret.sub(spy_r, axis=0), f)

    # market conditions (same value for every stock on a date)
    def bc(series):
        return pd.DataFrame(np.repeat(series.values[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)

    spy_lr = np.log(sc).diff()
    reg("market", "spy_c_sma50", bc(sc / sma(sc, 50) - 1), f)
    reg("market", "spy_c_sma200", bc(sc / sma(sc, 200) - 1), f)
    reg("market", "spy_ret1", bc(spy_r), f)
    reg("market", "spy_ret5", bc(sc / sc.shift(5) - 1), f)
    reg("market", "spy_rv20", bc(spy_lr.rolling(20).std()), f)
    liq = (Craw >= MIN_PRICE) & (dv20 >= MIN_DOLLAR_VOL)
    above50 = (C > sma(C, 50)).where(liq)
    reg("market", "breadth50", bc(above50.mean(axis=1)), f)
    reg("market", "univ_ret1", bc(ret.where(liq).median(axis=1)), f)

    # Fibonacci + Elliott wave approximation
    sw = swing_features(H, L, C, atrp)
    for kname in ("fib_retr", "fib_near", "fib_golden", "fib_room_up", "fib_ext_room"):
        reg("fibonacci", kname, sw[kname], f)
    for kname in ("ew_w3", "ew_w4", "ew_w5", "ew_abc"):
        reg("elliott_approx", kname, sw[kname], f)

    return f, liq
