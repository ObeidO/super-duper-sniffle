"""Trade outcomes: +3% target vs stop, entry at the open, resolved with intraday bars.

Rules (agreed):
  * entry at the regular-session open of day t (plus slippage/costs)
  * target = open * 1.03, stop = open * (1 - stop_pct)
  * daily bars decide every day where only one level (or neither) was touched;
    days where BOTH were touched are resolved with 5-minute bars, and if both are
    touched inside the same 5-minute bar it counts as a loss
  * on later days of a multi-day hold, an open beyond a level fills at that open
  * if neither is hit, exit at the close of the last day of the holding period
"""
import numpy as np

TARGET = 0.03
STOPS = (0.01, 0.015, 0.02, 0.03)
HORIZONS = (1, 2, 3, 5)
COST_SIDE = 0.0010   # 0.10% per side: spread + slippage (Alpaca charges no commission)
STOP_SLIP = 0.0010   # extra 0.10% when a stop is hit (market order in a falling tape)

WIN, LOSS, TIMEOUT, AMBIG = 1, -1, 0, 9


def config_name(stop, h):
    return f"s{int(round(stop * 1000)):02d}_h{h}"


def configs():
    return [(s, h) for h in HORIZONS for s in STOPS]


def simulate(O, H, L, C, rows_t, rows_j, stop, horizon, resolver=None):
    """Simulate one config for entry rows (rows_t = date index, rows_j = symbol index).

    O,H,L,C are numpy arrays (dates x symbols), split-adjusted.
    resolver(j, d, T, S) -> WIN/LOSS for a day where both levels were touched, or None.
    Returns outcome codes, exit price ratio (exit/entry, before costs), and the list of
    (j, d, T, S) days that still need intraday resolution.
    """
    n = len(rows_t)
    E = O[rows_t, rows_j]
    T = E * (1 + TARGET)
    S = E * (1 - stop)
    out = np.full(n, TIMEOUT, dtype=np.int8)
    px = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    last_close = np.full(n, np.nan)
    need = []
    nd = O.shape[0]
    for k in range(horizon):
        d = rows_t + k
        valid = (d < nd)
        dd = np.minimum(d, nd - 1)
        o, h, l, c = O[dd, rows_j], H[dd, rows_j], L[dd, rows_j], C[dd, rows_j]
        has = valid & ~np.isnan(c)
        act = ~done & has
        if k > 0:
            gu = act & (o >= T)
            out[gu], px[gu], done[gu] = WIN, o[gu], True
            gd = act & ~gu & (o <= S)
            out[gd], px[gd], done[gd] = LOSS, o[gd], True
            act = act & ~done
        ht, hs = act & (h >= T), act & (l <= S)
        both = ht & hs
        only_t = ht & ~hs
        only_s = hs & ~ht
        out[only_t], px[only_t], done[only_t] = WIN, T[only_t], True
        out[only_s], px[only_s], done[only_s] = LOSS, S[only_s], True
        for i in np.flatnonzero(both):
            r = resolver(rows_j[i], d[i], T[i], S[i]) if resolver else None
            if r is None:
                need.append((rows_j[i], d[i], T[i], S[i]))
                r = AMBIG
            out[i] = r
            px[i] = T[i] if r == WIN else S[i]
            done[i] = True
        upd = act & ~done
        last_close[upd] = c[upd]
        # a symbol that stopped trading mid-hold exits at its last close
    to = ~done
    px[to] = last_close[to]
    out[to & np.isnan(px)] = TIMEOUT
    ratio = px / E
    return out, ratio, need


def net_return(out, ratio):
    sell_cost = COST_SIDE + np.where(out == LOSS, STOP_SLIP, 0.0)
    return ratio * (1 - sell_cost) / (1 + COST_SIDE) - 1
