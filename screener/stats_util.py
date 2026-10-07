import numpy as np
from scipy import stats


def day_clustered_mean(values, days):
    """Mean with a standard error that treats each trading day as one independent cluster."""
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v, d = v[ok], np.asarray(days)[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    m = v.mean()
    _, inv = np.unique(d, return_inverse=True)
    sums = np.bincount(inv, weights=v - m)
    g = len(sums)
    se = np.sqrt(g / max(g - 1, 1) * (sums ** 2).sum()) / n
    return m, se, n


def ci95(m, se):
    return m - 1.96 * se, m + 1.96 * se


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def t_pvalue(m, se):
    if not np.isfinite(se) or se == 0:
        return np.nan
    return 2 * stats.norm.sf(abs(m / se))
