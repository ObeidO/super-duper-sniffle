"""Backtest research: base rates, signal tests, walk-forward models, calibration, baselines.

All evaluation is out-of-sample by calendar year: a model that scores year Y was trained
only on rows whose outcome was fully known at least 5 trading days before Y started.
Model hyperparameters are fixed in advance (never tuned on any test year).
"""
import json
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import labels as LB
from stats_util import bh_qvalues, ci95, day_clustered_mean, t_pvalue, wilson
from universe import DATA_DIR

RES = DATA_DIR.parent / "results"
MODELS = DATA_DIR.parent / "models"
MAIN = "s20_h1"               # reference config for signal tests: same day, 2% stop
TEST_YEARS = list(range(2019, 2027))
DISCOVERY_END = "2021-12-31"  # signal tests: discover 2016-2021, confirm 2022-2026
EMBARGO = 5                   # trading days dropped between train and test
SEED = 7
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000,
              feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, seed=SEED, num_threads=4)
ROUNDS = 250


def load():
    fe = pd.read_parquet(DATA_DIR / "features.parquet")
    lb = pd.read_parquet(DATA_DIR / "labels.parquet")
    assert (fe.date.values == lb.date.values).all() and (fe.symbol.values == lb.symbol.values).all()
    df = pd.concat([fe, lb.drop(columns=["date", "symbol"])], axis=1)
    fam = pickle.load(open(DATA_DIR / "families.pkl", "rb"))
    fam["entry_weekday"] = "market"
    nd = df.t.max()
    return df, fam, nd


def feature_cols(fam):
    return [k for k in fam]


def valid_rows(df, nd, h):
    # outcome must be fully observed: entry day + h-1 must be inside the data
    return df.t.values <= nd - (h - 1)


# ------------------------------------------------------------------ base rates
def base_rates(df, nd):
    out = {}
    d1 = df[valid_rows(df, nd, 1)]
    out["n_rows"] = int(len(d1))
    out["n_days"] = int(d1.date.nunique())
    out["avg_universe"] = float(d1.groupby("date").size().mean())
    out["touch3_any"] = float(d1.touch3_day.mean())
    cfg = []
    for stop, h in LB.configs():
        name = LB.config_name(stop, h)
        d = df[valid_rows(df, nd, h)]
        y = d[f"y_{name}"].values
        r = d[f"r_{name}"].values
        m, se, n = day_clustered_mean(r, cluster_ids(d.date.values, name))
        cfg.append(dict(config=name, stop=stop, horizon=h, win=float((y == LB.WIN).mean()),
                        loss=float((y == LB.LOSS).mean()), timeout=float((y == LB.TIMEOUT).mean()),
                        avg_net=float(m), avg_net_lo=float(ci95(m, se)[0]), avg_net_hi=float(ci95(m, se)[1]), n=int(n)))
    out["configs"] = cfg
    # by volatility decile (daily cross-sectional ATR% rank) - "comparable stocks"
    d1 = d1.copy()
    d1["vdec"] = vol_decile(d1)
    g = d1.groupby("vdec")
    out["by_vol_decile"] = [dict(decile=int(k), atr_pct=float(v.atr_pct.median()), touch3=float(v.touch3_day.mean()),
                                 win=float((v[f"y_{MAIN}"] == LB.WIN).mean()), avg_net=float(v[f"r_{MAIN}"].mean()),
                                 n=int(len(v))) for k, v in g]
    yr = d1.groupby(d1.date.str[:4])
    out["by_year"] = [dict(year=k, touch3=float(v.touch3_day.mean()), win=float((v[f"y_{MAIN}"] == LB.WIN).mean()),
                           universe=float(v.groupby("date").size().mean()), n=int(len(v))) for k, v in yr]
    return out


def vol_decile(d):
    return d.groupby("date").atr_pct.transform(lambda s: np.ceil(s.rank(pct=True) * 10).clip(1, 10)).astype(int)


# ------------------------------------------------------------------ individual signals
def signal_tests(df, fam, nd):
    d = df[valid_rows(df, nd, 1)].copy()
    y = (d[f"y_{MAIN}"] == LB.WIN).astype(float)
    r = d[f"r_{MAIN}"].astype(float)
    d["vdec"] = vol_decile(d)
    key = d.date + "|" + d.vdec.astype(str)
    # volatility-matched expectation: same day, same ATR% decile
    exp_y = y.groupby(key).transform("mean")
    exp_r = r.groupby(key).transform("mean")
    ex_y, ex_r = (y - exp_y).values, (r - exp_r).values
    disc = (d.date <= DISCOVERY_END).values
    rows = []
    for feat, family in fam.items():
        if family == "volatility" and feat != "bb_squeeze":
            continue  # volatility is the matching variable itself; reported separately
        x = d[feat]
        uniq = x.dropna().unique()
        if len(uniq) <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0}):
            tests = {f"{feat}": (x == 1).values}
        else:
            pr = x.groupby(d.date).rank(pct=True)
            tests = {f"{feat} (top 20%)": (pr > 0.8).values, f"{feat} (bottom 20%)": (pr <= 0.2).values}
        for tname, on in tests.items():
            rec = dict(signal=tname, feature=feat, family=family)
            for part, sel in (("disc", disc), ("conf", ~disc)):
                s = on & sel
                n = int(s.sum())
                rec[f"{part}_n"] = n
                if n < 200:
                    rec.update({f"{part}_{k}": np.nan for k in ("win", "base", "lift", "p", "net_lift", "net")})
                    continue
                m, se, _ = day_clustered_mean(ex_y[s], d.date.values[s])
                mr, ser, _ = day_clustered_mean(ex_r[s], d.date.values[s])
                rec[f"{part}_win"] = float(y.values[s].mean())
                rec[f"{part}_base"] = float(exp_y.values[s].mean())
                rec[f"{part}_lift"] = float(m)
                rec[f"{part}_p"] = float(t_pvalue(m, se))
                rec[f"{part}_net"] = float(r.values[s].mean())
                rec[f"{part}_net_lift"] = float(mr)
                rec[f"{part}_net_p"] = float(t_pvalue(mr, ser))
            rows.append(rec)
    t = pd.DataFrame(rows)
    t["disc_q"] = bh_qvalues(t.disc_p.values)
    t["disc_bonf"] = np.minimum(t.disc_p * t.disc_p.notna().sum(), 1)
    # earned weight: survives multiple-testing in discovery AND same direction, p<0.05 in confirmation
    t["confirmed"] = (t.disc_q < 0.05) & (t.conf_p < 0.05) & (np.sign(t.disc_lift) == np.sign(t.conf_lift))
    return t


# ------------------------------------------------------------------ walk-forward models
def year_of(df):
    return df.date.str[:4].astype(int).values


def fit_predict(df, feats, ycol, h, nd, years=None, subsample=1.0, store_last=False):
    years = list(TEST_YEARS) if years is None else years
    yrs = year_of(df)
    ok = valid_rows(df, nd, h)
    pred = np.full(len(df), np.nan)
    rng = np.random.default_rng(SEED)
    target = (df[ycol].values == LB.WIN).astype(np.float32)
    X = df[feats]
    model = last_wf = None
    for Y in years + ([None] if store_last is True else []):
        if Y is None:
            # final model for live use: everything with a fully known outcome
            tr = ok.copy()
            te = None
        else:
            te = (yrs == Y) & ok
            if not te.any():
                continue
            t0 = df.t.values[te].min()
            tr = ok & (df.t.values + (h - 1) < t0 - EMBARGO)
        idx = np.flatnonzero(tr)
        if len(idx) < 10000:
            continue  # not enough history to train for this year
        if subsample < 1:
            idx = rng.choice(idx, int(len(idx) * subsample), replace=False)
            idx.sort()
        ds = lgb.Dataset(X.values[idx], target[idx], feature_name=list(feats), free_raw_data=True)
        model = lgb.train(PARAMS, ds, ROUNDS)
        if te is not None:
            pred[te] = model.predict(X.values[te])
            last_wf = model
    return pred, (last_wf if store_last == "wf" else model)


def daily_top(df, score, k=1, mask=None):
    s = pd.Series(score, index=df.index)
    if mask is not None:
        s = s.where(mask)
    s = s.dropna()
    rk = s.groupby(df.date.loc[s.index]).rank(ascending=False, method="first")
    return s.index[rk.values <= k]


def cluster_ids(dates, name):
    # multi-day holds overlap, so cluster by calendar week instead of by day
    if name.endswith("_h1"):
        return np.asarray(dates)
    return pd.to_datetime(pd.Series(dates)).dt.strftime("%G-%V").values


def pick_stats(df, idx, name):
    sub = df.loc[idx]
    y = sub[f"y_{name}"].values
    r = sub[f"r_{name}"].values
    m, se, n = day_clustered_mean(r, cluster_ids(sub.date.values, name))
    k = int((y == LB.WIN).sum())
    lo, hi = wilson(k, n)
    return dict(n=int(n), win=float(k / n) if n else np.nan, win_lo=lo, win_hi=hi,
                avg_net=float(m), net_lo=float(ci95(m, se)[0]), net_hi=float(ci95(m, se)[1]),
                total_net=float(np.nansum(r)))


def baselines(df, name, h, nd, oos):
    d = df[oos & valid_rows(df, nd, h)]
    out = {}
    # random liquid stock: exact expectation = average over each day's universe
    r = d[f"r_{name}"].values
    y = (d[f"y_{name}"].values == LB.WIN).astype(float)
    day_r = pd.Series(r).groupby(d.date.values).mean()
    day_y = pd.Series(y).groupby(d.date.values).mean()
    out["Random liquid stock"] = dict(n=int(len(day_r)), win=float(day_y.mean()), avg_net=float(day_r.mean()),
                                      net_lo=float(day_r.mean() - 1.96 * day_r.std() / np.sqrt(len(day_r))),
                                      net_hi=float(day_r.mean() + 1.96 * day_r.std() / np.sqrt(len(day_r))))
    vd = vol_decile(d) == 10
    dv = d[vd.values]
    day_r = dv[f"r_{name}"].groupby(dv.date).mean()
    day_y = (dv[f"y_{name}"] == LB.WIN).groupby(dv.date).mean()
    out["Random pick among most volatile 10%"] = dict(n=int(len(day_r)), win=float(day_y.mean()), avg_net=float(day_r.mean()),
                                                     net_lo=float(day_r.mean() - 1.96 * day_r.std() / np.sqrt(len(day_r))),
                                                     net_hi=float(day_r.mean() + 1.96 * day_r.std() / np.sqrt(len(day_r))))
    rules = {
        "Most volatile stock (highest ATR%)": d.atr_pct.values,
        "Hit +3% most often in last 20 days": d.hit3_20.values + 1e-6 * d.atr_pct.values,
        "Yesterday's biggest gainer": d.roc1.values,
        "Most oversold (RSI-2) in an uptrend": np.where(d.c_sma200.values > 0, -d.rsi2.values, np.nan),
        "Highest relative volume": d.relvol.values,
    }
    for k, sc in rules.items():
        out[k] = pick_stats(d, daily_top(d, sc), name)
    return out


def calibration_table(df, pred, name, sel):
    d = df[sel]
    p = pred[sel]
    edges = [0, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, 1.01]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (p >= a) & (p < b)
        n = int(s.sum())
        if n == 0:
            continue
        y = (d[f"y_{name}"].values[s] == LB.WIN)
        m, se, _ = day_clustered_mean(d[f"r_{name}"].values[s], d.date.values[s])
        k = int(y.sum())
        lo, hi = wilson(k, n)
        rows.append(dict(lo=a, hi=min(b, 1.0), n=n, pred=float(p[s].mean()), win=float(k / n), win_lo=lo, win_hi=hi,
                         avg_net=float(m), net_lo=float(ci95(m, se)[0]), net_hi=float(ci95(m, se)[1]),
                         days=int(d.date.values[s].size and np.unique(d.date.values[s]).size)))
    return rows


def main(stage="all"):
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df, fam, nd = load()
    feats = feature_cols(fam)
    print("loaded", df.shape, f"{time.time()-t0:.0f}s", flush=True)
    report = {"generated": pd.Timestamp.utcnow().isoformat(), "main_config": MAIN,
              "data_start": df.date.min(), "data_end": df.date.max(), "test_years": TEST_YEARS,
              "costs": dict(cost_side=LB.COST_SIDE, stop_slip=LB.STOP_SLIP), "target": LB.TARGET,
              "params": PARAMS, "rounds": ROUNDS, "embargo_days": EMBARGO}
    report["base"] = base_rates(df, nd)
    print("base done", f"{time.time()-t0:.0f}s", flush=True)
    st = signal_tests(df, fam, nd)
    report["signals"] = json.loads(st.to_json(orient="records"))
    print("signals done", int(st.confirmed.sum()), "confirmed of", len(st), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(report, open(RES / "research_partial.json", "w"), default=float)

    yrs = year_of(df)
    oos = yrs >= TEST_YEARS[0]

    # ---- main walk-forward models for every stop/horizon config
    preds = {}
    cfg_rows = []
    per_year = {}
    for stop, h in LB.configs():
        name = LB.config_name(stop, h)
        pred, model = fit_predict(df, feats, f"y_{name}", h, nd, subsample=0.25, store_last="wf")
        preds[name] = pred
        # the live model is the most recent walk-forward model, so live scores share the
        # scale of the out-of-sample calibration tables
        model.save_model(str(MODELS / f"{name}_wf.txt"))
        sel = oos & valid_rows(df, nd, h)
        top1 = daily_top(df[sel], pred[sel])
        top5 = daily_top(df[sel], pred[sel], k=5)
        rec = dict(config=name, stop=stop, horizon=h, top1=pick_stats(df, top1, name), top5=pick_stats(df, top5, name))
        rec["baselines"] = baselines(df, name, h, nd, oos)
        sub = df.loc[top1]
        per_year[name] = [dict(year=int(Y), **pick_stats(df, sub.index[sub.date.str[:4].astype(int) == Y], name))
                          for Y in TEST_YEARS if (sub.date.str[:4].astype(int) == Y).any()]
        cfg_rows.append(rec)
        print(name, "top1 win", round(rec["top1"]["win"], 3), "net", round(rec["top1"]["avg_net"] * 100, 3),
              f"{time.time()-t0:.0f}s", flush=True)
    report["configs"] = cfg_rows
    report["per_year"] = per_year
    np.save(DATA_DIR / "oos_preds.npy", np.vstack([preds[LB.config_name(s, h)] for s, h in LB.configs()]))

    # ---- walk-forward config choice: for test year Y, use the stop with the best top-1
    # average net return over the OOS years before Y (same-day horizon is the agreed primary)
    same_day = [LB.config_name(s, 1) for s in LB.STOPS]
    chosen, track = {}, []
    for Y in TEST_YEARS:
        past = [r for r in cfg_rows if r["config"] in same_day]
        scores = {}
        for name in same_day:
            py = [x for x in per_year[name] if x["year"] < Y]
            scores[name] = np.average([x["avg_net"] for x in py], weights=[x["n"] for x in py]) if py else None
        cands = {k: v for k, v in scores.items() if v is not None}
        chosen[Y] = max(cands, key=cands.get) if cands else MAIN
    live_scores = {name: np.average([x["avg_net"] for x in per_year[name]], weights=[x["n"] for x in per_year[name]])
                   for name in same_day}
    live_cfg = max(live_scores, key=live_scores.get)
    wf_idx = []
    for Y in TEST_YEARS:
        name = chosen[Y]
        h = 1
        sel = (yrs == Y) & valid_rows(df, nd, h)
        idx = daily_top(df[sel], preds[name][sel])
        sub = df.loc[idx, ["date", "symbol", f"y_{name}", f"r_{name}"]].copy()
        sub.columns = ["date", "symbol", "y", "r"]
        sub["config"] = name
        sub["p"] = preds[name][df.index.get_indexer(idx)]
        wf_idx.append(sub)
    wf = pd.concat(wf_idx)
    m, se, n = day_clustered_mean(wf.r.values, wf.date.values)
    k = int((wf.y == LB.WIN).sum())
    report["walkforward_strategy"] = dict(chosen_by_year={str(k2): v for k2, v in chosen.items()}, live_config=live_cfg,
                                          live_scores=live_scores, n=int(n), win=k / n, win_ci=wilson(k, n),
                                          avg_net=m, net_ci=ci95(m, se), total_net=float(wf.r.sum()))
    # paired, same-day comparison against naive baselines (after costs)
    edge = {}
    for label, fn in (("random volatile stock (top 10% ATR)", lambda d: (vol_decile(d) == 10).values),
                      ("random liquid stock", lambda d: np.ones(len(d), bool))):
        diffs = []
        for Y in TEST_YEARS:
            name = chosen[Y]
            sel = (yrs == Y) & valid_rows(df, nd, 1)
            d = df[sel]
            base = d[fn(d)].groupby("date")[f"r_{name}"].mean()
            w = wf[wf.config == name].set_index("date").r
            w = w[w.index.str[:4].astype(int) == Y]
            diffs.append((w - base.reindex(w.index)).dropna())
        dd = pd.concat(diffs)
        mu, se = dd.mean(), dd.std() / np.sqrt(len(dd))
        edge[label] = dict(diff=float(mu), lo=float(mu - 1.96 * se), hi=float(mu + 1.96 * se), n=int(len(dd)),
                           p=float(t_pvalue(mu, se)))
    report["edge_vs_baseline"] = dict(detail=edge,
                                      beats_baseline=bool(edge["random volatile stock (top 10% ATR)"]["lo"] > 0))
    # opening gap of the picked stock (known at the open, so usable as a go/no-go check)
    wf2 = wf.merge(df[["date", "symbol", "entry_open", "prev_close"]], on=["date", "symbol"], how="left")
    wf2["gap"] = wf2.entry_open / wf2.prev_close - 1
    buckets = [(-1, -0.02, "gaps down more than 2%"), (-0.02, 0, "opens 0-2% lower"),
               (0, 0.02, "opens 0-2% higher"), (0.02, 1, "gaps up more than 2%")]
    gap_rows = []
    for a, b, lab in buckets:
        s = wf2[(wf2.gap >= a) & (wf2.gap < b)]
        if len(s):
            k = int((s.y == LB.WIN).sum())
            gap_rows.append(dict(bucket=lab, n=int(len(s)), win=k / len(s), win_ci=wilson(k, len(s)),
                                 avg_net=float(s.r.mean())))
    report["gap_buckets"] = gap_rows
    wf["equity"] = (1 + wf.r).cumprod()
    report["equity_curve"] = wf[["date", "symbol", "config", "p", "y", "r", "equity"]].assign(
        y=lambda x: x.y.astype(int)).to_dict(orient="records")
    # equity curve for the volatile-random baseline (same config each year) for comparison
    base_eq = []
    for Y in TEST_YEARS:
        name = chosen[Y]
        sel = (yrs == Y) & valid_rows(df, nd, 1)
        d = df[sel]
        top = d[(vol_decile(d) == 10).values]
        base_eq.append(top.groupby("date")[f"r_{name}"].mean())
    be = pd.concat(base_eq)
    report["baseline_curve"] = [dict(date=k2, equity=float(v)) for k2, v in (1 + be).cumprod().items()]

    # ---- calibration (OOS, all scored rows, live config)
    sel = oos & valid_rows(df, nd, 1)
    report["calibration"] = {name: calibration_table(df, preds[name], name, sel) for name in same_day}
    # calibration among the daily top-1 picks only
    top_cal = {}
    for name in same_day:
        idx = daily_top(df[sel], preds[name][sel])
        s2 = np.zeros(len(df), bool)
        s2[df.index.get_indexer(idx)] = True
        top_cal[name] = calibration_table(df, preds[name], name, s2)
    report["calibration_top1"] = top_cal

    # ---- model quality: AUC per year for the live config
    from sklearn.metrics import roc_auc_score
    aucs = []
    for Y in TEST_YEARS:
        s = (yrs == Y) & valid_rows(df, nd, 1)
        yv = df[f"y_{live_cfg}"].values[s] == LB.WIN
        aucs.append(dict(year=Y, auc=float(roc_auc_score(yv, preds[live_cfg][s]))))
    report["auc_by_year"] = aucs
    json.dump(report, open(RES / "research_partial.json", "w"), default=float)
    print("main models done", f"{time.time()-t0:.0f}s", flush=True)

    # ---- family analysis on the reference config (same day, 2% stop)
    fams = sorted(set(fam.values()))
    vol_feats = [f for f in feats if fam[f] == "volatility"]
    sel = oos & valid_rows(df, nd, 1)
    yv = (df[f"y_{MAIN}"].values == LB.WIN)

    def evaluate(pred, label):
        from sklearn.metrics import log_loss, roc_auc_score
        idx = daily_top(df[sel], pred[sel])
        idx10 = daily_top(df[sel], pred[sel], k=10)
        ll = log_loss(yv[sel], np.clip(pred[sel], 1e-4, 1 - 1e-4))
        return dict(model=label, auc=float(roc_auc_score(yv[sel], pred[sel])), logloss=float(ll),
                    top1=pick_stats(df, idx, MAIN), top10=pick_stats(df, idx10, MAIN))

    fam_rows = []
    pv, _ = fit_predict(df, vol_feats, f"y_{MAIN}", 1, nd, subsample=0.15)
    base_eval = evaluate(pv, "volatility only")
    fam_rows.append(dict(family="volatility (baseline)", kind="baseline", **base_eval))
    full_eval = evaluate(preds[MAIN], "all signals")
    fam_rows.append(dict(family="all signals", kind="full", **full_eval))
    for fname in fams:
        if fname == "volatility":
            continue
        fx = vol_feats + [f for f in feats if fam[f] == fname]
        p_add, _ = fit_predict(df, fx, f"y_{MAIN}", 1, nd, subsample=0.15)
        fx2 = [f for f in feats if fam[f] != fname]
        p_drop, _ = fit_predict(df, fx2, f"y_{MAIN}", 1, nd, subsample=0.15)
        ea, ed = evaluate(p_add, f"volatility + {fname}"), evaluate(p_drop, f"all minus {fname}")
        fam_rows.append(dict(family=fname, kind="add", **ea))
        fam_rows.append(dict(family=fname, kind="drop", **ed))
        print("family", fname, "add auc", round(ea["auc"], 4), "drop auc", round(ed["auc"], 4), f"{time.time()-t0:.0f}s", flush=True)
    report["families"] = fam_rows

    # feature importance (gain) of the final live model
    mdl = lgb.Booster(model_file=str(MODELS / f"{live_cfg}_wf.txt"))
    gain = mdl.feature_importance("gain")
    imp = sorted(zip(mdl.feature_name(), gain), key=lambda x: -x[1])
    tot = sum(gain)
    report["importance"] = [dict(feature=f, family=fam.get(f, "?"), share=float(g / tot)) for f, g in imp]
    fam_share = {}
    for f, g in imp:
        fam_share[fam.get(f, "?")] = fam_share.get(fam.get(f, "?"), 0) + g / tot
    report["importance_by_family"] = fam_share
    json.dump(report, open(RES / "research.json", "w"), default=float, indent=1)
    print("all done", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
