"""Today's trade of the day (run before the US open).

Steps: confirm today is a trading session -> download recent daily bars -> compute the same
point-in-time features used in research -> score with the walk-forward model -> look up the
out-of-sample calibration -> apply the pre-set bar -> verify current pre-market prices ->
grade earlier picks with intraday bars -> write results/today.json and results/picks_log.csv.
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
import labels as LB
from alpaca import DATA, PAPER, bars, get
from universe import DATA_DIR, tradable_symbols

ROOT = DATA_DIR.parent
RES = ROOT / "results"
LOG = RES / "picks_log.csv"
TOP_N = 15


def now_utc():
    return datetime.now(timezone.utc)


def session_info():
    clock = get(PAPER, "/v2/clock")
    today_ny = pd.Timestamp(clock["timestamp"]).tz_convert("America/New_York").date()
    cal = get(PAPER, "/v2/calendar", {"start": str(today_ny - timedelta(days=14)), "end": str(today_ny + timedelta(days=10))})
    days = [c["date"] for c in cal]
    is_session = str(today_ny) in days
    prev = [d for d in days if d < str(today_ny)][-1]
    nxt = str(today_ny) if is_session else [d for d in days if d > str(today_ny)][0]
    return clock, str(today_ny), is_session, prev, nxt, days


def fetch_recent(symbols, start, end, adjustment):
    rows = []
    for i in range(0, len(symbols), 100):
        try:
            rows += [(s, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"])
                     for s, b in bars(symbols[i:i + 100], "1Day", start, end, adjustment=adjustment)]
        except RuntimeError as e:
            # drop a symbol Alpaca rejects and retry the rest of the chunk one by one
            for s in symbols[i:i + 100]:
                try:
                    rows += [(s2, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"])
                             for s2, b in bars([s], "1Day", start, end, adjustment=adjustment)]
                except RuntimeError:
                    pass
    return pd.DataFrame(rows, columns=["symbol", "date", "o", "h", "l", "c", "v"])


def latest_prices(symbols, day):
    """Most recent pre-market / intraday trade for each symbol today.

    SIP (all US venues) bars are only available up to 15 minutes ago on the free plan, so
    we take SIP up to now-16min and top up with IEX for the last 15 minutes."""
    out = {}
    end_sip = (now_utc() - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = f"{day}T08:00:00Z"
    for feed, end in (("sip", end_sip), ("iex", now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"))):
        if end <= start:
            continue
        try:
            for s, b in bars(symbols, "1Min", start, end, adjustment="raw", feed=feed):
                if s not in out or b["t"] >= out[s]["t"]:
                    out[s] = {"t": b["t"], "price": b["c"], "feed": feed}
        except RuntimeError:
            pass
    return out


def grade_pick(row):
    """Grade a logged pick with the exact research rules (5-minute bars, same-day exit)."""
    day, sym, stop = row["date"], row["symbol"], float(row["stop"])
    d = list(bars([sym], "1Day", day, day, adjustment="raw"))
    if not d:
        return None
    b = d[0][1]
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    T, S = o * (1 + LB.TARGET), o * (1 - stop)
    if h >= T and l <= S:
        res = LB.LOSS
        for _, x in bars([sym], "5Min", f"{day}T13:00:00Z", f"{day}T21:00:00Z", adjustment="raw"):
            t = pd.Timestamp(x["t"]).tz_convert("America/New_York")
            if not (570 <= t.hour * 60 + t.minute < 960):
                continue
            ht, hs = x["h"] >= T, x["l"] <= S
            if ht or hs:
                res = LB.WIN if (ht and not hs) else LB.LOSS
                break
    elif h >= T:
        res = LB.WIN
    elif l <= S:
        res = LB.LOSS
    else:
        res = LB.TIMEOUT
    px = T if res == LB.WIN else S if res == LB.LOSS else c
    net = float(LB.net_return(np.array([res]), np.array([px / o]))[0])
    return dict(actual_open=o, actual_high=h, actual_low=l, actual_close=c,
                outcome={LB.WIN: "hit +3%", LB.LOSS: "stopped out", LB.TIMEOUT: "closed flat/exit at close"}[res],
                hit=int(res == LB.WIN), net_return=net)


def grade_log(today):
    if not LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(LOG, dtype={"date": str})
    for i, r in log.iterrows():
        if r.get("status") == "pending" and r["date"] < today and r["kind"] == "trade":
            g = grade_pick(r)
            if g:
                for k, v in g.items():
                    log.loc[i, k] = v
                log.loc[i, "status"] = "graded"
        elif r.get("status") == "pending" and r["date"] < today and r["kind"] == "no-trade":
            # still grade the best candidate so the 'no trade' calls can be checked too
            g = grade_pick(r)
            if g:
                for k, v in g.items():
                    log.loc[i, k] = v
                log.loc[i, "status"] = "graded"
    log.to_csv(LOG, index=False)
    return log


def calib_lookup(table, p):
    for row in table:
        if row["lo"] <= p < row["hi"] or (p >= row["hi"] and row is table[-1]):
            return row
    return None


def main():
    research = json.load(open(RES / "research.json"))
    wf = research["walkforward_strategy"]
    cfg = wf["live_config"]
    stop = float(cfg.split("_")[0][1:]) / 1000
    clock, today, is_session, prev, nxt, days = session_info()
    trade_day = nxt
    print("today", today, "session" if is_session else "no session", "previous session", prev, "trade day", trade_day)

    cand = tradable_symbols()
    active = sorted(s for s, v in cand.items() if v["status"] == "active") + ["SPY"]
    end = (now_utc() - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (pd.Timestamp(prev) - pd.Timedelta(days=700)).strftime("%Y-%m-%d")
    sp = fetch_recent(active, start, end, "split")
    rw = fetch_recent(active, (pd.Timestamp(prev) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), end, "raw")
    sp = sp[sp.date <= prev]
    rw = rw[rw.date <= prev]
    last_bar = sp.loc[sp.symbol == "SPY", "date"].max()
    if last_bar != prev:
        raise SystemExit(f"latest SPY bar {last_bar} != previous session {prev}; data not ready")
    dates = sorted(sp.loc[sp.symbol == "SPY", "date"].unique().tolist()) + [trade_day]
    W = {k: sp.pivot(index="date", columns="symbol", values=k).reindex(dates) for k in "ohlcv"}
    craw = rw.pivot(index="date", columns="symbol", values="c").reindex(dates)
    craw = craw.reindex(columns=W["c"].columns).ffill(limit=0)
    cols = [c for c in W["c"].columns if c != "SPY"]
    # raw close for the price filter: only the most recent ~10 days are fetched raw; the
    # features themselves only need split-adjusted data, so fill older raw rows from adjusted.
    craw = craw[cols].where(craw[cols].notna(), W["c"][cols])
    spy = pd.DataFrame({k: W[k]["SPY"] for k in "ohlc"})
    feats, liq = F.build(*(W[k][cols] for k in ("o", "h", "l", "c", "v")), craw, spy)
    liq_prev = liq.shift(1, fill_value=False).astype(bool).loc[trade_day]
    row = {n: fr.shift(1).loc[trade_day] for n, fr in feats.items()}
    X = pd.DataFrame(row)
    X["entry_weekday"] = float(pd.Timestamp(trade_day).dayofweek)
    X = X[liq_prev.reindex(X.index).fillna(False).values & X["atr_pct"].notna().values]

    model = lgb.Booster(model_file=str(ROOT / "models" / f"{cfg}.txt"))
    X = X[model.feature_name()]
    p = pd.Series(model.predict(X.values), index=X.index).sort_values(ascending=False)
    calib = research["calibration"][cfg]
    calib_top = research["calibration_top1"][cfg]
    prev_close = W["c"][cols].loc[prev]
    top = p.head(TOP_N)
    live_px = latest_prices(list(top.index), today) if is_session else {}

    # ---- the bar, fixed in advance:
    #  (1) setups scoring like this must have made money after costs out-of-sample with the
    #      95% interval above zero, AND
    #  (2) the walk-forward strategy must beat the 'random volatile stock' baseline after costs.
    edge = research["edge_vs_baseline"]
    cands = []
    for sym, score in top.items():
        cal = calib_lookup(calib, score) or {}
        calt = calib_lookup(calib_top, score) or {}
        info = cand.get(sym, {})
        lp = live_px.get(sym)
        ref = lp["price"] if lp else float(prev_close[sym])
        reasons = []
        if not cal or cal.get("net_lo", -1) <= 0:
            reasons.append("setups scoring like this did not reliably make money after costs in past out-of-sample years")
        if not edge["beats_baseline"]:
            reasons.append("the model has not shown a reliable edge over simply picking a random volatile stock")
        fx = X.loc[sym]
        cands.append(dict(symbol=sym, name=info.get("name", ""), score=float(score),
                          prob=cal.get("win"), prob_lo=cal.get("win_lo"), prob_hi=cal.get("win_hi"), n=cal.get("n"),
                          avg_net=cal.get("avg_net"), net_lo=cal.get("net_lo"), net_hi=cal.get("net_hi"),
                          top_prob=calt.get("win"), top_n=calt.get("n"),
                          prev_close=float(prev_close[sym]), live_price=lp["price"] if lp else None,
                          live_time=lp["t"] if lp else None, live_feed=lp["feed"] if lp else None,
                          est_entry=ref, est_target=ref * (1 + LB.TARGET), est_stop=ref * (1 - stop),
                          clears_bar=not reasons, reasons=reasons,
                          features={k: (None if pd.isna(fx[k]) else float(fx[k])) for k in
                                    ("atr_pct", "hit3_20", "rsi14", "rsi2", "adx14", "relvol", "rs20", "c_sma50",
                                     "c_sma200", "resist_room", "fib_retr", "fib_golden", "ew_w3", "gap_last", "roc1")}))
    best = cands[0]
    decision = "trade" if best["clears_bar"] else "no-trade"
    out = dict(generated=now_utc().isoformat(), today=today, is_session=is_session, trade_day=trade_day,
               data_through=prev, config=cfg, stop=stop, horizon=1, target=LB.TARGET, decision=decision,
               best=best, candidates=cands, universe_size=int(len(X)),
               market={k: float(feats[k].shift(1).loc[trade_day].iloc[0]) for k in
                       ("spy_c_sma50", "spy_c_sma200", "spy_ret5", "spy_rv20", "breadth50")},
               clock=clock)
    json.dump(out, open(RES / "today.json", "w"), indent=1, default=float)

    log = grade_log(today) if LOG.exists() else pd.DataFrame()
    if is_session and (log.empty or not ((log.date == trade_day)).any()):
        rec = dict(date=trade_day, kind=decision, symbol=best["symbol"], config=cfg, stop=stop,
                   predicted_prob=best["prob"], calib_n=best["n"], score=best["score"],
                   ref_price=best["est_entry"], status="pending")
        log = pd.concat([log, pd.DataFrame([rec])], ignore_index=True)
        log.to_csv(LOG, index=False)
    print(json.dumps({k: out[k] for k in ("trade_day", "decision", "config")}, indent=1))
    print("best", best["symbol"], best["name"], "score", round(best["score"], 3), "prob", best["prob"], "n", best["n"],
          "net", best["avg_net"], best["net_lo"], "clears", best["clears_bar"])


if __name__ == "__main__":
    main()
