"""Assemble results/*.json into the single-file dashboard (results/dashboard.html)."""
import json
import math

import pandas as pd

import labels as LB
from universe import DATA_DIR

RES = DATA_DIR.parent / "results"
HERE = DATA_DIR.parent / "screener"


def clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else round(o, 6)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def breakeven(stop):
    win = (1 + LB.TARGET) * (1 - LB.COST_SIDE) / (1 + LB.COST_SIDE) - 1
    loss = (1 - stop) * (1 - LB.COST_SIDE - LB.STOP_SLIP) / (1 + LB.COST_SIDE) - 1
    return -loss / (win - loss)


def main():
    research = json.load(open(RES / "research.json"))
    today = json.load(open(RES / "today.json"))
    research["breakeven"] = {LB.config_name(s, h): breakeven(s) for s, h in LB.configs()}
    cf = research["configs"]
    best_by_h = {}
    for h in LB.HORIZONS:
        rows = [c for c in cf if c["horizon"] == h]
        if not rows:
            continue
        b = max(rows, key=lambda c: c["top1"]["avg_net"])
        best_by_h[h] = b
    parts = [f"{'same day' if h == 1 else str(h) + '-day hold'}: best {b['stop']*100:.1f}% stop, hit {b['top1']['win']*100:.1f}%, "
             f"avg net {b['top1']['avg_net']*100:+.3f}%" for h, b in best_by_h.items()]
    research["horizon_summary"] = "Best stop for each holding period (model #1 pick, out-of-sample): " + "; ".join(parts) + "."
    log = pd.read_csv(RES / "picks_log.csv", dtype={"date": str}).to_dict(orient="records") if (RES / "picks_log.csv").exists() else []
    audit = json.load(open(RES / "audit_truncation.json")) if (RES / "audit_truncation.json").exists() else {}
    review = json.load(open(RES / "review.json")) if (RES / "review.json").exists() else None
    # keep the page light: equity curve rows are small; signals table ~200 rows
    data = clean(dict(research=research, today=today, log=log, audit=audit, review=review))
    html = open(HERE / "dashboard_template.html").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (RES / "dashboard.html").write_text(html)
    print("dashboard", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
