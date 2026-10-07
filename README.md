# Trade of the Day screener

A US stock screener that proposes one long trade per session (buy at the open, sell at +3%
or at a stop, exit by the close), together with an honestly measured, out-of-sample
probability of success. When nothing clears a pre-set bar it says **no trade** and shows the
best candidate and why it fell short.

This is a statistical tool, not financial advice.

## Pipeline

All scripts live in `screener/` and are run from there. Market data comes from Alpaca
(`data.alpaca.markets`); `paper-api.alpaca.markets` is used only for `/v2/assets`,
`/v2/calendar` and `/v2/clock`. No order or account endpoint is ever called. Auth is added
by the network proxy, so no keys are stored here.

| Step | Command | Output |
|---|---|---|
| Download daily bars 2016→ (split-adjusted + raw) | `python fetch_daily.py` | `data/daily_*.parquet` |
| Point-in-time features + universe | `python dataset.py features` | `data/features.parquet` |
| Find days needing intraday resolution | `python dataset.py needs` | `data/intraday_needs.parquet` |
| Download 5-minute bars for those days | `python fetch_intraday.py` | `data/intraday_bars.pkl` |
| Outcomes for every stop / holding period | `python dataset.py labels` | `data/labels.parquet` |
| Research: base rates, signal tests, walk-forward models, baselines, calibration | `python research.py` | `results/research.json`, `models/*.txt` |
| Look-ahead audit | `python audit_lookahead.py` | `results/audit_truncation.json` |
| Today's pick (run before the open) | `python live.py` | `results/today.json`, `results/picks_log.csv` |
| Dashboard | `python build_dashboard.py` | `results/dashboard.html` |

`live.py` needs only `models/`, `results/research.json` and the asset lists; it downloads
the ~2 years of daily bars it needs on each run, so it works in a fresh environment.

## Key choices

- Universe: NYSE/Nasdaq common stocks (ETFs, warrants, units, preferreds excluded by name),
  raw close ≥ $5 and 20-day average dollar volume ≥ $20M, both as of the previous close.
- Entry at the official open; target +3%; stops tested at 1%, 1.5%, 2% and 3%; holding
  periods of 1, 2, 3 and 5 days. Costs: 0.10% per side, plus 0.10% extra on stop exits.
- Days where the daily bar touched both levels are decided with 5-minute bars; a tie inside
  one bar counts as a loss.
- Model: LightGBM with fixed settings, walk-forward by calendar year (test years 2019→),
  5-day embargo between training and test. The live stop distance is chosen by the same
  walk-forward rule using only out-of-sample years before the decision.
