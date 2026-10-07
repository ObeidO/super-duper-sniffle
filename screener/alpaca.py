"""Minimal, rate-limited Alpaca client.

Auth is injected by the network proxy, so no keys are sent from here.
Only market-data endpoints plus /v2/assets, /v2/calendar and /v2/clock are used.
No order or account endpoint is ever called.
"""
import threading
import time

import requests

DATA = "https://data.alpaca.markets"
PAPER = "https://paper-api.alpaca.markets"
ALLOWED_PAPER_PATHS = ("/v2/assets", "/v2/calendar", "/v2/clock")

_MAX_PER_MIN = 190  # free plan allows 200/min; keep a margin
_lock = threading.Lock()
_stamps = []
_session = requests.Session()


def _throttle():
    while True:
        with _lock:
            now = time.time()
            while _stamps and now - _stamps[0] > 60:
                _stamps.pop(0)
            if len(_stamps) < _MAX_PER_MIN:
                _stamps.append(now)
                return
            wait = 60 - (now - _stamps[0]) + 0.05
        time.sleep(max(wait, 0.05))


def get(base, path, params=None, retries=6):
    if base == PAPER and not path.startswith(ALLOWED_PAPER_PATHS):
        raise ValueError(f"refusing to call {path}: only assets/calendar/clock allowed")
    url = base + path
    for attempt in range(retries):
        _throttle()
        try:
            r = _session.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt + 1)
            continue
        raise RuntimeError(f"{r.status_code} {url} {params} {r.text[:300]}")
    raise RuntimeError(f"giving up on {url} {params}")


def bars(symbols, timeframe, start, end, adjustment="split", feed="sip"):
    """Yield (symbol, bar_dict) for all pages of a multi-symbol bars request."""
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": adjustment,
        "feed": feed,
        "sort": "asc",
    }
    while True:
        js = get(DATA, "/v2/stocks/bars", params)
        for sym, rows in (js.get("bars") or {}).items():
            for b in rows:
                yield sym, b
        tok = js.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
