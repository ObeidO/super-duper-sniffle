"""Candidate symbol list: NYSE/Nasdaq common stocks, active and (where Alpaca lists them) inactive."""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NON_STOCK = re.compile(
    r"\b(ETF|ETN|ETP|Fund|Funds|Index|Portfolio|ProShares|iShares|Direxion|GraniteShares|"
    r"Invesco|SPDR|Vanguard|WisdomTree|VanEck|Leveraged|Inverse|Bull|Bear|2X|3X|"
    r"Warrants?|Units?|Rights?|Preferred|Pfd|Notes?|Debentures?|Bonds?|due \d{4}|"
    r"Perpetual|Cumulative|Subordinated|Trust Units|Royalty Trust|Depositary Shares, each)\b|%",
    re.IGNORECASE,
)
FOREIGN_OTC = re.compile(r"^[A-Z]{4}[FY]$")


def load_assets():
    out = []
    for f in ("assets_active.json", "assets_inactive.json"):
        out += json.load(open(DATA_DIR / f))
    return out


def candidate_symbols():
    keep = {}
    for a in load_assets():
        sym, name, exch, status = a["symbol"], a.get("name") or "", a["exchange"], a["status"]
        if a.get("class") != "us_equity" or "/" in sym or " " in sym:
            continue
        if NON_STOCK.search(name):
            continue
        if status == "active" and exch not in ("NYSE", "NASDAQ"):
            continue
        if status == "inactive":
            # Delisted names often end up tagged OTC; keep them (the point-in-time
            # liquidity filter decides if they were tradable), but skip foreign ordinaries/ADRs.
            if exch not in ("NYSE", "NASDAQ", "OTC"):
                continue
            if exch == "OTC" and FOREIGN_OTC.match(sym):
                continue
        keep.setdefault(sym, {"symbol": sym, "name": name, "exchange": exch, "status": status})
    # Alpaca tags some delisted names "XYZ_DELISTED"; bars live under the plain symbol.
    for sym in [s for s in keep if s.endswith("_DELISTED")]:
        info = keep.pop(sym)
        base = sym[: -len("_DELISTED")]
        if base.isalpha() and base not in keep:
            keep[base] = {**info, "symbol": base}
    return {s: v for s, v in keep.items() if s.replace(".", "").isalnum()}


def tradable_symbols():
    """Candidates minus CUSIP-style codes (CVRs, escrow shares) that are not stocks."""
    return {s: v for s, v in candidate_symbols().items() if not s[0].isdigit()}


if __name__ == "__main__":
    c = candidate_symbols()
    import collections
    print(len(c), collections.Counter((v["status"], v["exchange"]) for v in c.values()))
