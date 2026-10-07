"""Download daily bars (split-adjusted and raw) for every candidate symbol + SPY."""
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from alpaca import bars
from universe import DATA_DIR, candidate_symbols

START = "2016-01-01"


def fetch_batch(i, syms, adjustment, end, outdir):
    out = outdir / f"b{i:05d}.parquet"
    if out.exists():
        return i, "cached"
    syms = [s for s in syms if not s[0].isdigit()]  # CUSIP-style codes (CVRs, escrows), not stocks
    rows = []
    while syms:
        try:
            rows = [
                (s, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw"), b.get("n"))
                for s, b in bars(syms, "1Day", START, end, adjustment=adjustment)
            ]
            break
        except RuntimeError as e:
            m = re.search(r"invalid symbol: ([^\"]+)", str(e))
            if not m or m.group(1) not in syms:
                raise
            syms.remove(m.group(1))
    df = pd.DataFrame(rows, columns=["symbol", "date", "o", "h", "l", "c", "v", "vw", "n"])
    df.to_parquet(out, index=False)
    return i, len(df)


def main(end, batch=100, workers=6):
    syms = sorted(candidate_symbols()) + ["SPY"]
    batches = [syms[i : i + batch] for i in range(0, len(syms), batch)]
    print(f"{len(syms)} symbols in {len(batches)} batches", flush=True)
    for adj in ("split", "raw"):
        outdir = DATA_DIR / f"daily_{adj}"
        outdir.mkdir(exist_ok=True)
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(fetch_batch, i, b, adj, end, outdir) for i, b in enumerate(batches)]
            for n, f in enumerate(as_completed(futs)):
                i, res = f.result()
                if n % 20 == 0:
                    print(adj, n, "/", len(batches), "batch", i, res, flush=True)
        df = pd.concat(pd.read_parquet(p) for p in sorted(outdir.glob("b*.parquet")))
        df.to_parquet(DATA_DIR / f"daily_{adj}.parquet", index=False)
        print(adj, "done", df.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-10-06T23:00:00Z")
