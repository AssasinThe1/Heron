#!/usr/bin/env python3
"""Derive the frozen profiler rules on TRAIN, and show the cardinality histogram
that justifies K. LOCAL, train-only.

The profiler (utils._samples_block) lists a column's values only when the column
has <= K distinct values -- an enum worth spelling out -- and skips it otherwise.
K, the per-prompt value budget, the scan cap and truncation are *rules*: global,
frozen, derived here on train alone. This script exists so those numbers come
from the data rather than a guess, and so the derivation is separated from the
per-database profiler by a module boundary: this script never emits per-column
decisions, and the profiler never imports a split file. Only a JSON of scalars
ever crosses from train into the pipeline -- that is the leakage guarantee, the
structural version of the two-looks rule.

It derives on the databases of a train-only split (default train-fit, which is
disjoint from train-val and from dev), so nothing here ever reads a held-out or
eval database.

Usage
-----
  # just look at the histogram and the recommendation (freezes nothing)
  python derive_profile_rules.py

  # freeze the rules once you've chosen K and the budget
  python derive_profile_rules.py --k 12 --budget 60 --emit profile_rules.json
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from utils import (DEFAULT_DATA_ROOT, _bounded_distinct, _col_kind, _ID_RE,
                   load_split)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

HIST_CAP = 200          # count distinct only this far; beyond is "high-card"
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20),
           (21, 50), (51, 100), (101, HIST_CAP)]
K_CANDIDATES = [5, 8, 10, 12, 15, 20]


def bucket_label(n):
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return f"{HIST_CAP}+"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train-fit",
                    help="train-only split whose databases to derive on")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--scan-cap", type=int, default=20000,
                    help="max rows scanned per column (bounds build cost)")
    ap.add_argument("--k", type=int, default=None,
                    help="enum cutoff to freeze; omit to only show the histogram")
    ap.add_argument("--budget", type=int, default=60,
                    help="per-prompt value budget to freeze")
    ap.add_argument("--str-trunc", type=int, default=20)
    ap.add_argument("--emit", default=None,
                    help="path to write the frozen rules JSON (requires --k)")
    args = ap.parse_args()

    examples, db_map = load_split(args.split, args.data_root)
    dbs = sorted({ex["db_id"] for ex in examples if ex["db_id"] in db_map})

    # distinct-count histogram over listable (non-id) columns, plus a per-kind
    # split and, per candidate K, how many columns would qualify as enums and
    # how many values that implies per database (to sanity-check the budget).
    hist = Counter()
    kind_totals = Counter()
    kind_enum = {k: Counter() for k in K_CANDIDATES}   # kind -> qualifying cols
    values_per_db = {k: [] for k in K_CANDIDATES}
    n_cols = n_id = 0

    for db_id in tqdm(dbs, desc=f"profiling {args.split}", unit="db"):
        con = sqlite3.connect(db_map[db_id])
        try:
            tables = [x[0] for x in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            db_vals = {k: 0 for k in K_CANDIDATES}
            for t in tables:
                try:
                    cols = list(con.execute(f'PRAGMA table_info("{t}")'))
                except sqlite3.Error:
                    continue
                for c in cols:
                    cname, ctype = c[1], c[2]
                    if _ID_RE.search(cname):
                        n_id += 1
                        continue
                    kind = _col_kind(cname, ctype)
                    if kind == "date":
                        n_cols += 1
                        kind_totals["date"] += 1
                        continue
                    vals = _bounded_distinct(con, t, cname, args.scan_cap,
                                             HIST_CAP + 1)
                    if vals is None:
                        continue
                    card = len(vals)
                    n_cols += 1
                    kind_totals[kind] += 1
                    hist[bucket_label(card)] += 1
                    for k in K_CANDIDATES:
                        if card <= k:
                            kind_enum[k][kind] += 1
                            db_vals[k] += card
            for k in K_CANDIDATES:
                values_per_db[k].append(db_vals[k])
        finally:
            con.close()

    # ---- report --------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"split={args.split}  dbs={len(dbs)}  "
          f"listable cols={n_cols}  id cols skipped={n_id}")
    print(f"{'=' * 60}")

    print("\ndistinct-count histogram (non-id, non-date columns):")
    order = [bucket_label(lo) for lo, _ in BUCKETS] + [f"{HIST_CAP}+"]
    total = max(1, sum(hist.values()))
    cum = 0
    for b in order:
        n = hist.get(b, 0)
        cum += n
        bar = "#" * int(40 * n / total)
        print(f"  {b:>8}  {n:>6}  {n/total:>6.1%}  cum {cum/total:>6.1%}  {bar}")

    print(f"\nby kind (non-id):  " +
          "  ".join(f"{k}={v}" for k, v in kind_totals.most_common()))

    print("\nif K = ... :   enums    (text/numeric/date get 1)    "
          "median vals/db   max vals/db")
    for k in K_CANDIDATES:
        n_enum = sum(kind_enum[k].values())
        vpd = sorted(values_per_db[k])
        med = vpd[len(vpd) // 2] if vpd else 0
        mx = vpd[-1] if vpd else 0
        share = n_enum / max(1, n_cols)
        print(f"  K={k:<3}  {n_enum:>6} cols ({share:>5.1%})   "
              f"text={kind_enum[k]['text']:<5} num={kind_enum[k]['numeric']:<5}"
              f"       {med:>6}          {mx:>6}")

    # recommendation: the knee -- smallest K past the point where extra distinct
    # values stop being enum-like. We flag where cumulative coverage crosses ~90%
    # of everything that will ever qualify at K=20, as a starting point.
    print("\nnote: 'max vals/db' is the number the per-prompt budget must cover "
          "for the widest schema;\n      pick --budget at or above it, or accept "
          "that the widest few schemas get truncated.")

    if args.k is None:
        print("\n(no --k given: nothing frozen. Re-run with --k N --budget M "
              "--emit profile_rules.json to freeze.)")
        return

    rules = {
        "k": args.k,
        "value_budget": args.budget,
        "scan_cap": args.scan_cap,
        "str_trunc": args.str_trunc,
        "provenance": {
            "derived_on": args.split,
            "n_dbs": len(dbs),
            "n_listable_columns": n_cols,
            "created": datetime.now(timezone.utc).isoformat(),
            "note": ("Frozen profiler rules. Derived on a train-only split. "
                     "Global and fixed: do NOT tune per database -- that is "
                     "leakage. The per-database VALUES are computed by "
                     "utils._samples_block at build time, including on dev."),
        },
    }
    out = args.emit or "profile_rules.json"
    json.dump(rules, open(out, "w"), indent=2)
    print(f"\nfroze rules: K={args.k}  budget={args.budget}  "
          f"scan_cap={args.scan_cap}  -> {out}")


if __name__ == "__main__":
    main()
