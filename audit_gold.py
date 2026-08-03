#!/usr/bin/env python3
"""Audit gold SQL against the shipped databases, per split. LOCAL.

retail_world in train-val turned out to be Northwind gold queries run against a
reduced, renamed .sqlite -- 199/226 golds reference tables and columns the file
does not contain. Those questions are unwinnable by construction: they drag EX
down and, worse, produce zero-coverage questions during rejection sampling on
train-fit. This is a PACKAGING error (the db file does not match its own golds),
distinct from an annotation error (wrong gold for the right db), and it is
mechanically detectable -- a gold that will not execute is a red flag, and a
database where most golds will not execute is almost certainly the wrong file.

This runs every gold in a split once and aggregates failures by database. Two
tiers, because the right response differs by kind:

  corrupted_dbs  fail_rate >= --fail-threshold. The whole file is wrong; drop
                 the database. Feed straight to make_splits.py --exclude-dbs.
  broken_ids     individual golds that fail in databases NOT wholesale-corrupted.
                 Ordinary annotation noise; exclude the question ids, keep the db.

`partial_dbs` (fail_rate in [--partial-threshold, --fail-threshold)) is reported
so a database with notable-but-not-fatal breakage gets a human glance rather than
silent deletion.

Gold is executed RAW -- no clean_sql. clean_sql splits on the first semicolon,
which is right for model output but wrong for gold: a semicolon inside a string
literal would turn a valid gold into invalid SQL and manufacture a false
positive, the worst failure mode for a tool that decides which databases to
throw away. This also matches score.py, which runs gold raw, so the exclusions
line up with what the scorer counts as unwinnable.

`referenced_but_absent_tables` is a per-db set difference (regex over gold
FROM/JOIN targets, minus the db's actual table names and any CTE names). It is
what made retail_world diagnosable at a glance -- three whole tables the golds
name and the file lacks -- rather than an investigation. Informational only; the
exclusion tiers are driven by fail_rate.

Writes (with --out): {split, fail_threshold, partial_threshold, per_db: {...},
corrupted_dbs: [...], partial_dbs: [...], broken_ids: [...]}.

Example
-------
  python audit_gold.py --split dev   --out audit_dev.json
  python audit_gold.py --split train --out audit_train.json
  python make_splits.py --exclude-dbs $(jq -r '.corrupted_dbs[]' audit_train.json)
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict

from utils import add_common_args, load_split, run_query

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm is optional
    def tqdm(x, **k):
        return x


# FROM/JOIN target: backticked, double-quoted, bracketed, or bare identifier.
# Deliberately a regex, not a parse -- gold that sqlglot chokes on still needs
# auditing, and we only want the table token, not a full AST.
_TBL_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_]\w*)',
    re.IGNORECASE)
# CTE names defined by `WITH x AS (` / `, x AS (` -- subtract these so a common
# table expression is not mistaken for a missing physical table.
_CTE_RE = re.compile(r'(?:\bWITH|,)\s+([A-Za-z_]\w*)\s+AS\s*\(', re.IGNORECASE)


def _referenced_tables(sql):
    sql = sql or ""
    refs = {m.group(1).strip('`"[]').lower() for m in _TBL_RE.finditer(sql)}
    ctes = {m.group(1).lower() for m in _CTE_RE.finditer(sql)}
    return refs - ctes


def _db_tables(db_path):
    con = sqlite3.connect(db_path)
    try:
        return {r[0].lower() for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def audit(split, data_root, split_dir, timeout, fail_threshold, partial_threshold):
    examples, db_map = load_split(split, data_root, split_dir)

    stats = defaultdict(lambda: {"n": 0, "fail_ids": [], "errors": [],
                                 "refs": set()})
    for ex in tqdm(examples, desc=f"auditing {split}", unit="gold"):
        db_id = ex["db_id"]
        st = stats[db_id]
        st["n"] += 1
        st["refs"] |= _referenced_tables(ex["SQL"])
        if db_id not in db_map:
            st["fail_ids"].append(ex.get("orig_id"))
            st["errors"].append([ex.get("orig_id"), "no .sqlite for db_id"])
            continue
        _gset, gerr = run_query(db_map[db_id], ex["SQL"], timeout)  # gold RAW
        if gerr is not None:
            st["fail_ids"].append(ex.get("orig_id"))
            st["errors"].append([ex.get("orig_id"), gerr])

    per_db = {}
    for db_id in sorted(stats):
        st = stats[db_id]
        n_fail = len(st["fail_ids"])
        rate = n_fail / st["n"] if st["n"] else 0.0
        actual = _db_tables(db_map[db_id]) if db_id in db_map else set()
        per_db[db_id] = {
            "n": st["n"], "fail": n_fail, "fail_rate": round(rate, 4),
            "referenced_but_absent_tables": sorted(st["refs"] - actual),
            "sample_errors": st["errors"][:5],
        }

    corrupted = sorted(d for d, s in per_db.items()
                       if s["fail_rate"] >= fail_threshold)
    partial = sorted(d for d, s in per_db.items()
                     if partial_threshold <= s["fail_rate"] < fail_threshold)
    cset = set(corrupted)
    broken_ids = sorted(i for d in stats if d not in cset
                        for i in stats[d]["fail_ids"] if i is not None)

    total = sum(s["n"] for s in per_db.values())
    total_fail = sum(s["fail"] for s in per_db.values())
    print(f"\nsplit={split}  golds={total}  failed={total_fail} "
          f"({total_fail / max(1, total):.1%})  dbs={len(per_db)}")

    print("\ndatabases with any gold failure (rate desc):")
    any_fail = False
    for d in sorted(per_db, key=lambda d: -per_db[d]["fail_rate"]):
        s = per_db[d]
        if s["fail"] == 0:
            continue
        any_fail = True
        tier = ("  <-- CORRUPTED (drop db)" if d in cset
                else "  <-- partial (exclude ids)" if d in set(partial) else "")
        print(f"  {d:<28} {s['fail']:>4}/{s['n']:<4} {s['fail_rate']:>6.1%}{tier}")
        if s["referenced_but_absent_tables"]:
            print(f"      absent tables gold names: "
                  f"{s['referenced_but_absent_tables']}")
    if not any_fail:
        print("  (none -- every gold executed)")

    print(f"\ncorrupted_dbs (>= {fail_threshold:.0%}, drop whole db): {corrupted}")
    print(f"partial_dbs   ([{partial_threshold:.0%}, {fail_threshold:.0%})): "
          f"{partial}")
    print(f"broken_ids    (individual golds outside corrupted dbs): "
          f"{len(broken_ids)}")

    return {"split": split, "fail_threshold": fail_threshold,
            "partial_threshold": partial_threshold, "per_db": per_db,
            "corrupted_dbs": corrupted, "partial_dbs": partial,
            "broken_ids": broken_ids}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train",
                    help="physical or virtual split to audit (train, dev, ...)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--fail-threshold", type=float, default=0.5,
                    help="fail_rate at/above this drops the whole database")
    ap.add_argument("--partial-threshold", type=float, default=0.02,
                    help="fail_rate at/above this (but below --fail-threshold) "
                         "flags the db for a look; its golds go to broken_ids")
    ap.add_argument("--out", default=None)
    add_common_args(ap)
    args = ap.parse_args()

    report = audit(args.split, args.data_root, args.split_dir, args.timeout,
                   args.fail_threshold, args.partial_threshold)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
