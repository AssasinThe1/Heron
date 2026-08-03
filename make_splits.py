#!/usr/bin/env python3
"""Split BIRD train into train-fit / train-val by DATABASE, not by row. LOCAL.

Why by database: dev's databases are disjoint from train's, so the capability
that transfers to dev is reading an unseen schema cold, not knowing any
particular schema. A row-level split leaks schema knowledge -- the model would
have seen formula_1's tables, the fact that raceId lives on five of them, and
the exact spelling of CreaionDate -- and validation accuracy inflates in a way
that will not survive contact with dev.

Selection is stratified by schema complexity so train-val mirrors dev's mix
rather than being accidentally all-easy. Databases are bucketed by table count
(dev spans small 3-table schemas up to formula_1-scale 13-table ones) and
sampled round-robin across buckets.

Writes:
  splits/train_fit.json      {"parent": "train", "ids": [...], "dbs": [...]}
  splits/train_val.json
  splits/train_val_1k.json   fixed stratified subsample for fast iteration
  splits/manifest.json       provenance: seed, counts, db lists, timestamp

Re-running with the same --seed reproduces the split exactly. Do not re-run with
a different seed once you have generated SFT data -- the fit/val boundary would
move and your validation number would become meaningless.
"""

import argparse
import json
import os
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from utils import DEFAULT_DATA_ROOT, load_split


def table_count(db_path):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    except sqlite3.Error:
        return 0
    finally:
        con.close()


def pick_val_dbs(db_stats, n_val, seed):
    """Round-robin across complexity buckets so the val set spans difficulty."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for db, st in db_stats.items():
        t = st["tables"]
        b = "small" if t <= 4 else ("medium" if t <= 8 else "large")
        buckets[b].append(db)
    for b in buckets:
        buckets[b].sort()
        rng.shuffle(buckets[b])

    order = ["large", "medium", "small"]
    chosen, i = [], 0
    while len(chosen) < n_val and any(buckets[b] for b in order):
        b = order[i % len(order)]
        if buckets[b]:
            chosen.append(buckets[b].pop())
        i += 1
    return sorted(chosen)


def stratified_subsample(examples, ids_by_db, n_total, seed):
    """Proportional-by-db subsample, at least 1 per db, deterministic."""
    rng = random.Random(seed)
    dbs = sorted(ids_by_db)
    total = sum(len(ids_by_db[d]) for d in dbs)
    picked = []
    for d in dbs:
        share = max(1, min(round(n_total * len(ids_by_db[d]) / total),
                   int(n_total * 0.15)))
        pool = sorted(ids_by_db[d])
        rng.shuffle(pool)
        picked += pool[:min(share, len(pool))]
    picked = sorted(set(picked))
    if len(picked) > n_total:
        rng.shuffle(picked)
        picked = sorted(picked[:n_total])
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--n-val-dbs", type=int, default=10,
                    help="how many databases to hold out (dev has 11)")
    ap.add_argument("--val-dbs", nargs="*", default=None,
                    help="explicit db_ids to hold out, overrides stratification")
    ap.add_argument("--subsample", type=int, default=1000,
                    help="size of the fixed train-val subsample")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude-dbs", nargs="*", default=[],
                help="db_ids to drop entirely (from audit_gold.py corrupted_dbs)")
    ap.add_argument("--exclude-ids", nargs="*", type=int, default=[],
                help="orig_ids to drop at the question level "
                     "(from audit_gold.py broken_ids)")
    args = ap.parse_args()

    examples, db_map = load_split("train", args.data_root)

    ids_by_db = defaultdict(list)
    for ex in examples:
        ids_by_db[ex["db_id"]].append(ex["orig_id"])

    # Question-level exclusions (broken_ids from audit_gold.py): scattered gold
    # breakage in otherwise-sound databases. Drop the ids before db_stats is
    # built so table/question counts and stratification see only the clean set.
    exclude_ids = set(args.exclude_ids)
    n_removed = 0
    for d in list(ids_by_db):
        kept = [i for i in ids_by_db[d] if i not in exclude_ids]
        n_removed += len(ids_by_db[d]) - len(kept)
        ids_by_db[d] = kept
    print(f"excluded {n_removed} broken question ids "
          f"({len(exclude_ids)} requested)")

    db_stats = {}
    for db in sorted(ids_by_db):
        if db not in db_map:
            print(f"warning: db_id {db!r} in train.json has no .sqlite, skipping")
            continue
        db_stats[db] = {"tables": table_count(db_map[db]),
                        "questions": len(ids_by_db[db])}
    for d in args.exclude_dbs:
        db_stats.pop(d, None)
        ids_by_db.pop(d, None)
    print(f"excluded {len(args.exclude_dbs)} corrupted dbs: {args.exclude_dbs}")

    # A database emptied by id-level exclusions must not survive as a
    # zero-question val candidate.
    for d in [d for d, s in db_stats.items() if s["questions"] == 0]:
        db_stats.pop(d, None)
        ids_by_db.pop(d, None)

    val_dbs = args.val_dbs or pick_val_dbs(db_stats, args.n_val_dbs, args.seed)
    missing = [d for d in val_dbs if d not in db_stats]
    if missing:
        raise SystemExit(f"unknown db_ids: {missing}")
    fit_dbs = sorted(d for d in db_stats if d not in set(val_dbs))

    fit_ids = sorted(i for d in fit_dbs for i in ids_by_db[d])
    val_ids = sorted(i for d in val_dbs for i in ids_by_db[d])
    sub_ids = stratified_subsample(
        examples, {d: ids_by_db[d] for d in val_dbs}, args.subsample, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(name, ids, dbs):
        path = os.path.join(args.out_dir, name)
        json.dump({"parent": "train", "ids": ids, "dbs": dbs},
                  open(path, "w"))
        return path

    dump("train_fit.json", fit_ids, fit_dbs)
    dump("train_val.json", val_ids, val_dbs)
    dump("train_val_1k.json", sub_ids, val_dbs)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "data_root": os.path.abspath(args.data_root),
        "n_fit_questions": len(fit_ids), "n_val_questions": len(val_ids),
        "n_val_1k": len(sub_ids),
        "fit_dbs": fit_dbs, "val_dbs": val_dbs,
        "excluded_dbs": sorted(args.exclude_dbs),
        "excluded_ids": sorted(exclude_ids),
        "db_stats": db_stats,
    }
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"),
              indent=2)

    print(f"train-fit: {len(fit_ids):>5} questions over {len(fit_dbs)} dbs")
    print(f"train-val: {len(val_ids):>5} questions over {len(val_dbs)} dbs")
    print(f"train-val-1k: {len(sub_ids)} questions (fixed subsample)")
    print("\nheld out:")
    for d in val_dbs:
        s = db_stats[d]
        print(f"  {d:<28} {s['tables']:>3} tables  {s['questions']:>4} questions")
    print(f"\n-> {args.out_dir}/  (manifest.json records seed and db lists)")

    overlap = set(fit_dbs) & set(val_dbs)
    assert not overlap, f"LEAK: dbs in both splits: {overlap}"
    print("\nassertion passed: no database appears in both splits")


if __name__ == "__main__":
    main()
