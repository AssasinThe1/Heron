#!/usr/bin/env python3
"""Turn teacher candidates into an execution-verified SFT dataset. LOCAL.

Rejection sampling: run the 32B teacher at temperature ~0.8 with n=4-8 per
question, then keep only completions whose RESULT SET matches gold. The scorer
is the filter -- same set-equality check as score.py.

Why not train on BIRD gold directly: gold is noisy. You have already seen
CreaionDate-style breakage, odd numeric scaling, and quirky groupings, and the
published audits put annotation error rates high enough that training on gold
teaches the noise. Verified survivors give you multiple correct phrasings per
question with none of that.

Coverage is the thing to watch. Questions where NO candidate matches produce
zero training rows, and they are exactly the hard ones (join-heavy,
literal-format) you most need the student to learn. For those, this script can
fall back to gold -- but only gold that executes cleanly, and every fallback row
is tagged `source: "gold"` so you can ablate it with a one-line filter instead of
regenerating the dataset.

Outputs:
  sft_{tag}.jsonl   one JSON object per line: {id, db_id, messages, completion, source}
  sft_{tag}.stats.json   coverage, pass@k, dedup and cap effects

Example
-------
  python build_sft_data.py --candidates candidates_train-fit_fk+samples.json \
      --payload payload_train-fit_fk+samples.json --cap 3 --gold-fallback
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd

from analysis import normalize
from utils import add_common_args, clean_sql, load_split, run_query

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True,
                    help="candidates_{tag}.json from run_inference.py --n K")
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--cap", type=int, default=3,
                    help="max training rows per question after dedup. Without a "
                         "cap, easy questions with 8 near-identical survivors "
                         "dominate the loss.")
    ap.add_argument("--gold-fallback", action="store_true",
                    help="for uncovered questions, use executable gold, tagged")
    ap.add_argument("--majority-filter", action="store_true",
                    help="prefer survivors matching the modal normalized form; "
                         "cheap defence against coincidental set-equality matches")
    ap.add_argument("--keep-truncated", action="store_true",
                    help="do not drop candidates with finish_reason=length "
                         "(you almost never want this)")
    add_common_args(ap)
    args = ap.parse_args()

    payload_rows = json.load(open(args.payload))
    payload = {r["id"]: r for r in payload_rows}
    split = payload_rows[0].get("split", "train-fit")
    variant = payload_rows[0].get("variant", "ddl")
    _, db_map = load_split(split, args.data_root, args.split_dir)
    cands = json.load(open(args.candidates))

    tag = os.path.basename(args.candidates).replace("candidates_", "").replace(".json", "")
    out_path = args.out or f"sft_{tag}.jsonl"

    rows = []
    stats = {"n_questions": len(cands), "covered": 0, "uncovered": 0,
             "gold_fallback": 0, "gold_broken": 0,
             "candidates_seen": 0, "candidates_truncated": 0,
             "candidates_correct": 0, "after_dedup": 0, "after_cap": 0,
             "pass_at_k_by_k": {}}

    hit_at = Counter()      # how many questions were solved by >=1 of first k
    for rec in tqdm(cands, desc="Filtering", unit="q"):
        p = payload.get(rec["id"])
        if p is None:
            continue
        db_path = db_map[p["db_id"]]

        gset, gerr = run_query(db_path, p["gold"], args.timeout)
        if gerr is not None:
            stats["gold_broken"] += 1
            continue          # unwinnable question: no valid target exists

        survivors, first_hit = [], None
        for k, c in enumerate(rec["candidates"], start=1):
            stats["candidates_seen"] += 1
            if c.get("finish_reason") == "length" and not args.keep_truncated:
                stats["candidates_truncated"] += 1
                continue
            sql = clean_sql(c["sql"])
            if not sql:
                continue
            pset, perr = run_query(db_path, sql, args.timeout)
            if perr is None and pset == gset:
                stats["candidates_correct"] += 1
                survivors.append(sql)
                if first_hit is None:
                    first_hit = k
        if first_hit:
            for k in range(first_hit, len(rec["candidates"]) + 1):
                hit_at[k] += 1

        # ---- dedup on canonical form ---------------------------------------
        seen, deduped = set(), []
        for s in survivors:
            key = normalize(s) or " ".join(s.lower().split())
            if key in seen:
                continue
            seen.add(key)
            deduped.append((key, s))
        stats["after_dedup"] += len(deduped)

        # ---- optional majority filter --------------------------------------
        if args.majority_filter and len(deduped) > 1:
            modal = Counter(k for k, _ in deduped).most_common(1)[0][0]
            ordered = ([s for k, s in deduped if k == modal]
                       + [s for k, s in deduped if k != modal])
        else:
            ordered = [s for _, s in deduped]

        # ---- cap ------------------------------------------------------------
        kept = ordered[:args.cap]
        if kept:
            stats["covered"] += 1
            for s in kept:
                rows.append({"id": rec["id"], "db_id": p["db_id"],
                             "messages": p["messages"], "completion": s,
                             "source": "teacher"})
        else:
            stats["uncovered"] += 1
            if args.gold_fallback:
                rows.append({"id": rec["id"], "db_id": p["db_id"],
                             "messages": p["messages"],
                             "completion": clean_sql(p["gold"]),
                             "source": "gold"})
                stats["gold_fallback"] += 1
        stats["after_cap"] += len(kept)

    n_eval = stats["n_questions"] - stats["gold_broken"]
    stats["pass_at_k_by_k"] = {str(k): round(v / max(1, n_eval), 4)
                               for k, v in sorted(hit_at.items())}
    stats["coverage"] = round(stats["covered"] / max(1, n_eval), 4)
    stats["variant"] = variant
    stats["split"] = split
    stats["cap"] = args.cap
    stats["gold_fallback_enabled"] = args.gold_fallback

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(out_path.replace(".jsonl", ".stats.json"), "w"), indent=2)

    print(f"\n{'=' * 58}")
    print(f"split={split}  variant={variant}")
    print(f"questions:            {stats['n_questions']}")
    print(f"broken gold (skipped):{stats['gold_broken']:>6}")
    print(f"covered by teacher:   {stats['covered']:>6}  "
          f"({stats['coverage']:.1%})")
    print(f"uncovered:            {stats['uncovered']:>6}")
    if args.gold_fallback:
        print(f"  gold fallback rows: {stats['gold_fallback']}  "
              "(tagged source='gold', filter to ablate)")
    print(f"\ncandidates seen:      {stats['candidates_seen']}")
    print(f"  truncated dropped:  {stats['candidates_truncated']}")
    print(f"  execution-correct:  {stats['candidates_correct']}")
    print(f"  after dedup:        {stats['after_dedup']}")
    print(f"  after cap={args.cap}:        {stats['after_cap']}")
    print(f"\npass@k: " + "  ".join(
        f"k={k}:{v:.3f}" for k, v in stats["pass_at_k_by_k"].items()))
    print(f"\ntraining rows: {len(rows)} -> {out_path}")

    if rows:
        src = pd.Series([r["source"] for r in rows]).value_counts()
        print(src.to_string())
    if stats["coverage"] < 0.70 and not args.gold_fallback:
        print("\nWARNING: coverage below 70% and no --gold-fallback. You are "
              "deleting the hard tail from training, which is the part the "
              "student most needs.")


if __name__ == "__main__":
    main()
