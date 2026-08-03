#!/usr/bin/env python3
"""Build chat-format prompts for a text-to-SQL run. LOCAL.

Emits two aligned files keyed on `id` (which equals orig_id in the parent
split, so ids are stable across train / train-fit / train-val):

  prompts_{tag}.json  -> Colab. id + messages only. No gold.
  payload_{tag}.json  -> stays local. id + db_id + question + evidence
                         + messages + gold + the schema variant used.

The payload records `variant`. score.py refuses to score a results file against
a payload whose variant differs from what the model was trained on, because the
schema format is frozen at SFT time and a mismatch silently costs several points.

Examples
--------
  # baseline
  python build_prompts.py --split train-val-1k --variant ddl

  # the four schema arms
  for v in ddl fk samples fk+samples; do
      python build_prompts.py --split train-val-1k --variant $v
  done

  # training-data generation set
  python build_prompts.py --split train-fit --variant fk+samples

  # repair pass: only the questions that failed a previous run
  python build_prompts.py --split train-val-1k --variant fk+samples \
      --repair-from results_train-val-1k_fk+samples.json \
      --repair-payload payload_train-val-1k_fk+samples.json
"""

import argparse
import json
import os

from utils import (SYSTEM, VARIANTS, add_common_args, clean_sql, load_split,
                    load_tables_meta, run_query, user_block)

DEFAULT_SHOT_IDS = [3476, 1213, 8007, 737]


def build_shots(shot_ids, shot_examples, shot_dbs, variant, tables_meta,
                with_schema=True):
    """Fixed demonstrations. Kept for completeness -- few-shot was abandoned
    (net +6 questions on n=1534, McNemar chi2=0.18) at ~4x the prompt tokens."""
    by_orig = {ex["orig_id"]: ex for ex in shot_examples}
    turns = []
    for sid in shot_ids:
        if sid not in by_orig:
            raise IndexError(f"shot id {sid} not present in the shot split")
        ex = by_orig[sid]
        turns.append({"role": "user",
                      "content": user_block(ex, shot_dbs, variant, tables_meta,
                                            with_schema)})
        turns.append({"role": "assistant", "content": ex["SQL"]})
    return turns


# Repair mode is restricted to HARD errors (query did not execute) rather than
# all misses: a repair prompt built around the SQLite error message has
# something concrete to act on, whereas a silent wrong-rows failure offers the
# model no feedback signal to work from. That logic lives inline in main().


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train-val-1k",
                    help="train | dev | train-fit | train-val | train-val-1k")
    ap.add_argument("--variant", default="ddl", choices=VARIANTS,
                    help="schema representation; FROZEN once SFT data is built")
    ap.add_argument("--shot-split", default="train-fit",
                    help="which split demonstrations come from (must not be --split)")
    ap.add_argument("--shot-ids", type=int, nargs="*", default=None)
    ap.add_argument("--n-shots", type=int, default=None,
                    help=f"first N of {DEFAULT_SHOT_IDS}")
    ap.add_argument("--no-shot-schema", dest="shot_schema", action="store_false")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N eval questions (smoke tests)")
    ap.add_argument("--tag", default=None)

    ap.add_argument("--repair-from", default=None,
                    help="results JSON; build prompts only for its errored ids")
    ap.add_argument("--repair-payload", default=None,
                    help="payload JSON matching --repair-from")
    ap.add_argument("--timeout", type=float, default=30.0)
    add_common_args(ap, out_dir=".")
    args = ap.parse_args()

    if args.n_shots is not None:
        shot_ids = DEFAULT_SHOT_IDS[:args.n_shots]
    else:
        shot_ids = args.shot_ids or []
    if shot_ids and args.shot_split == args.split:
        raise SystemExit("--shot-split must differ from --split (leakage)")

    examples, db_map = load_split(args.split, args.data_root, args.split_dir)
    tables_meta = load_tables_meta(args.split, args.data_root)
    if args.limit:
        examples = examples[:args.limit]

    # ---- repair mode: narrow to previously-errored questions ----------------
    repair_notes = {}
    if args.repair_from:
        if not args.repair_payload:
            raise SystemExit("--repair-from requires --repair-payload")
        prev_payload = {r["id"]: r for r in json.load(open(args.repair_payload))}
        prev_results = json.load(open(args.repair_from))
        keep = set()
        for r in prev_results:
            p = prev_payload.get(r["id"])
            if p is None:
                continue
            pred = clean_sql(r["sql"])
            _, err = run_query(db_map[p["db_id"]], pred, args.timeout)
            if err:
                keep.add(r["id"])
                repair_notes[r["id"]] = (pred, err)
        examples = [ex for ex in examples if ex["orig_id"] in keep]
        print(f"repair mode: {len(examples)} previously-errored questions")

    shots = []
    if shot_ids:
        shot_examples, shot_dbs = load_split(args.shot_split, args.data_root,
                                             args.split_dir)
        shot_meta = load_tables_meta(args.shot_split, args.data_root)
        shots = build_shots(shot_ids, shot_examples, shot_dbs, args.variant,
                            shot_meta, args.shot_schema)

    payload, to_colab = [], []
    for ex in examples:
        uid = ex["orig_id"]
        msgs = [{"role": "system", "content": SYSTEM}] + shots
        msgs.append({"role": "user",
                     "content": user_block(ex, db_map, args.variant, tables_meta)})
        if uid in repair_notes:
            bad, err = repair_notes[uid]
            # The error string is the whole point of a repair pass: SQLite tells
            # you exactly which identifier is wrong.
            msgs.append({"role": "assistant", "content": bad})
            msgs.append({"role": "user", "content":
                         f"That query failed with the SQLite error: {err}\n"
                         "Fix it using only tables and columns that appear in "
                         "the schema above, copied exactly. Respond with ONLY "
                         "the corrected SQL."})
        payload.append({"id": uid, "db_id": ex["db_id"],
                        "question": ex["question"],
                        "evidence": ex.get("evidence", ""),
                        "split": args.split, "variant": args.variant,
                        "messages": msgs, "gold": ex["SQL"]})
        to_colab.append({"id": uid, "messages": msgs})

    shot_tag = "fewshot" if shot_ids else "zeroshot"
    base = f"{args.split}_{args.variant}"
    if shot_ids:
        base += f"_{shot_tag}"
    if args.repair_from:
        base += "_repair"
    tag = args.tag or base

    os.makedirs(args.out_dir, exist_ok=True)
    pp = os.path.join(args.out_dir, f"prompts_{tag}.json")
    yp = os.path.join(args.out_dir, f"payload_{tag}.json")
    json.dump(to_colab, open(pp, "w"))
    json.dump(payload, open(yp, "w"))

    longest = max((len(json.dumps(x["messages"])) for x in to_colab), default=0)
    mean = (sum(len(json.dumps(x["messages"])) for x in to_colab)
            / max(1, len(to_colab)))
    print(f"{len(to_colab)} prompts -> {pp}")
    print(f"{len(payload)} payload rows -> {yp}")
    print(f"variant={args.variant}  shots={len(shots) // 2}")
    print(f"prompt chars: mean {mean:,.0f} (~{mean / 3:,.0f} tok), "
          f"max {longest:,} (~{longest / 3:,.0f} tok)")
    print("NOTE: chars/3 is a rough proxy. Use measure_lengths.py for real "
          "tokenizer counts before setting max_model_len or max_seq_len.")


if __name__ == "__main__":
    main()
