#!/usr/bin/env python3
"""Measure real tokenized lengths before setting max_model_len / max_seq_len.

Run this after freezing the schema variant and BEFORE training. The failure it
prevents: an example longer than max_seq_len gets silently truncated, the tail of
the schema block disappears, and you train the model to invent columns -- exactly
the hallucination the whole project is trying to fix.

chars/3 estimates are not good enough here. formula_1 and european_football_2
with FK lists and sample values are the long tail, and they are the databases
where truncation would hurt most.

Works on either a prompts/payload JSON or an SFT JSONL. Prints the distribution,
recommends a cutoff, and lists which databases dominate the tail.

Example
-------
  python measure_lengths.py --input sft_train-fit_fk+samples.jsonl \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --budget 12288
"""

import argparse
import json
import os
from collections import Counter

import numpy as np


def read_rows(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path) if l.strip()]
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="prompts_*.json, payload_*.json, or sft_*.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="tokenizer to use -- MUST be the model you will train")
    ap.add_argument("--budget", type=int, default=None,
                    help="if set, report how many rows exceed it and which dbs")
    ap.add_argument("--completion-margin", type=int, default=512,
                    help="headroom reserved for the generated SQL")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = read_rows(args.input)
    lens, comp_lens, dbs = [], [], []
    for r in rows:
        msgs = r["messages"]
        # apply_chat_template is the only honest measurement: the template adds
        # role markers and special tokens, and the count must match what vLLM
        # will produce at serving time.
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
        target = r.get("completion") or r.get("gold") or ""
        comp_lens.append(len(tok(target, add_special_tokens=False)["input_ids"]))
        dbs.append(r.get("db_id", "?"))

    a, c = np.array(lens), np.array(comp_lens)
    total = a + c

    print(f"\n{len(rows)} rows | tokenizer: {args.model}")
    print(f"\nprompt tokens")
    for q in (50, 90, 95, 99, 100):
        print(f"  p{q:<3} {int(np.percentile(a, q)):>7,}")
    print(f"\ncompletion tokens")
    for q in (50, 95, 100):
        print(f"  p{q:<3} {int(np.percentile(c, q)):>7,}")

    rec = int(total.max()) + args.completion_margin
    rec_pow2 = 1 << (rec - 1).bit_length()
    print(f"\nmax prompt+completion observed: {int(total.max()):,}")
    print(f"recommended max_seq_len:        {rec:,}  (next power of 2: {rec_pow2:,})")
    print(f"recommended max_model_len:      {rec_pow2:,}")

    if args.budget:
        over = total > (args.budget - args.completion_margin)
        n_over = int(over.sum())
        print(f"\nat budget {args.budget:,}: {n_over} rows exceed "
              f"({n_over / len(rows):.2%})")
        if n_over:
            print("  dominated by:")
            for db, n in Counter(np.array(dbs)[over]).most_common(6):
                print(f"    {db:<28} {n}")
            print("\n  DO NOT truncate these. Either raise the budget, or add a "
                  "deterministic degradation rule to the frozen variant (fewer "
                  "sample values above a schema-size threshold) and rebuild.")
            ids = [rows[i].get("id") for i in np.where(over)[0]]
            path = os.path.splitext(args.input)[0] + ".overlong_ids.json"
            json.dump(ids, open(path, "w"))
            print(f"  ids written to {path}")


if __name__ == "__main__":
    main()
