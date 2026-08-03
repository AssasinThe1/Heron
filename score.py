#!/usr/bin/env python3
"""Score results by execution accuracy, with full failure taxonomy. LOCAL.

Execution accuracy = predicted and gold queries return the same SET of result
tuples. Set equality, not string match -- BIRD's metric.

Outputs, in order of usefulness:
  * EX overall, and EX on winnable questions only (excluding broken gold)
  * hard-error breakdown, with missing_column split into wrong_table vs bad_name
  * result-shape triage over the silent failures (the ~89% that execute)
  * optional AST-diff category (needs sqlglot)
  * cross-tab of shape x AST -- the cell counts tell you which arm to build
  * EX with the identifier repair guard applied, reported separately
  * per-query CSV for sample_errors.py

Example
-------
  python score.py --payload payload_train-val-1k_fk+samples.json \
                  --results results_train-val-1k_fk+samples.json \
                  --ast --guard --out-csv errors_fk+samples.csv
"""

import argparse
import json
import os

import pandas as pd

from analysis import ast_kind, err_kind, missing_column_subkind, shape_kind
from utils import (add_common_args, clean_sql, load_split, run_query,
                    schema_columns)

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm is optional
    def tqdm(x, **k):
        return x


def all_columns_lower(db_map, db_id, cache):
    if db_id not in cache:
        cols = set()
        for _t, cs in schema_columns(db_id, db_map).items():
            cols |= {c.lower() for c in cs}
        cache[db_id] = cols
    return cache[db_id]


def score(payload_path, results_path, data_root, split_dir, timeout,
          use_ast=False, use_guard=False, out_csv=None, show=10,
          expect_variant=None):
    payload_rows = json.load(open(payload_path))
    payload = {r["id"]: r for r in payload_rows}
    results = json.load(open(results_path))

    split = payload_rows[0].get("split", "dev")
    variant = payload_rows[0].get("variant", "ddl")
    if expect_variant and variant != expect_variant:
        raise SystemExit(
            f"variant mismatch: payload was built with {variant!r} but "
            f"--expect-variant is {expect_variant!r}. The schema format is "
            "frozen at SFT time; scoring a tuned model against a different "
            "format silently costs several points."
        )
    _, db_map = load_split(split, data_root, split_dir)

    col_cache, schema_cache = {}, {}
    rows = []
    for r in tqdm(results, desc="Evaluating", unit="query"):
        p = payload.get(r["id"])
        if p is None:
            continue
        db_path = db_map[p["db_id"]]
        pred = clean_sql(r["sql"])

        pset, perr = run_query(db_path, pred, timeout)
        gset, gerr = run_query(db_path, p["gold"], timeout)
        correct = int(perr is None and gerr is None and pset == gset)

        ek = err_kind(perr)
        sub = None
        if ek == "missing_column":
            sub = missing_column_subkind(
                perr, all_columns_lower(db_map, p["db_id"], col_cache))

        row = {
            "id": r["id"], "db": p["db_id"], "correct": correct,
            "question": p.get("question", ""), "evidence": p.get("evidence", ""),
            "pred_err": perr, "gold_err": gerr,
            "err_kind": ek, "err_subkind": sub,
            "shape": shape_kind(pset, gset),
            "n_pred_rows": None if pset is None else len(pset),
            "n_gold_rows": None if gset is None else len(gset),
            "finish_reason": r.get("finish_reason"),
            "pred": pred, "gold": p["gold"],
        }
        if use_ast and not correct:
            row["ast"] = ast_kind(pred, p["gold"])

        # --- repair guard, scored as a separate column -----------------------
        if use_guard:
            from repair import repair_identifiers
            if p["db_id"] not in schema_cache:
                schema_cache[p["db_id"]] = schema_columns(p["db_id"], db_map)
            fixed, changes = repair_identifiers(pred, schema_cache[p["db_id"]])
            if changes and fixed != pred:
                pset2, perr2 = run_query(db_path, fixed, timeout)
                row["correct_guarded"] = int(
                    perr2 is None and gerr is None and pset2 == gset)
                row["guard_changes"] = "; ".join(
                    f"{a}->{b}({c})" for a, b, c in changes)
            else:
                row["correct_guarded"] = correct
                row["guard_changes"] = ""
        rows.append(row)

    res = pd.DataFrame(rows)
    n = len(res)
    winnable = res[res.gold_err.isna()]

    print(f"\n{'=' * 62}")
    print(f"{os.path.basename(results_path)}   split={split}  variant={variant}")
    print(f"{'=' * 62}")
    print(f"scored:               {n}")
    print(f"Execution Accuracy:   {res.correct.mean():.4f}  "
          f"({int(res.correct.sum())}/{n})")
    print(f"EX on winnable only:  {winnable.correct.mean():.4f}  "
          f"({int(winnable.correct.sum())}/{len(winnable)})")
    if use_guard:
        print(f"EX with repair guard: {res.correct_guarded.mean():.4f}  "
              f"({int(res.correct_guarded.sum())}/{n})   "
              f"[+{int(res.correct_guarded.sum() - res.correct.sum())} questions]")

    n_gold_err = int(res.gold_err.notna().sum())
    if n_gold_err:
        print(f"\nGOLD queries that errored: {n_gold_err} (unwinnable)")
    if res.finish_reason.notna().any():
        n_trunc = int((res.finish_reason == "length").sum())
        print(f"truncated completions: {n_trunc}"
              + ("  <-- raise max_tokens or investigate looping" if n_trunc else ""))

    hard = res[res.err_kind.notna()]
    if len(hard):
        print(f"\nhard errors: {len(hard)} "
              f"({len(hard) / n:.1%} of all queries, "
              f"{len(hard) / max(1, n - int(res.correct.sum())):.1%} of misses)")
        print(hard.err_kind.value_counts().to_string())
        mc = hard[hard.err_subkind.notna()]
        if len(mc):
            print("\n  missing_column split:")
            for k, v in mc.err_subkind.value_counts().items():
                print(f"    {k:<12} {v}")
            print("    wrong_table -> FK edge list arm; "
                  "bad_name -> repair guard")

    silent = res[(res.correct == 0) & res.err_kind.isna()]
    if len(silent):
        print(f"\nsilent failures: {len(silent)} "
              f"({len(silent) / max(1, n - int(res.correct.sum())):.1%} of misses)")
        print(silent["shape"].value_counts().to_string())
        pe = int((silent["shape"] == "pred_empty").sum())
        nm = int((silent["shape"] == "ncol_mismatch").sum())
        print(f"\n  pred_empty={pe} -> wrong literal format "
              "(direct measure for the sample-values arm)")
        print(f"  ncol_mismatch={nm} -> SELECT-clause disagreement; a chunk of "
              "these are strict-metric artifacts, not model errors")

    if use_ast and "ast" in res.columns:
        a = res[res.ast.notna()]
        print("\nAST diff over misses:")
        print(a.ast.value_counts().to_string())
        print("\nshape x AST (rows=shape, cols=AST):")
        print(pd.crosstab(a["shape"], a["ast"]).to_string())

    by_db = (res.groupby("db").correct.agg(["mean", "count"])
             .sort_values("mean").head(show))
    print(f"\nweakest {show} databases:")
    print(by_db.to_string())

    if out_csv:
        res.to_csv(out_csv, index=False)
        print(f"\nper-query detail -> {out_csv}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--ast", action="store_true",
                    help="add sqlglot structural diff (slower, needs sqlglot)")
    ap.add_argument("--guard", action="store_true",
                    help="also score with the identifier repair guard applied")
    ap.add_argument("--expect-variant", default=None,
                    help="assert the payload used this schema variant")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--show", type=int, default=10)
    add_common_args(ap)
    args = ap.parse_args()

    score(args.payload, args.results, args.data_root, args.split_dir,
          args.timeout, args.ast, args.guard, args.out_csv, args.show,
          args.expect_variant)


if __name__ == "__main__":
    main()
