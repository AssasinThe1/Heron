#!/usr/bin/env python3
"""Sample and compare predictions for manual error analysis. LOCAL.

Reads the per-query CSV from score.py --out-csv. Self-sufficient: never reaches
back into train.json or dev.json, which is how the original notebook version
ended up printing train questions beside dev results.

Examples
--------
  python sample_errors.py --errors errors_fk+samples.csv --only wrong -n 10
  python sample_errors.py --errors errors_fk+samples.csv --shape pred_empty
  python sample_errors.py --errors errors_fk+samples.csv --subkind bad_name
  python sample_errors.py --errors errors_tuned.csv --compare errors_base.csv \
      --only regressions
"""

import argparse
import os

import pandas as pd

COLS = ["id", "db", "correct", "question", "evidence", "pred", "gold",
        "pred_err", "err_kind", "err_subkind", "shape", "ast",
        "n_pred_rows", "n_gold_rows", "gold_err"]


def load(path):
    df = pd.read_csv(path)
    need = {"id", "db", "correct", "pred", "gold"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}. Regenerate with "
                         "score.py --out-csv.")
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df


def apply_filters(res, args):
    if args.db:
        res = res[res.db == args.db]
    if args.errored_only:
        res = res[res.err_kind.notna()]
    if args.silent_only:
        res = res[res.err_kind.isna() & (res.correct == 0)]
    if args.shape:
        res = res[res["shape"] == args.shape]
    if args.subkind:
        res = res[res.err_subkind == args.subkind]
    if args.ast:
        res = res[res.ast == args.ast]
    return res


def sample_table(res, n, seed, only):
    parts = []
    if only in ("both", "correct"):
        pool = res[res.correct == 1]
        parts.append(pool.sample(min(n, len(pool)), random_state=seed))
    if only in ("both", "wrong"):
        pool = res[res.correct == 0]
        parts.append(pool.sample(min(n, len(pool)), random_state=seed))
    if not parts or all(p.empty for p in parts):
        return res.iloc[0:0][COLS]
    return pd.concat(parts).reset_index(drop=True)[COLS]


def regressions(res, base, n, seed):
    """Churn analysis plus McNemar. Aggregate accuracy hides this: a change can
    land +2pp overall while breaking 40 questions it previously got right."""
    m = res.merge(base[["id", "correct"]], on="id", suffixes=("", "_base"))
    lost = m[(m.correct == 0) & (m.correct_base == 1)]
    gained = m[(m.correct == 1) & (m.correct_base == 0)]
    b, c = len(lost), len(gained)
    print(f"vs baseline: {c} newly correct, {b} newly wrong (net {c - b:+d})")
    chi2, verdict = _mcnemar(c, b)
    if b + c:
        print(f"McNemar chi2={chi2:.2f}  ({verdict}, 1 dof)")
    else:
        print("no discordant pairs -- the two runs agree everywhere")
    return lost.sample(min(n, b), random_state=seed).reset_index(drop=True)[COLS]


def arm_name(path):
    """errors_fk+samples.csv -> 'fk+samples'. Falls back to the bare stem."""
    b = os.path.basename(path)
    if b.startswith("errors_") and b.endswith(".csv"):
        return b[len("errors_"):-len(".csv")]
    return os.path.splitext(b)[0]


def _mcnemar(gain, lost):
    """Continuity-corrected McNemar chi2 and a verdict, for one discordant pair
    count. Shared by regressions() and the arms comparison so both report the
    same statistic."""
    if gain + lost == 0:
        return 0.0, "no discordant pairs"
    chi2 = (abs(gain - lost) - 1) ** 2 / (gain + lost)
    return chi2, ("SIGNIFICANT at p<0.05" if chi2 > 3.84 else "not significant")


def compare_arms(paths, baseline, n, seed):
    """Aggregate comparison across >=2 arm CSVs.

    Aggregate EX hides the mechanism: two arms can post near-identical accuracy
    while disagreeing on a fifth of the questions, and a richer prompt can raise
    EX in one database while quietly regressing another. This lays that out --
    per-arm EX and error mix, a per-database EX matrix sorted by spread, the
    all-right / all-wrong / contested split, each arm's unique wins, the
    best-per-question ceiling, and pairwise McNemar against a baseline -- so the
    format decision is made on churn and significance, not a single number.
    """
    frames = {arm_name(p): load(p) for p in paths}
    names = list(frames)
    common = sorted(set.intersection(*[set(f.id) for f in frames.values()]))
    frames = {a: f[f.id.isin(common)].set_index("id").sort_index()
              for a, f in frames.items()}
    base = baseline or ("ddl" if "ddl" in names else names[0])
    if base not in names:
        raise SystemExit(f"baseline {base!r} not among arms {names}")

    corr = pd.DataFrame({a: frames[a].correct for a in names})
    print(f"\ncomparing {len(names)} arms on {len(common)} common questions "
          f"(baseline={base})")
    print(f"arms: {names}")

    print("\n=== per-arm summary ===")
    for a in names:
        f = frames[a]
        trunc = int(f.finish_reason.eq("length").sum()) if "finish_reason" in f else 0
        herr = int(f.err_kind.notna().sum())
        to = int(f.err_kind.eq("timeout").sum())
        print(f"  {a:<14} EX={f.correct.mean():.4f}  "
              f"({int(f.correct.sum())}/{len(f)})  hard_err={herr:<3} "
              f"truncated={trunc:<3} timeouts={to}")

    n_q = len(corr)
    allc = int((corr.sum(axis=1) == len(names)).sum())
    allw = int((corr.sum(axis=1) == 0).sum())
    ceiling = float((corr.max(axis=1) == 1).mean())
    print(f"\n=== consensus over {n_q} questions ===")
    print(f"  all arms correct:  {allc:>4} ({allc/n_q:.1%})")
    print(f"  all arms wrong:    {allw:>4} ({allw/n_q:.1%})  <- unreachable by "
          "any format")
    print(f"  contested:         {n_q - allc - allw:>4} "
          f"({(n_q - allc - allw)/n_q:.1%})")
    print(f"  best-per-question ceiling (oracle over arms): {ceiling:.4f} "
          f"(+{ceiling - corr[base].mean():.4f} over {base})")

    print("\n=== questions ONLY this arm gets right (unique wins) ===")
    for a in names:
        others = [o for o in names if o != a]
        only = int(((corr[a] == 1) & (corr[others].sum(axis=1) == 0)).sum())
        print(f"  {a:<14} {only}")

    dbs = frames[names[0]].db
    ex_by_db = {a: frames[a].groupby(frames[a].db).correct.mean() for a in names}
    mat = pd.DataFrame(ex_by_db)
    mat.insert(0, "n", dbs.groupby(dbs).size())
    mat["best"] = mat[names].idxmax(axis=1)
    mat["spread"] = mat[names].max(axis=1) - mat[names].min(axis=1)
    print("\n=== per-db EX by arm (sorted by spread; where formats disagree "
          "most) ===")
    print(mat.sort_values("spread", ascending=False).round(3).to_string())

    print(f"\n=== pairwise vs {base} (McNemar, 1 dof) ===")
    for a in names:
        if a == base:
            continue
        gain = int(((corr[a] == 1) & (corr[base] == 0)).sum())
        lost = int(((corr[a] == 0) & (corr[base] == 1)).sum())
        chi2, verdict = _mcnemar(gain, lost)
        print(f"  {a:<14} +{gain:<3} -{lost:<3} net {gain - lost:+d}   "
              f"chi2={chi2:.2f} ({verdict})")
    return mat


def render(tbl):
    if tbl.empty:
        print("no rows matched")
        return
    for _, r in tbl.iterrows():
        mark = "OK  " if r.correct == 1 else "FAIL"
        print("=" * 78)
        head = f"[{mark}] id={r.id}  db={r.db}"
        for f in ("err_kind", "err_subkind", "shape", "ast"):
            v = r.get(f)
            if pd.notna(v) and v:
                head += f"  {f}={v}"
        print(head)
        print(f"Q:        {r.question}")
        if pd.notna(r.evidence) and str(r.evidence).strip():
            print(f"evidence: {r.evidence}")
        if pd.notna(r.n_pred_rows) or pd.notna(r.n_gold_rows):
            print(f"rows:     pred={r.n_pred_rows}  gold={r.n_gold_rows}")
        print(f"\npred:\n  {r.pred}")
        print(f"\ngold:\n  {r.gold}")
        if pd.notna(r.pred_err) and str(r.pred_err).strip():
            print(f"\npred error: {r.pred_err}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--errors", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="2+ error CSVs for an aggregate comparison, e.g. "
                         "errors_ddl.csv errors_fk.csv errors_samples.csv")
    ap.add_argument("--baseline", default=None,
                    help="arm name to compare the rest against (default ddl)")
    ap.add_argument("-n", "--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="both",
                    choices=["both", "correct", "wrong", "regressions"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--errored-only", action="store_true")
    ap.add_argument("--silent-only", action="store_true",
                    help="misses that executed cleanly (the ~89%)")
    ap.add_argument("--shape", default=None, help="e.g. pred_empty, ncol_mismatch")
    ap.add_argument("--subkind", default=None, help="wrong_table | bad_name")
    ap.add_argument("--ast", default=None, help="e.g. join_path, literals_only")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    if args.arms:
        if len(args.arms) < 2:
            ap.error("--arms needs at least 2 error CSVs")
        compare_arms(args.arms, args.baseline, args.n, args.seed)
        return
    if not args.errors:
        ap.error("--errors is required (or use --arms for a multi-arm compare)")

    res = load(args.errors)
    if args.only == "regressions":
        if not args.compare:
            ap.error("--only regressions requires --compare")
        tbl = regressions(res, load(args.compare), args.n, args.seed)
    else:
        tbl = sample_table(apply_filters(res, args), args.n, args.seed, args.only)

    render(tbl)
    if args.out_csv:
        tbl.to_csv(args.out_csv, index=False)
        print(f"\n-> {args.out_csv}")
    if args.out_md:
        open(args.out_md, "w").write(tbl.to_markdown(index=False))
        print(f"-> {args.out_md}")


if __name__ == "__main__":
    main()
