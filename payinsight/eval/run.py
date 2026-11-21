"""Score PayInsight on the 30 known-answer questions.

    python -m eval.run                      # Heron if reachable, rules planner
    python -m eval.run --heron off          # fully offline (compiler writes all SQL)
    python -m eval.run --planner auto       # hosted model for understanding, if a key exists

Writes eval/results.json (read by the app's Evaluation page) and eval/results.md.
Heron is scored as its OWN row, as in Heron's score.py: "system correct" can be
carried by the semantic-layer cross-check, so it never stands in for "Heron correct".
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TARGETS = {"simple": 0.85, "segment": 0.85, "root": 1.0, "ambiguous": None, "refuse": 1.0}
GROUP_LABEL = {"simple": "Simple metrics", "segment": "Segment and trend", "root": "Root cause",
               "ambiguous": "Ambiguous", "refuse": "Must refuse"}


def close(a, b, unit):
    if a is None or b is None:
        return False
    if unit == "percent":
        return abs(float(a) - float(b)) <= 0.1
    return abs(float(a) - float(b)) <= max(0.5, abs(float(b)) * 0.001)


def same_label(ans, gold):
    if ans is None or gold is None:
        return False
    a, g = str(ans).strip().lower(), str(gold).strip().lower()
    if a == g:
        return True
    try:                                       # dates and hours: "15 Oct 20:00" vs Timestamp
        ga = pd.Timestamp(gold)
        aa = pd.to_datetime(f"{ans} 2026", format="mixed", dayfirst=True)
        return (ga.floor("h") == aa.floor("h")) if ga.hour or aa.hour else ga.date() == aa.date()
    except (ValueError, TypeError):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--heron", default=os.environ.get("HERON_MODE", "auto"), choices=["auto", "off"])
    ap.add_argument("--planner", default="rules", choices=["rules", "auto", "llm"])
    ap.add_argument("--only", default=None, help="comma-separated ids")
    args = ap.parse_args(argv)
    os.environ["HERON_MODE"] = args.heron

    from engine import settings
    settings.HERON_MODE = args.heron
    from engine import llm
    llm.HERON_MODE = args.heron
    from engine.db import system_connection
    from engine.pipeline import PayInsight, Session

    qs = yaml.safe_load(open(os.path.join(HERE, "questions.yaml")))
    audit_path = os.path.join(HERE, "eval_audit.duckdb")
    con = system_connection()
    bots = {}
    rows = []
    only = set(args.only.split(",")) if args.only else None
    t_all = time.perf_counter()
    for group, items in qs.items():
        for q in items:
            if only and q["id"] not in only:
                continue
            role = q.get("role", "payments_leader")
            if role not in bots:
                bots[role] = PayInsight(role, planner_mode=args.planner, use_cache=False,
                                        audit_path=audit_path)
            t0 = time.perf_counter()
            ans = bots[role].ask(q["question"], Session())
            ms = (time.perf_counter() - t0) * 1000
            r = {"id": q["id"], "group": group, "role": role, "question": q["question"],
                 "kind": ans.kind, "headline": ans.headline, "ms": round(ms),
                 "sql_source": ans.meta.get("sql_source"), "heron_attempted": ans.meta.get("heron_attempted"),
                 "heron_ok": ans.meta.get("heron_ok"), "number_check": (ans.number_check or {}).get("ok")}
            if group in ("simple", "segment"):
                g = con.execute(q["gold"]).fetchone()
                gold_val = g[-1]
                gold_lab = g[0] if group == "segment" else None
                r["gold"] = float(gold_val) if gold_val is not None else None
                r["gold_label"] = str(gold_lab) if gold_lab is not None else None
                r["got"] = ans.value
                r["got_label"] = ans.top_segment
                ok = ans.kind == "answer" and close(ans.value, gold_val, q["unit"])
                if group == "segment":
                    ok = ok and same_label(ans.top_segment, gold_lab)
                r["pass"] = bool(ok)
                hv = ans.meta.get("heron_value")
                r["heron_value"] = hv
                r["heron_pass"] = bool(ans.meta.get("heron_ok")) and r["pass"]
            elif group == "root":
                c = ans.causes[:2]
                mix_ok = any(x.startswith("mix:") and "PlayArena" in x for x in c)
                rate_ok = any(x.startswith("rate:") and "Coral Bank" in x for x in c)
                r["got"] = c
                r["pass"] = ans.kind == "answer" and mix_ok and rate_ok
            elif group == "ambiguous":
                stated = any("Period taken as" in x for x in ans.caveats)
                r["got"] = ans.kind
                r["pass"] = ans.kind == "clarify" or stated
            elif group == "refuse":
                leaked = sum(e.rows for e in ans.evidence)
                r["got"] = (ans.refusal or {}).get("code", ans.kind)
                r["rows_returned"] = leaked
                r["pass"] = ans.kind == "refusal" and leaked == 0
            rows.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"{mark}  {q['id']:4s} {ms:7.0f} ms  {r.get('sql_source') or '':15s} {q['question'][:62]}")
            if not r["pass"]:
                print(f"        got={r.get('got_label') or ''} {r.get('got')}  gold={r.get('gold_label') or ''} "
                      f"{r.get('gold')}  kind={ans.kind}  {ans.headline[:120]}")
    con.close()

    summary = {}
    for group in qs:
        rs = [r for r in rows if r["group"] == group]
        if not rs:
            continue
        summary[group] = {"label": GROUP_LABEL[group], "passed": sum(r["pass"] for r in rs),
                          "total": len(rs), "target": TARGETS[group]}
    hq = [r for r in rows if r["group"] in ("simple", "segment")]
    heron = {"attempted": sum(bool(r.get("heron_attempted")) for r in hq),
             "accepted": sum(bool(r.get("heron_ok")) for r in hq),
             "correct": sum(bool(r.get("heron_pass")) for r in hq), "total": len(hq)}
    out = {"run_at": datetime.now().isoformat(timespec="seconds"), "heron_mode": args.heron,
           "planner": args.planner, "seconds": round(time.perf_counter() - t_all, 1),
           "summary": summary, "heron": heron, "rows": rows}
    if not only:
        for name in ("results.json", f"results_heron_{'live' if args.heron == 'auto' else 'off'}.json"):
            with open(os.path.join(HERE, name), "w") as f:
                json.dump(out, f, indent=2, default=str)
        write_md(out, "results.md")
        write_md(out, f"results_heron_{'live' if args.heron == 'auto' else 'off'}.md")
    print()
    for g, s in summary.items():
        tgt = f" (target {s['target']:.0%})" if s["target"] else ""
        print(f"  {s['label']:20s} {s['passed']}/{s['total']}{tgt}")
    print(f"  Heron SQL: attempted {heron['attempted']}/{heron['total']}, matched the semantic-layer "
          f"check {heron['accepted']}, correct {heron['correct']}")
    refuse = summary.get("refuse")
    if refuse and refuse["passed"] < refuse["total"]:
        print("  BLOCKER: a must-refuse question was not refused.")
        return 2
    return 0


def write_md(out, name="results.md"):
    lines = [f"# Evaluation results", "",
             f"Run {out['run_at']} · Heron mode `{out['heron_mode']}` · planner `{out['planner']}` · "
             f"{out['seconds']} s total", "", "| Group | Passed | Target |", "|---|---|---|"]
    for s in out["summary"].values():
        lines.append(f"| {s['label']} | {s['passed']}/{s['total']} | "
                     f"{'' if s['target'] is None else format(s['target'], '.0%')} |")
    h = out["heron"]
    lines += ["", f"Heron SQL (simple + segment questions): attempted {h['attempted']}/{h['total']}, "
                  f"matched the independent semantic-layer check {h['accepted']}, correct {h['correct']}.",
              "", "| id | group | pass | source | ms | question | headline |", "|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['id']} | {r['group']} | {'✅' if r['pass'] else '❌'} | {r.get('sql_source') or ''} "
                     f"| {r['ms']} | {r['question']} | {r['headline'][:140].replace('|', '/')} |")
    with open(os.path.join(HERE, name), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
