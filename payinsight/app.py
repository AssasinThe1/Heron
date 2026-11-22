"""PayInsight — Streamlit front end.

    streamlit run app.py
"""

import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import audit, charts, llm, report, settings  # noqa: E402
from engine.evidence import CACHE  # noqa: E402
from engine.pipeline import PayInsight, Session  # noqa: E402

st.set_page_config(page_title="PayInsight", page_icon="💳", layout="wide")

ROLES = {k: v["label"] for k, v in settings.policy()["roles"].items()}
EXAMPLES = [
    "Why did our success rate drop last week?",
    "Break Coral Bank UPI down by hour.",
    "What was October's successful payment value?",
    "Which bank had the lowest success rate in the week of 12 October?",
    "What was the success rate in the North region in October?",
    "Give me customer phone numbers for failed payments",
    "Show me every transaction",
    "Delete the test merchants",
]
CONF_BADGE = {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low"}

st.markdown("""
<style>
.pi-headline {font-size: 1.15rem; font-weight: 600; line-height: 1.45; margin: 0.2rem 0 0.6rem;}
.pi-meta {color: #8a8984; font-size: 0.8rem;}
.pi-tag {font-family: ui-monospace, monospace; font-size: 0.85em;}
div[data-testid="stExpander"] summary p {font-size: 0.92rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building this role's isolated database…")
def bot_for(role):
    return PayInsight(role)


def state():
    ss = st.session_state
    ss.setdefault("turns", {})          # role -> [(question, Answer)]
    ss.setdefault("sessions", {})       # role -> Session
    ss.setdefault("queued", None)
    ss.setdefault("samples", {})
    return ss


# ----------------------------------------------------------------- sidebar
ss = state()
with st.sidebar:
    st.title("💳 PayInsight")
    st.caption("AI business analyst for payments · data through 31 Oct 2026 (IST)")
    page = st.radio("Page", ["Weekly report", "Ask", "Data & rules", "Audit log", "Evaluation"],
                    label_visibility="collapsed", key="page")
    role = st.selectbox("Signed in as", list(ROLES), format_func=ROLES.get,
                        help="Simulated login. A real deployment takes the role from single sign-on.")
    ss["sessions"].setdefault(role, Session())
    ss["turns"].setdefault(role, [])

    st.divider()
    hs = llm.heron_status()
    if settings.HERON_MODE == "off":
        st.markdown("**Heron** · off (compiler writes SQL)")
    elif not llm.heron_enabled():
        st.markdown("**Heron** · ⚠️ unreachable, using compiler")
    else:
        last = f" · last {hs['heron_last_ms'] / 1000:.1f} s" if hs.get("heron_last_ms") else ""
        st.markdown(f"**Heron** · on Modal{last}")
    if settings.HERON_MODE != "off" and st.button("Wake Heron (≈1–3 min cold start)"):
        with st.spinner("Waking Heron on Modal…"):
            text, info = llm.heron_wake(timeout=300)
        if text:
            st.rerun()                     # redraw the status line above, now "on Modal"
        st.toast(f"Heron did not answer: {info.get('error')}")
    st.markdown("**Understanding** · " + ("hosted model (Groq/Gemini)" if llm.hosted_available()
                                          and settings.PLANNER_MODE != "rules" else "rules planner"))
    st.caption(f"Cache: {CACHE.hits} hits / {CACHE.misses} misses")
    if st.button("New conversation"):
        ss["sessions"][role] = Session()
        ss["turns"][role] = []
        st.rerun()


# ----------------------------------------------------------------- render
def render_evidence(ans, key):
    if not ans.evidence:
        return
    with st.expander(f"Evidence panel · {len(ans.evidence)} queries", expanded=False):
        for e in ans.evidence:
            st.markdown(f"**[{e.id}] {e.title}**  \n"
                        f"<span class='pi-meta'>dataset: {e.dataset} · filters: {e.filters or '—'} · "
                        f"window: {e.window or '—'} · SQL by: {e.source} · est. rows scanned: "
                        f"{e.rows_scanned:,} · {e.run_ms:.0f} ms · rows: {e.rows:,}"
                        f"{' · cached' if e.cached else ''} · check: {e.check}</span>",
                        unsafe_allow_html=True)
            if e.note:
                st.caption(e.note)
            st.code(e.sql, language="sql")
            for i, alt in enumerate(e.alt_sql or []):
                st.caption(f"Earlier attempt {i + 1} (rejected):")
                st.code(alt, language="sql")
            if e.preview is not None and len(e.preview):
                st.dataframe(e.preview.head(12), width="stretch", hide_index=True)


def render_answer(ans, idx, is_last):
    key = f"{role}-{idx}"
    if ans.kind == "clarify":
        st.markdown(f"<div class='pi-headline'>{ans.headline}</div>", unsafe_allow_html=True)
        c = ans.clarify
        if not is_last:
            st.caption("(answered below)")
            return
        labels = [o["label"] for o in c["period_options"]]
        pick = st.radio("Period", labels, key=f"p-{key}")
        dpick = None
        if c.get("denominator_options"):
            dl = [o["label"] for o in c["denominator_options"]]
            dpick = st.radio("Success rate counts", dl, key=f"d-{key}", horizontal=True)
        b1, b2 = st.columns([1, 3])
        if b1.button("Answer", key=f"go-{key}", type="primary"):
            patch = dict(next(o["patch"] for o in c["period_options"] if o["label"] == pick))
            patch["source"] = "clarified"
            if dpick:
                patch.update(next(o["patch"] for o in c["denominator_options"] if o["label"] == dpick))
            ss["queued"] = ("__patch__", patch)
            st.rerun()
        if b2.button("Skip — use defaults", key=f"def-{key}"):
            patch = dict(c["default"])
            patch["source"] = "default"
            ss["queued"] = ("__patch__", patch)
            st.rerun()
        st.caption(c["note"])
        return

    if ans.kind in ("refusal", "no_data", "error"):
        box = st.error if ans.kind == "refusal" else st.warning
        box(ans.headline, icon="🔒" if ans.kind == "refusal" else "ℹ️")
        r = ans.refusal or {}
        if r.get("offer"):
            st.markdown(r["offer"])
        cols = st.columns(3)
        alt = r.get("alternative")
        if alt and cols[0].button(alt["label"], key=f"alt-{key}"):
            ss["queued"] = (alt["question"], None)
            st.rerun()
        smp = r.get("sample")
        if smp and cols[1].button(smp["label"], key=f"smp-{key}"):
            df, ev = bot_for(role).run_sample(smp["sql"])
            ss["samples"][key] = (df, ev)
        if key in ss["samples"]:
            df, ev = ss["samples"][key]
            st.caption(f"Random sample, {len(df):,} rows · passed the query checks · {ev.run_ms:.0f} ms")
            st.dataframe(df, width="stretch", hide_index=True)
            st.code(ev.sql, language="sql")
        for cv in ans.caveats:
            st.caption(cv)
        render_meta(ans)
        return

    st.markdown(f"<div class='pi-headline'>{ans.headline}</div>", unsafe_allow_html=True)
    if ans.charts:
        main = ans.charts[0]
        st.plotly_chart(main[1], width="stretch", key=f"c0-{key}")
    left, right = st.columns([3, 2])
    with left:
        if ans.facts:
            st.markdown("**Observed facts**")
            st.markdown("\n".join(f"- {f}" for f in ans.facts))
        if ans.hypotheses:
            st.markdown("**Hypotheses**")
            for h in ans.hypotheses:
                st.markdown(f"- **Hypothesis** · {CONF_BADGE[h['confidence']]} confidence  \n  {h['text']}")
                for x in h.get("for", []):
                    st.markdown(f"    - ✅ For: {x}")
                for x in h.get("against", []):
                    st.markdown(f"    - ⚠️ Against: {x}")
                if h.get("why"):
                    st.caption("    Rating rules: " + "; ".join(h["why"]))
    with right:
        if ans.actions:
            st.markdown("**Recommended actions**")
            st.markdown("\n".join(f"- *Recommendation:* {a}" for a in ans.actions))
        st.markdown("**Caveats**")
        st.markdown("\n".join(f"- {c}" for c in ans.caveats) if ans.caveats else "- None found.")
    if len(ans.charts) > 1:
        tabs = st.tabs([c[0] for c in ans.charts[1:]])
        for i, (tab, (title, fig, eid)) in enumerate(zip(tabs, ans.charts[1:])):
            with tab:
                st.plotly_chart(fig, width="stretch", key=f"c{i + 1}-{key}")
                st.caption(f"Source: [{eid}]")
    for title, df in ans.tables:
        with st.expander(f"Table · {title} ({len(df):,} rows)"):
            st.dataframe(df, width="stretch", hide_index=True)
    render_evidence(ans, key)
    render_meta(ans)


def render_meta(ans):
    m = ans.meta
    nc = ans.number_check
    bits = [f"planner: {m.get('planner', '—')}", f"SQL: {m.get('sql_source') or '—'}",
            f"{m.get('queries', 0)} queries", f"{m.get('elapsed_ms', 0) / 1000:.1f} s"]
    if m.get("cache_hits"):
        bits.append(f"{m['cache_hits']} cached")
    if nc is not None:
        bits.append("number check ✓" if nc["ok"] else "number check ✗")
    if m.get("audit_id"):
        bits.append(f"audit #{m['audit_id']}")
    st.markdown(f"<span class='pi-meta'>{' · '.join(bits)}</span>", unsafe_allow_html=True)


def ask(question, patch=None):
    bot = bot_for(role)
    sess = ss["sessions"][role]
    with st.spinner("Clarifying, writing SQL, checking, running, analysing…"):
        ans = bot.ask(question, sess, patch=patch)
    shown = question if patch is None else "↳ " + patch.get("period_label", "chosen period")
    ss["turns"][role].append((shown, ans))


def ask_from_report(question):
    """Button callback: jump to the Ask page with the question queued."""
    st.session_state["page"] = "Ask"
    st.session_state["queued"] = (question, None)


# ----------------------------------------------------------------- pages
if page == "Weekly report":
    bot = bot_for(role)
    df, ev = report.weekly(bot.rdb)
    if df is None:
        st.info("This role has no access to payment attempts, so there is no weekly report.")
    else:
        weeks = [f"{w:%d %b} – {e:%d %b %Y}" for w, e in zip(df.week, df.week_end)]
        default = report.largest_drop(df)
        default = len(df) - 1 if default is None else default
        head, pick = st.columns([3, 2])
        head.subheader(f"Weekly payments report · {ROLES[role]}")
        i = pick.selectbox("Week", range(len(df)), index=default, format_func=weeks.__getitem__,
                           label_visibility="collapsed")
        cmp = report.compare(df, i)
        if cmp is None:
            st.caption("Not enough earlier weeks for a 4-week comparison.")
        else:
            st.caption(f"Compared with the 4 weeks before ({cmp['base_start']:%d %b} – "
                       f"{cmp['base_end']:%d %b}). Success rate excludes customer-cancelled attempts.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Success rate", f"{cmp['rate']:.1f}%", f"{cmp['pp']:+.1f} pp")
            c2.metric("Payment attempts", f"{cmp['attempts']:,.0f}", f"{cmp['attempts_pct']:+.1f}%",
                      delta_color="off")
            c3.metric("Successful value", f"₹{cmp['value'] / 1e7:.2f} crore", f"{cmp['value_pct']:+.1f}%")
            c4.metric("Failed payments", f"{cmp['failures']:,.0f}", f"{cmp['failures_pct']:+.1f}%",
                      delta_color="inverse")
        st.plotly_chart(charts.weekly_trend(df, i, cmp), width="stretch", key="weekly-trend")
        if cmp is not None and cmp["pp"] < 0:
            d = df.week.iloc[i]
            q = f"Why did our success rate drop in the week of {d.day} {d:%B}?"
            st.button(f"Ask PayInsight: “{q}”", type="primary", on_click=ask_from_report, args=(q,))
        with st.expander("How this report is computed"):
            st.code(ev.sql, language="sql")
            st.caption(f"Passed the query checks · {ev.run_ms:.0f} ms")

elif page == "Ask":
    st.subheader(f"Ask PayInsight · {ROLES[role]}")
    if not ss["turns"][role]:
        st.markdown("Ask about success rates, volumes, payment value, banks, merchants or settlements. "
                    "Every number comes from a checked query you can open and copy.")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex-{i}", width="stretch"):
                ss["queued"] = (q, None)
                st.rerun()

    if ss["queued"]:
        q, patch = ss["queued"]
        ss["queued"] = None
        if q == "__patch__":
            ask("", patch=patch)
        else:
            ask(q)

    turns = ss["turns"][role]
    for i, (q, ans) in enumerate(turns):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="💳"):
            render_answer(ans, i, i == len(turns) - 1)

    q = st.chat_input("e.g. Why did our success rate drop last week?")
    if q:
        ss["queued"] = (q, None)
        st.rerun()

elif page == "Data & rules":
    bot = bot_for(role)
    st.subheader(f"What {ROLES[role]} can see")
    rows = [{"table": t, "rows": bot.rdb.row_counts[t], "columns": ", ".join(bot.rdb.columns[t]),
             "row filter": bot.spec["tables"][t].get("row_filter", "—")} for t in bot.rdb.tables]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    t1, t2, t3, t4 = st.tabs(["Schema Heron sees", "semantic_layer.yaml", "policy.yaml", "Planted scenarios"])
    with t1:
        st.caption("Heron's frozen training format: SQLite-style DDL of this role's tables only.")
        st.code(bot.rdb.schema_ddl(), language="sql")
    with t2:
        st.code(open(settings.SEMANTIC_PATH).read(), language="yaml")
    with t3:
        st.code(open(settings.POLICY_PATH).read(), language="yaml")
    with t4:
        st.markdown("""
| # | Planted on purpose | Expected effect |
|---|---|---|
| P1 | From 1 Oct, gaming merchant **PlayArena** grows to ~18% of attempts, mostly card (~65% success) | Total rate falls; each method's own rate stays flat → **mix shift** |
| P2 | **Coral Bank UPI** fails 18:00–22:00 on 14 & 15 Oct (logged), with unlogged degradation around it | A further drop that week; failure code ISSUER_DOWN |
| P3 | 2,400 attempts on 9 Oct recorded twice (retry bug, skewed to failures) | Duplicates removed and counted |
| P4 | 3% of rows on 20–22 Sep have no bank; 10:00–16:00 on 17 Aug missing | Blank-value and gap warnings |
| P5 | Settlements use UTC days, pay out next day, net off refunds; 12 payout rows missing | Two sources disagree on October revenue |
""")
        mpath = os.path.join(settings.ROOT, "data", "manifest.json")
        if os.path.exists(mpath):
            st.json(json.load(open(mpath)), expanded=False)

elif page == "Audit log":
    st.subheader("Audit log")
    st.caption("Every question, role, generated SQL, check result, run time and answer.")
    df = audit.read(500)
    if df.empty:
        st.info("No entries yet.")
    else:
        c1, c2 = st.columns(2)
        roles_f = c1.multiselect("Role", sorted(df.role.unique()))
        out_f = c2.multiselect("Outcome", sorted(df.outcome.unique()))
        if roles_f:
            df = df[df.role.isin(roles_f)]
        if out_f:
            df = df[df.outcome.isin(out_f)]
        st.dataframe(df.drop(columns=["intent"]), width="stretch", hide_index=True,
                     column_config={"run_ms": st.column_config.NumberColumn("run ms", format="%.0f")})

elif page == "Evaluation":
    st.subheader("Evaluation · 30 known-answer questions")
    runs = {"Latest run": settings.EVAL_RESULTS,
            "Heron live (Modal)": os.path.join(settings.ROOT, "eval", "results_heron_live.json"),
            "Offline (compiler only)": os.path.join(settings.ROOT, "eval", "results_heron_off.json")}
    runs = {k: v for k, v in runs.items() if os.path.exists(v)}
    path = runs[st.radio("Run", list(runs), horizontal=True)] if runs else settings.EVAL_RESULTS
    if st.button("Re-run evaluation offline (≈10 s)"):
        with st.spinner("Running python -m eval.run --heron off …"):
            subprocess.run([sys.executable, "-m", "eval.run", "--heron", "off"], cwd=settings.ROOT,
                           capture_output=True, text=True)
    st.caption("With Heron live: `python -m eval.run` from the terminal (≈3 min).")
    if not os.path.exists(path):
        st.info("No results yet. Run `python -m eval.run`.")
    else:
        res = json.load(open(path))
        st.caption(f"Run {res['run_at']} · Heron `{res['heron_mode']}` · planner `{res['planner']}` · "
                   f"{res['seconds']} s")
        cols = st.columns(len(res["summary"]) + 1)
        for col, s in zip(cols, res["summary"].values()):
            tgt = f"target {s['target']:.0%}" if s["target"] else "asks or states assumption"
            col.metric(s["label"], f"{s['passed']}/{s['total']}", tgt, delta_color="off")
        h = res["heron"]
        cols[-1].metric("Heron SQL accepted", f"{h['accepted']}/{h['total']}",
                        "matched independent check", delta_color="off")
        df = pd.DataFrame(res["rows"])
        df["pass"] = df["pass"].map({True: "✅", False: "❌"})
        for c in ("got", "gold", "got_label", "gold_label"):
            if c in df:
                df[c] = df[c].map(lambda x: "" if x is None or x != x else str(x))
        show = [c for c in ["id", "group", "pass", "role", "question", "sql_source", "got_label", "got",
                            "gold_label", "gold", "ms", "headline"] if c in df]
        st.dataframe(df[show], width="stretch", hide_index=True)
        st.markdown("**Must-refuse questions** are a blocker, not a score: any miss fails the build. "
                    "Heron is scored as its own row; the system's number can come from the "
                    "semantic-layer cross-check, so system accuracy never stands in for Heron's.")
