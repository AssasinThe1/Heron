"""The eight steps, end to end.

  1 understand (planner: rules or hosted model)      AI may help
  2 clarify if the period/denominator is unclear      code decides when
  3 policy pre-checks (write / PII / region / range)  code
  4 write SQL (Heron; compiler fallback)              AI
  5 check + run (guardrails, cache, timeout)          code
  6 analyse (segments, mix vs rate, drill-down)       code
  7 explain (templates filled from results)           code; AI may polish wording
  8 verify numbers, write audit log                   code
"""

import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from . import analysis, audit, charts, compiler, dq, guardrails, llm, planner, reconcile, sqlgen
from .answer import Answer, Fmt, finalize
from .db import lookup_values, role_db
from .evidence import CACHE, EvidenceLog, QueryError, Runner
from .planner import fmt_range
from .settings import data_meta, policy, semantic

METRIC_LABEL = {
    "success_rate": "Success rate", "order_success_rate": "Order success rate",
    "attempts": "Payment attempts", "failures": "Failed attempts",
    "successful_value": "Successful payment value", "avg_ticket": "Average ticket",
    "settled_net": "Net settlements",
}
DIM_LABEL = {d: s["label"].lower() for d, s in semantic()["dimensions"].items()}
DENOM_CAVEAT = ("Success rate = successful attempts ÷ all attempts, excluding attempts the "
                "customer cancelled. A unique-order success rate is also available.")


@lru_cache(maxsize=1)
def lookups():
    return lookup_values()


@dataclass
class Session:
    history: list = field(default_factory=list)       # (question, Intent, Answer)
    pending: object = None                            # Intent awaiting clarification

    @property
    def last_intent(self):
        for _q, it, ans in reversed(self.history):
            if ans.kind == "answer" and it.period:
                return it
        return None


class PayInsight:
    def __init__(self, role, planner_mode=None, use_cache=True, audit_path=None):
        self.role = role
        self.rdb = role_db(role)
        self.spec = self.rdb.spec
        self.planner_mode = planner_mode
        self.use_cache = use_cache
        self.audit_path = audit_path

    # ------------------------------------------------------------------ entry
    def ask(self, question, session=None, patch=None, auto_default=False):
        session = session if session is not None else Session()
        t0 = time.perf_counter()
        log = EvidenceLog()
        runner = Runner(self.rdb, log, use_cache=self.use_cache)
        hits0 = CACHE.hits
        F = Fmt()

        if patch is not None and session.pending is not None:
            it = planner.apply_patch(session.pending, patch)
            question = it.question
            session.pending = None
        else:
            prev = session.history[-1][0] if session.history else None
            it = planner.plan(question, lookups(), context=prev, mode=self.planner_mode)
            self._inherit(it, session)

        try:
            ans = self._route(it, runner, F, session, auto_default)
        except QueryError as e:
            ans = self._refusal_from_check(e.check, it, F) if e.check else Answer(
                kind="error", headline="The query could not be completed: " + str(e.error)[:200],
                caveats=["Nothing was answered from partial results."])
        except compiler.CompileError as e:
            ans = Answer(kind="refusal", headline=(
                f"That needs {e} that your role ({self.spec['label']}) does not include."),
                refusal={"code": "forbidden_table", "reason": str(e)})
        ans.fmt = ans.fmt or F
        ans.evidence = log.items
        ans.meta.update({"intent": it.to_json(), "planner": it.planner,
                         "elapsed_ms": (time.perf_counter() - t0) * 1000,
                         "queries": len(log.items), "cache_hits": CACHE.hits - hits0})
        if ans.kind == "answer":
            finalize(ans)
        if ans.kind == "clarify":
            session.pending = it
        session.history.append((question, it, ans))
        self._audit(question, it, ans)
        return ans

    def _inherit(self, it, session):
        """Follow-ups ("break Coral Bank UPI down by hour") reuse the last period."""
        last = session.last_intent
        if last is None or it.period or it.kind not in ("metric", "why"):
            return
        if it.vague_period and it.period_label not in ("unspecified",):
            return                              # "last week" etc. are asked, not inherited
        if planner.VAGUE_RE.search(it.question):
            return
        it.period, it.period_label = last.period, last.period_label
        it.period_source, it.vague_period = "inherited", False
        if it.metric == "success_rate" and last.metric == "order_success_rate" \
                and not it.denominator_explicit:
            it.metric = last.metric

    # ---------------------------------------------------------------- routing
    def _route(self, it, runner, F, session, auto_default):
        meta = data_meta()
        regions = self.spec.get("regions", "all")

        if it.kind == "write":
            return self._refusal("not_read_only",
                                 "PayInsight is read-only: it cannot change, delete or load data.",
                                 "Ask a data engineer to change records; I can show the merchants "
                                 "you had in mind instead.", None, F)
        if it.kind == "pii" and self.spec.get("pii", "none") == "none":
            return self._refusal("pii", "Customer personal data (phone numbers, UPI IDs) is not "
                                 f"available to your role ({self.spec['label']}).",
                                 "I can show counts or rates by segment instead.",
                                 {"label": "Show attempts by payment method instead",
                                  "question": "How many payment attempts by payment method last full week?"}, F)
        if regions != "all" and it.filters.get("region"):
            outside = [r for r in it.filters["region"] if r not in regions]
            if outside:
                alt_q = it.question
                for r in outside:
                    alt_q = re.sub(rf"\b{r}(ern)?\b", regions[0], alt_q, flags=re.I)
                return self._refusal(
                    "region_scope",
                    f"Your role ({self.spec['label']}) covers the {', '.join(regions)} region only, "
                    f"so I can't show {', '.join(outside)}.",
                    f"I can show the {regions[0]} region instead.",
                    {"label": f"Show the {regions[0]} region instead", "question": alt_q}, F)

        if it.kind == "dump":
            return self._dump(it, runner, F)

        # period: clarify, default, range and coverage checks
        if not it.period and it.kind in ("metric", "why", "reconcile"):
            if it.kind == "reconcile":
                it.period = planner.month_range(meta["end_date"].year, meta["end_date"].month)
                it.period_label, it.period_source = fmt_range(*it.period), "default"
            elif auto_default:
                c = it.period_candidates[0] if it.period_candidates else planner.period_candidates()[0]
                planner.apply_patch(it, {"period": c["period"], "period_label": c["label"],
                                         "source": "default"})
            else:
                return self._clarify(it, runner, F)
        a, b = it.period
        max_days = policy()["limits"]["max_range_days"]
        if (b - a).days > max_days:
            return self._refusal(
                "range_too_long", f"That covers {(b - a).days:,} days; queries are limited to "
                f"{max_days} days so they stay fast and cheap.",
                f"The data covers {fmt_range(meta['start_date'], meta['end_date'])}; I can use that.",
                {"label": "Use the full data window",
                 "question": f"{METRIC_LABEL[it.metric]} between {meta['start_date']} and {meta['end_date']}"}, F)
        if b < meta["start_date"] or a > meta["end_date"]:
            return Answer(kind="no_data", headline=(
                f"There is no data for {it.period_label or fmt_range(a, b)}. The data covers "
                f"{fmt_range(meta['start_date'], meta['end_date'])}."))
        clipped = False
        if a < meta["start_date"] or b > meta["end_date"]:
            it.period = (max(a, meta["start_date"]), min(b, meta["end_date"]))
            clipped = True

        if it.kind == "why":
            if it.metric not in compiler.RATE_METRICS:
                it.kind, it.grain = "metric", it.grain or "day"
                ans = self._metric(it, runner, F)
                ans.caveats.insert(0, "Root-cause analysis is built for success rates; showing the "
                                      f"{METRIC_LABEL[it.metric].lower()} trend instead.")
                return ans
            if "transactions" not in self.rdb.tables:
                return self._refusal("forbidden_table", "Explaining a change needs transaction-level "
                                     f"detail, which your role ({self.spec['label']}) does not include.",
                                     "I can show totals over time instead.", None, F)
            ans = self._why(it, runner, F)
        elif it.kind == "reconcile" or (it.metric == "successful_value" and not it.group_by
                                        and not it.grain and not it.filters
                                        and reconcile.can_reconcile(self.rdb)):
            if not reconcile.can_reconcile(self.rdb):
                it.kind = "metric"
                ans = self._metric(it, runner, F)
                ans.caveats.append("Settlements are outside your access, so this figure is not "
                                   "reconciled against payouts.")
            else:
                ans = self._reconcile(it, runner, F)
        else:
            ans = self._metric(it, runner, F)
        if clipped:
            ans.caveats.insert(0, f"The data covers {fmt_range(meta['start_date'], meta['end_date'])}; "
                                  f"the period was trimmed to {fmt_range(*it.period)}.")
        return ans

    # ------------------------------------------------------------- refusals
    def _refusal(self, code, message, offer, alternative, F):
        return Answer(kind="refusal", headline=message,
                      refusal={"code": code, "reason": message, "offer": offer,
                               "alternative": alternative},
                      caveats=["Refused before any query ran. The attempt is in the audit log."], fmt=F)

    def _refusal_from_check(self, chk, it, F):
        alt = None
        if chk.code == "too_many_rows":
            alt = {"label": "Show a daily summary instead",
                   "question": f"Attempts and success rate by day {it.period_label or ''}".strip()}
        return Answer(kind="refusal", headline=chk.message,
                      refusal={"code": chk.code, "reason": chk.reason, "offer": "",
                               "alternative": alt},
                      caveats=["Blocked by the query checks before it ran. Logged in the audit log."],
                      fmt=F)

    def _dump(self, it, runner, F):
        meta = data_meta()
        a, b = it.period or (meta["start_date"], meta["end_date"])
        table = "transactions" if "transactions" in self.rdb.tables else self.rdb.tables[0]
        tcol = "created_at" if table == "transactions" else "hour"
        sql = (f"SELECT * FROM {table} WHERE {tcol} >= '{a}' AND {tcol} < '{b + timedelta(days=1)}'")
        chk = guardrails.check(sql, self.rdb)
        if chk.ok:                                   # a short window that fits the limit
            df, ev = runner.run(sql, title="Requested rows", dataset=table, window=fmt_range(a, b),
                                check=chk)
            ans = Answer(headline=f"{F.n(len(df))} rows for {F.drange(a, b)} [{ev.id}].",
                         tables=[("Rows", df)], fmt=F)
            return ans
        ans = self._refusal_from_check(chk, it, F)
        n = policy()["limits"]["sample_rows"]
        ans.refusal["offer"] = (f"I can show a daily summary, or a random sample of {n:,} rows.")
        ans.refusal["alternative"] = {"label": "Show a daily summary instead",
                                      "question": f"Attempts and success rate by day between {a} and {b}"}
        ans.refusal["sample"] = {"label": f"Show a sample of {n:,} rows",
                                 "sql": f"WITH w AS (SELECT * FROM {table} WHERE {tcol} >= '{a}' AND "
                                        f"{tcol} < '{b + timedelta(days=1)}') "
                                        f"SELECT * FROM w TABLESAMPLE RESERVOIR ({n} ROWS) REPEATABLE (42)"}
        return ans

    def run_sample(self, sql):
        """The 'sample of 1,000 rows' offer, still through the checks."""
        log = EvidenceLog()
        runner = Runner(self.rdb, log, use_cache=self.use_cache)
        df, ev = runner.run(sql, title="Random sample", dataset="transactions", source="template",
                            note="Reservoir sample, fixed seed")
        return df, ev

    # ------------------------------------------------------------- clarify
    def _clarify(self, it, runner, F):
        options = [{"label": c["label"], "patch": {"period": c["period"], "period_label": c["label"]}}
                   for c in (it.period_candidates or planner.period_candidates())]
        if it.kind == "why" and "transactions" in self.rdb.tables:
            wk = self._largest_drop_week(runner)
            if wk:
                lab = f"Week of {fmt_range(*wk)} (largest weekly drop)"
                if not any(o["patch"]["period"] == wk for o in options):
                    options.insert(1, {"label": lab, "patch": {"period": wk, "period_label":
                                                               f"week of {fmt_range(*wk)}"}})
        denoms = None
        if it.metric in compiler.RATE_METRICS and not it.denominator_explicit:
            denoms = [{"label": "All attempts (default)", "patch": {"metric": "success_rate"}},
                      {"label": "Unique orders", "patch": {"metric": "order_success_rate"}}]
        what = "which period" + (" and which success rate" if denoms else "")
        return Answer(kind="clarify", headline=f"Quick check: {what} do you mean?",
                      clarify={"period_options": options, "denominator_options": denoms,
                               "default": options[0]["patch"],
                               "note": "If you skip this, I'll use the first option and say so."},
                      fmt=F)

    def _largest_drop_week(self, runner):
        meta = data_meta()
        a = meta["start_date"]
        b = planner.last_full_week()[1]
        try:
            df, _ = runner.run(
                f"SELECT CAST(date_trunc('week', t.created_at) AS DATE) AS week, "
                f"COUNT(DISTINCT t.txn_id) AS n, COUNT(DISTINCT t.txn_id) FILTER (WHERE t.status = 'success') AS k "
                f"FROM transactions t WHERE t.status <> 'user_cancelled' AND t.created_at >= '{a}' "
                f"AND t.created_at < '{b + timedelta(days=1)}' GROUP BY 1 ORDER BY 1",
                title="Weekly success rate (to find the largest drop)", dataset="transactions",
                window=fmt_range(a, b))
        except QueryError:
            return None
        df = df[df.week >= pd.Timestamp(a + timedelta(days=(7 - a.weekday()) % 7))]  # whole weeks only
        df["rate"] = df.k / df.n
        best, wk = 0, None
        for i in range(4, len(df)):
            prior = df.iloc[i - 4:i]
            drop = df.rate.iloc[i] - prior.k.sum() / prior.n.sum()
            if drop < best:
                best, wk = drop, df.week.iloc[i].date()
        return (wk, wk + timedelta(days=6)) if wk else None

    # --------------------------------------------------------------- metric
    def _metric(self, it, runner, F):
        comp = compiler.compile_intent(it, self.rdb)
        label = METRIC_LABEL[it.metric]
        window = fmt_range(*it.period)
        flabel = _flt(it.filters)
        sres = sqlgen.generate(it, self.rdb)
        if sres.stop is not None:
            return self._refusal_from_check(sres.stop, it, F)
        df, ev_c = runner.run(comp.sql, title=f"{label}: semantic-layer query",
                              dataset=comp.table + _joins_label(comp.sql), filters=flabel,
                              window=window, source="semantic layer")
        cite = ev_c.id
        heron_note = None
        if sres.ok:
            match = results_match(sres.df, df, comp.unit)
            ev_h = runner.log.add(
                title=f"{label}: query written by {'Heron' if sres.provider == 'heron' else 'hosted model'}",
                dataset=comp.table, filters=flabel, window=window, sql=sres.sql,
                source=sres.provider, rows_scanned=sres.check.scanned, run_ms=sres.run_ms,
                rows=len(sres.df), preview=sres.df.head(50),
                note=("Cross-checked against the semantic-layer query "
                      f"[{ev_c.id}]: {'results match' if match else 'RESULTS DIFFER'}."
                      + (f" Repaired identifiers: {sres.attempts[-1].repaired}." if sres.attempts
                         and sres.attempts[-1].repaired else "")
                      + (f" Attempts: {len(sres.attempts)}." if len(sres.attempts) > 1 else "")))
            ev_h.alt_sql = [a.sql or a.raw for a in sres.attempts[:-1] if a.sql or a.raw]
            if match:
                cite = ev_h.id
                ev_c.note = f"Independent check of [{ev_h.id}] from semantic_layer.yaml: results match."
            else:
                heron_note = (f"{'Heron' if sres.provider == 'heron' else 'The hosted model'}'s query "
                              f"[{ev_h.id}] returned a different result from the semantic-layer "
                              f"definition [{ev_c.id}]; this answer uses the definition.")
        else:
            reason = "unreachable" if not sres.attempts or sres.attempts[-1].error and \
                "unreachable" in (sres.attempts[-1].error or "") else "did not produce a passing query"
            if not llm.heron_enabled() and not sres.attempts:
                reason = "offline"
            ev_c.note = (f"Heron {reason}; SQL compiled from semantic_layer.yaml."
                         + (f" Heron attempts: {len(sres.attempts)}." if sres.attempts else ""))
        ans = Answer(fmt=F)
        ans.meta["sql_source"] = sres.provider if (sres.ok and cite != ev_c.id) else "semantic layer"
        ans.meta["heron_ok"] = bool(sres.ok and cite != ev_c.id)
        ans.meta["heron_attempted"] = bool(sres.attempts)
        ans.meta["heron_value"] = _primary_value(sres.df) if sres.ok else None
        self._metric_text(it, comp, df, cite, ans, F)
        fig = charts.metric_chart(df, comp.dims, comp.bucket, comp.unit,
                                  f"{label} — {flabel if it.filters else 'all payments'}, {window}")
        if fig is not None:
            ans.charts.append((fig.layout.title.text, fig, cite))
        if len(df) > 1:
            ans.tables.append(("Result", df))

        # drill-down extras: outage log and failure codes when a bank is in focus
        if it.filters.get("bank") and "transactions" in self.rdb.tables:
            self._bank_context(it, runner, ans, F, df, comp, cite)

        if it.metric in compiler.RATE_METRICS and not it.denominator_explicit:
            ans.caveats.append(DENOM_CAVEAT if it.metric == "success_rate" else
                               "Order success rate = unique orders with at least one success ÷ unique orders.")
        if it.period_source in ("default", "clarified", "inherited"):
            ans.caveats.append(f"Period taken as {it.period_label}"
                               + (" (from your previous question)." if it.period_source == "inherited" else "."))
        if it.order and it.metric in compiler.RATE_METRICS and (comp.dims or comp.bucket):
            ans.caveats.append(F._reg(f"Ranking ignores segments with fewer than "
                                      f"{compiler.rank_floor(it)} attempts."))
        if heron_note:
            ans.caveats.append(heron_note)
        if "transactions" in comp.table:
            self._dq_caveats(ans, [it.period], comp.dims, F, runner)
        return ans

    def _metric_text(self, it, comp, df, cite, ans, F):
        label = METRIC_LABEL[it.metric]
        unit = comp.unit
        scope = _flt_text(it.filters)
        per = F.drange(*it.period)
        v = lambda x: _fmt_val(F, x, unit)                                      # noqa: E731
        if df.empty or df.value.isna().all():
            ans.headline = f"No matching payments for {scope} in {per} [{cite}]."
            ans.value = None
            return
        if not comp.dims and not comp.bucket:
            r = df.iloc[0]
            ans.value = float(r.value)
            ans.headline = f"{label} for {scope} in {per}: {v(r.value)} [{cite}]."
            if unit == "percent" and "k" in df:
                lo, hi = analysis.wilson(r.k, r.n)
                ans.facts.append(f"{F.n(r.k)} of {F.n(r.n)} "
                                 f"{'orders' if it.metric == 'order_success_rate' else 'attempts'} "
                                 f"succeeded [{cite}]; " + F._reg("95% range ") + f"{F.pct(lo * 100)} to {F.pct(hi * 100)}.")
            elif unit == "inr":
                ans.facts.append(f"From {F.n(r.n)} successful payments [{cite}].")
            return
        keycols = ([comp.bucket] if comp.bucket else []) + comp.dims
        shown = df[keycols].copy()
        if comp.bucket in ("hour", "day", "week"):
            fmt_b = {"hour": "%d %b %H:00", "day": "%d %b", "week": "week of %d %b"}[comp.bucket]
            shown[comp.bucket] = pd.to_datetime(shown[comp.bucket]).dt.strftime(fmt_b).str.lstrip("0")
        names = shown.astype(str).agg(" · ".join, axis=1)
        names = [_label(F, n) for n in names]
        if it.order:
            word = "lowest" if it.order == "asc" else "highest"
            top = df.iloc[0]
            ans.value = float(top.value)
            ans.top_segment = names[0]
            what = " and ".join(DIM_LABEL.get(d, d) for d in keycols)
            ans.headline = (f"{names[0]} had the {word} {label.lower()} by {what} for {scope} in "
                            f"{per}: {v(top.value)} [{cite}].")
            for n, (_, r) in list(zip(names, df.iterrows()))[1:6]:
                ans.facts.append(f"{n}: {v(r.value)} ({F.n(r.n)} {_n_word(it.metric)}) [{cite}].")
            return
        if comp.bucket and not comp.dims:
            big = df[df.n >= 30] if (unit == "percent" and (df.n >= 30).any()) else df
            imin, imax = int(big.value.idxmin()), int(big.value.idxmax())
            ans.value = float(df.value.mean()) if unit == "percent" else float(df.value.sum())
            ans.headline = (f"{label} for {scope}, {per}, by {comp.bucket}: lowest {v(df.value[imin])} "
                            f"({names[imin]}), highest {v(df.value[imax])} ({names[imax]}) [{cite}].")
            ans.top_segment = names[imin]
            if comp.bucket == "hour" and unit == "percent":
                med = float(big.value.median())
                low = big[big.value < med - 15]
                if len(low):
                    spans = _hour_spans(low[comp.bucket])
                    ans.facts.append("Hours more than 15 pp below the typical hour "
                                     f"({v(med)}): " + "; ".join(
                                         f"{F.date(s.date())} {F.hour(s.hour)}–{F.hour(e.hour)}" for s, e in spans)
                                     + f" [{cite}].")
            ans.facts.append(f"{F.n(len(df))} {comp.bucket}s; first {names[0]} at {v(df.value.iloc[0])}, "
                             f"last {names[-1]} at {v(df.value.iloc[-1])} [{cite}].")
            if unit == "percent" and len(big) < len(df):
                ans.caveats.append(F._reg(f"Lowest/highest ignore {comp.bucket}s with fewer than 30 attempts."))
            return
        # grouped, unordered
        order = df.sort_values("value", ascending=False)
        top, bot = order.index[0], order.index[-1]
        ans.value = float(df.value.iloc[0]) if len(df) == 1 else None
        ans.top_segment = names[list(df.index).index(bot)] if unit == "percent" else \
            names[list(df.index).index(top)]
        ans.headline = (f"{label} for {scope} in {per} ranges from {v(df.value[bot])} "
                        f"({names[list(df.index).index(bot)]}) to {v(df.value[top])} "
                        f"({names[list(df.index).index(top)]}) [{cite}].")
        for n, (_, r) in list(zip(names, df.iterrows()))[:8]:
            ans.facts.append(f"{n}: {v(r.value)} ({F.n(r.n)} {_n_word(it.metric)}) [{cite}].")

    def _bank_context(self, it, runner, ans, F, df, comp, cite):
        a, b = it.period
        banks = it.filters["bank"]
        cond = [f"b.name IN ({', '.join(analysis._q(x) for x in banks)})",
                f"e.started_at < '{b + timedelta(days=1)}'", f"e.ended_at > '{a}'"]
        if it.filters.get("method"):
            cond.append(f"e.method IN ({', '.join(analysis._q(x) for x in it.filters['method'])})")
        evs, e = runner.run(
            "SELECT b.name AS bank, e.method, e.started_at, e.ended_at, e.reason FROM "
            "bank_downtime_events e JOIN banks b ON e.bank_id = b.bank_id WHERE "
            + " AND ".join(cond) + " ORDER BY e.started_at",
            title="Logged bank outages in the period", dataset="bank_downtime_events + banks",
            filters=_flt(it.filters), window=fmt_range(a, b))
        if len(evs):
            spans = "; ".join(f"{F.ts(r.started_at)}–{F.hour(r.ended_at.hour)}" for r in evs.itertuples())
            ans.facts.append(f"{F.n(len(evs))} logged outage{'s' if len(evs) > 1 else ''} for "
                             f"{_flt_text(it.filters)}: {spans} [{e.id}].")
            if comp.bucket == "hour" and "k" in df:
                inside = np.zeros(len(df), dtype=bool)
                for r in evs.itertuples():
                    inside |= (df.hour >= r.started_at.floor("h")) & (df.hour < r.ended_at)
                fails = df.n - df.k
                n_out = df.n[~inside].sum()
                r_out = df.k[~inside].sum() / n_out if n_out else 0
                excess = (fails - df.n * (1 - r_out)).clip(lower=0)
                tot = excess.sum()
                if tot > 0:
                    share = excess[inside].sum() / tot
                    ans.facts.append(f"Outside the logged windows success is {F.pct(r_out * 100)}; inside "
                                     f"them it is {F.pct(100 * df.k[inside].sum() / max(df.n[inside].sum(), 1))} "
                                     f"[{cite}, {e.id}].")
                    ans.facts.append(f"{F.share(share)} of the extra failures (above the normal rate) "
                                     f"fall inside the logged windows [{cite}, {e.id}].")
                    ans.hypotheses.append({
                        "text": "The low hours are the logged outage, not a general decline.",
                        "confidence": "high" if share >= 0.7 else "medium",
                        "why": [f"{share:.0%} of extra failures inside logged windows"],
                        "for": [f"Extra failures cluster in the logged windows [{e.id}]."],
                        "against": ([f"{F.share(1 - share)} of the extra failures fall outside them [{cite}]."]
                                    if share < 0.98 else ["None found in the data checked."])})
            if ans.charts and comp.bucket == "hour":
                fig = ans.charts[0][1]
                for r in evs.itertuples():
                    fig.add_vrect(x0=r.started_at, x1=r.ended_at, fillcolor=charts.CRITICAL,
                                  line_width=0, annotation_text="logged outage",
                                  annotation_position="top left", annotation_font_size=10)
        else:
            ans.facts.append(f"No outage is logged for {_flt_text(it.filters)} in {F.drange(a, b)} [{e.id}].")
        # failure codes for the same scope
        fs, joins = analysis._filters_sql(it.filters)
        codes, e2 = runner.run(
            f"SELECT t.failure_code AS code, COUNT(DISTINCT t.txn_id) AS failures FROM "
            f"{analysis._from(joins)} WHERE t.status = 'failed' AND t.created_at >= '{a}' AND "
            f"t.created_at < '{b + timedelta(days=1)}' AND {' AND '.join(fs)} GROUP BY 1 ORDER BY 2 DESC",
            title="Failure codes in the period", dataset="transactions + banks",
            filters=_flt(it.filters) + "; status = failed", window=fmt_range(a, b))
        if len(codes):
            top = codes.iloc[0]
            ans.facts.append(f"Top failure code: {F.label(top.code)}, {F.share(top.failures / codes.failures.sum())} "
                             f"of {F.n(codes.failures.sum())} failures [{e2.id}].")
            ans.tables.append(("Failure codes", codes))
        if not ans.actions and len(evs):
            ans.actions.append(f"Ask {banks[0]} to confirm the outage windows and root cause.")

    # ------------------------------------------------------------------ why
    def _why(self, it, runner, F):
        current = it.period
        baseline = it.compare or analysis.default_baseline(current)
        rep = dq.run([baseline, current], self.spec["tables"]["transactions"].get("row_filter"))
        res = analysis.why(it, runner, fmt_range, rep.warned_dims())
        T = res.totals
        ans = Answer(fmt=F)
        e1 = res.ev[res.primary_dim]
        d = T["delta"]
        verb = "fell" if d < 0 else "rose"
        weeks = semantic()["analysis"]["baseline_weeks"]
        base_lab = (F._reg(f"the {weeks} weeks before") + f" ({F.drange(*baseline)})" if not it.compare
                    else F.drange(*baseline))
        mix_share = T["mix"] / d if d else 0
        rate_share = T["rate"] / d if d else 0
        metric_lab = METRIC_LABEL[it.metric]
        scope = "" if not it.filters else f" for {_flt_text(it.filters)}"
        head = (f"{metric_lab}{scope} {verb} from {F.pct(T['R0'] * 100)} to {F.pct(T['R1'] * 100)} "
                f"({F.pp(d * 100)}) in {F.drange(*current)}, compared with {base_lab} [{e1}].")
        if abs(d) < 0.001 or analysis.overlaps(T["ci0"], T["ci1"]):
            head += (" The overall change is within normal variation (the 95% ranges overlap)"
                     + (", but one segment changed clearly; see below." if res.rate_path else "."))
            F._reg("95%")
        elif mix_share >= 0.5:
            head += (f" About {F.share(mix_share)} of the {'drop' if d < 0 else 'rise'} comes from a change "
                     "in transaction mix, not from worse performance." if d < 0 else
                     f" About {F.share(mix_share)} of the rise comes from a change in transaction mix.")
        else:
            head += (f" Most of it ({F.share(rate_share)}) comes from segments actually performing "
                     f"{'worse' if d < 0 else 'better'}, not from the mix.")
        ans.headline = head
        ans.value = d * 100
        pd_lab = DIM_LABEL.get(res.primary_dim, res.primary_dim)
        ans.facts.append(f"Split by {pd_lab}: mix effect {F.pp(T['mix'] * 100, 2)}, rate effect "
                         f"{F.pp(T['rate'] * 100, 2)}; together {F.pp(d * 100, 2)} [{e1}].")

        causes = []
        mp = res.mix_path
        if mp:
            r = mp["row"]
            seg = mp["segment"]
            flat = bool(r.noise)
            ans.facts.append(
                f"{_cap(seg)} grew from {F.share(r.w0)} to {F.share(r.w1)} of attempts; its own success "
                f"rate {'held at about' if flat else 'moved to'} {F.pct(r.r1 * 100)} "
                f"(was {F.pct(r.r0 * 100)}{', within noise' if flat else ''}) [{e1}].")
            drv = mp.get("driver")
            if drv:
                dr = drv["row"]
                ev_d = res.ev.get(drv["dim"])
                info = mp.get("merchant_info") or {}
                cat = f" ({info.get('category')})" if info.get("category") else ""
                ans.facts.append(f"{drv['segment']}{cat} went from {F.share(dr.w0, 1)} to "
                                 f"{F.share(dr.w1, 1)} of attempts, succeeding at {F.pct(dr.r1 * 100)} "
                                 f"against {F.pct(T['R0'] * 100)} overall [{ev_d}].")
                mm = mp.get("merchant_methods")
                if mm is not None and len(mm):
                    t0 = mm.iloc[0]
                    ans.facts.append(f"{F.share(t0.share)} of {drv['segment']}'s attempts are "
                                     f"{_cap(t0.method)} payments, succeeding at {F.pct(t0.rate * 100)} "
                                     f"[{mp['ev_merchant_methods']}].")
                pf = mp.get("primary_from_driver")
                if pf is not None and "current" in pf.index:
                    c = pf.loc["current"]
                    b0 = pf.loc["baseline"] if "baseline" in pf.index else None
                    line = (f"{drv['segment']} supplied {F.share(c.from_merchant / c.attempts)} of "
                            f"{seg} attempts in the current period")
                    if b0 is not None:
                        line += f" vs {F.share(b0.from_merchant / b0.attempts)} in the baseline"
                    if c.n_excl:
                        line += (f"; without it, {seg} success was {F.pct(100 * c.ok_excl / c.n_excl)}"
                                 + (f" (baseline {F.pct(100 * b0.ok_excl / b0.n_excl)})" if b0 is not None
                                    and b0.n_excl else ""))
                    ans.facts.append(line + f" [{mp['ev_primary_from_driver']}].")
            event = mp.get("event")
            warn = res.primary_dim in rep.warned_dims()
            conf, why_ = analysis.confidence(abs(mix_share), int(T["N1"]), bool(event), warn, False)
            subject = drv["segment"] if drv else f"{seg} payments"
            h = {"text": (f"The {'drop' if d < 0 else 'rise'} is mostly a mix shift: {subject}"
                          + (f", a {info.get('category')} merchant onboarded on {F.date(event['date'])}," if
                             drv and event and drv["dim"] == "merchant" else "")
                          + f" brought in a large share of {seg} payments, which succeed less often than "
                            f"average. Nothing in the {seg} flow itself got worse."),
                 "confidence": conf, "why": why_,
                 "for": [f"The mix effect explains {F.share(abs(mix_share))} of the change [{e1}]."]
                 + ([F._reg(f"{_cap(seg)}'s own success rate is flat within its 95% range [{e1}].")] if flat else [])
                 + ([f"{drv['segment']} was onboarded on {F.date(event['date'])} [{mp['ev_merchant']}]."]
                    if event else []),
                 "against": ([f"{_cap(seg)}'s own success rate also moved, by "
                              f"{F.pp((r.r1 - r.r0) * 100)} [{e1}]."] if not flat else [])
                 or ["None found in the data checked."]}
            ans.hypotheses.append(h)
            causes.append(f"mix:{drv['segment'] if drv else seg}")
            ans.top_segment = drv["segment"] if drv else seg

        rp = res.rate_path
        if rp:
            name = " ".join(_cap(str(lv["segment"])) for lv in rp["levels"])
            leaf = rp["levels"][-1]["row"]
            excl = rp.get("exclude")
            ex_txt = (f" (after setting aside {list(excl.values())[0][0]})" if excl else "")
            ev_lvl = rp["levels"][-1].get("ev") or rp["levels"][0].get("ev")
            ans.facts.append(f"{name} success fell from {F.pct(leaf.r0 * 100)} to {F.pct(leaf.r1 * 100)} "
                             f"for the period{ex_txt} [{ev_lvl}].")
            bad_days = rp.get("bad_days") or []
            if bad_days and rp.get("bad_days_rate") is not None:
                days = _days_text(F, bad_days)
                ans.facts.append(f"The fall is concentrated on {days}: {F.pct(rp['bad_days_rate'] * 100)} "
                                 f"success on those days [{rp['ev_days']}].")
            bh = rp.get("bad_hours")
            if bh is not None and len(bh):
                spans = _hour_spans(bh.hour)
                ans.facts.append("Hours far below normal: " + "; ".join(
                    f"{F.date(s.date())} {F.hour(s.hour)}–{F.hour(e.hour)}" for s, e in spans)
                    + f" [{rp['ev_hourly']}].")
            codes = rp.get("codes")
            if codes is not None and len(codes):
                top = codes.index[0]
                ans.facts.append(f"{F.label(top)} went from {F.share(codes.share0.iloc[0])} to "
                                 f"{F.share(codes.share1.iloc[0])} of {name} failures [{rp['ev_codes']}].")
            evs = rp.get("events")
            explains = abs(rp["explains"])
            has_event = evs is not None and len(evs) > 0
            warn = any(lv["dim"] in rep.warned_dims() for lv in rp["levels"])
            conf, why_ = analysis.confidence(explains, int(leaf.n1), has_event, warn, bool(leaf.noise))
            if has_event:
                spans = "; ".join(f"{F.ts(r.started_at)}–{F.hour(r.ended_at.hour)}" for r in evs.itertuples())
                h = {"text": f"The {name} fall is the outage logged in the bank outage log ({spans}).",
                     "confidence": conf, "why": why_,
                     "for": [f"Logged outage{'s' if len(evs) > 1 else ''} for {name} on "
                             f"{_days_text(F, sorted({r.started_at.date() for r in evs.itertuples()}))} "
                             f"[{rp['ev_events']}].",
                             f"It explains {F.share(explains)} of the total change [{ev_lvl}]."]
                     + ([f"{F.label(codes.index[0])} failures jumped [{rp['ev_codes']}]."]
                        if codes is not None and len(codes) and codes.index[0] == "ISSUER_DOWN" else []),
                     "against": ([f"{F.share(rp['outside_share'])} of the extra failures on those days fall "
                                  f"outside the logged hours [{rp['ev_hourly']}]."]
                                 if rp.get("outside_share", 0) > 0.02 else ["None found in the data checked."])}
            else:
                h = {"text": f"{name} got worse during the period; no outage is logged to explain it.",
                     "confidence": conf, "why": why_,
                     "for": [f"It explains {F.share(explains)} of the total change [{ev_lvl}]."],
                     "against": ["No matching event in the bank outage log."
                                 + (f" [{rp['ev_events']}]" if rp.get("ev_events") else "")]}
            ans.hypotheses.append(h)
            causes.append("rate:" + "/".join(str(lv["segment"]) for lv in rp["levels"]))

        ans.hypotheses.sort(key=lambda h: {"high": 0, "medium": 1, "low": 2}[h["confidence"]])
        ans.causes = causes

        # recommendations, each tied to a fact or hypothesis
        if mp:
            ans.actions.append(f"Report success rate per {pd_lab} alongside the total, so mix changes "
                               "are not read as failures (see the mix hypothesis).")
            if mp.get("driver"):
                ans.actions.append(f"Review {mp['driver']['segment']}'s {mp['segment']} flow with the merchant "
                                   "(authentication and decline codes): its volume, not a breakdown, "
                                   "is what moves the total.")
        if rp:
            bank = next((lv["segment"] for lv in rp["levels"] if lv["dim"] == "bank"), None)
            name = " ".join(_cap(str(lv["segment"])) for lv in rp["levels"])
            if bank and rp.get("events") is not None and len(rp["events"]):
                ans.actions.append(f"Ask {bank} to confirm the outage window and cause, including the "
                                   "failures just outside the logged hours.")
            else:
                ans.actions.append(f"Raise the {name} drop with the owner; no logged event explains it.")
        ans.actions = ans.actions[:3]

        # caveats
        if it.period_source in ("default", "clarified", "inherited"):
            ans.caveats.append(f"Period taken as {it.period_label} (Monday–Sunday, IST), compared with "
                               f"{base_lab}.")
        if not it.denominator_explicit:
            ans.caveats.append(DENOM_CAVEAT)
        if rp and rp.get("exclude"):
            drv = list(rp["exclude"].values())[0][0]
            ans.caveats.append(f"Performance drops were searched after setting aside {drv}, the mix driver, "
                               "so its own effect is not counted twice.")
        ans.caveats.append("Causes are hypotheses: they line up in time with the change; they do not "
                           "prove cause and effect.")
        self._dq_caveats(ans, [baseline, current], [lv["dim"] for lv in (rp or {}).get("levels", [])],
                         F, runner, rep)

        # charts
        cur_lab, base_lab_s = f"Current ({fmt_range(*current)})", "Baseline (4 wks)"
        ans.charts.append(("Mix vs rate", charts.mix_rate_waterfall(
            T["R0"], T["mix"], T["rate"], T["R1"], base_lab_s, cur_lab), e1))
        pt = res.tables[res.primary_dim]
        ans.charts.append(("Share by " + pd_lab, charts.share_and_rate(pt, pd_lab), e1))
        ans.charts.append(("Rate by " + pd_lab, charts.rate_by_segment(pt, pd_lab), e1))
        ans.charts.append(("Contributions", charts.contributions(res.ranked), e1))
        if rp and rp.get("hourly") is not None:
            name = " ".join(_cap(str(lv["segment"])) for lv in rp["levels"])
            ans.charts.append((f"{name} by hour", charts.hourly(
                rp["hourly"], f"{name}: hourly success rate", rp.get("r0"), rp.get("events")),
                rp["ev_hourly"]))
        tbl = res.ranked.head(12)[["dim", "segment", "n0", "n1", "w0", "w1", "r0", "r1", "mix",
                                   "rate", "contrib", "noise"]].copy()
        for c in ("w0", "w1", "r0", "r1"):
            tbl[c] = (tbl[c] * 100).round(1)
        for c in ("mix", "rate", "contrib"):
            tbl[c] = (tbl[c] * 100).round(2)
        tbl.columns = ["dimension", "segment", "attempts (base)", "attempts (now)", "share % (base)",
                       "share % (now)", "SR % (base)", "SR % (now)", "mix pp", "rate pp", "total pp",
                       "within noise"]
        ans.tables.append(("Ranked segments (all dimensions)", tbl))
        ans.meta["sql_source"] = "template"
        return ans

    # ------------------------------------------------------------ reconcile
    def _reconcile(self, it, runner, F):
        rc = reconcile.run(it.period, runner, fmt=F)
        ans = Answer(fmt=F)
        lab = F.drange(*it.period)
        gap_pct = rc.gap / rc.txn_value * 100 if rc.txn_value else 0
        ans.value = rc.txn_value
        ans.headline = (f"{lab} successful payment value is {F.inr(rc.txn_value)} in transactions "
                        f"[{rc.ev['txn']}] but {F.inr(rc.settled_before_fees)} in settlements before fees "
                        f"[{rc.ev['settle']}]: a gap of {F.inr(rc.gap)} ({F.pct(gap_pct, 2)}). "
                        + ("The known steps below explain all of it."
                           if abs(rc.unexplained) < 1 else
                           f"Known steps explain all but {F.inr(rc.unexplained)}."))
        for label, amt, eid, note in rc.steps:
            ans.facts.append(f"{label}: {F.inr(amt)} [{eid}]. {note}")
        if rc.missing_rows is not None and len(rc.missing_rows):
            mr = rc.missing_rows
            ans.tables.append(("Merchant-days with no payout record", mr))
        ans.facts.append(f"Unexplained remainder: {F.inr(rc.unexplained)}.")
        ans.facts.append(f"Settlements also deducted {F.inr(rc.fees)} in fees; merchants were paid "
                         f"{F.inr(rc.net)} net [{rc.ev['settle']}].")
        if rc.missing_rows is not None and len(rc.missing_rows):
            ans.hypotheses.append({
                "text": "The merchant-days with sales but no payout row are a gap in the settlement "
                        "file, not sales that went unpaid.",
                "confidence": "low", "why": ["no second source to confirm the payouts"],
                "for": [f"Each has successful sales and no payout record [{rc.ev['missing']}]."],
                "against": ["Without the bank's payout statement this can't be confirmed."]})
            ans.actions.append(f"Ask settlement operations to trace the {F.n(len(rc.missing_rows))} "
                               f"merchant-days with no payout record [{rc.ev['missing']}].")
        ans.actions.append("Report revenue on one agreed basis (IST transaction day) and show settled "
                           "amounts as a separate line, with this bridge attached.")
        ans.caveats.append("Fees are excluded from the comparison: transactions do not record fees.")
        ans.caveats.append(F._reg(f"Settlements group sales by UTC day and pay out {rc.lag_days} day later; "
                                  "both rules come from the payout process, not from the data."))
        if "transactions" not in self.rdb.tables:
            ans.caveats.append(F._reg("Your role sees hourly transaction totals, which can't split the "
                                      "05:00–06:00 IST hour at the 05:30 cut-off; that hour's split lands in "
                                      "the unexplained remainder and may add false 'missing payout' matches."))
        else:
            a, b = it.period
            self._dq_caveats(ans, [(a - timedelta(days=2), b)], [], F, runner)
        ans.charts.append(("Reconciliation", charts.recon_waterfall(rc), rc.ev["settle"]))
        ans.meta["sql_source"] = "template"
        return ans

    # ---------------------------------------------------------- data quality
    def _dq_caveats(self, ans, windows, dims, F, runner, rep=None):
        tspec = self.spec["tables"].get("transactions")
        if tspec is None:
            return
        rep = rep or dq.run(windows, tspec.get("row_filter"))
        for title, sql, df, ms in rep.queries:
            runner.record(df, sql, title=f"Data-quality check: {title}",
                          dataset="transactions (raw, before de-duplication)",
                          window="; ".join(fmt_range(a, b) for a, b in windows), ms=ms)
        if rep.duplicates:
            days = ", ".join(f"{F.n(n)} on {F.date(d)}" for d, n in rep.duplicate_days[:3])
            ans.caveats.append(f"{F.n(rep.duplicates)} duplicate attempt records were removed ({days}).")
        for col, share in rep.missing.items():
            dim = dq.COLUMN_DIM[col]
            if share > dq.MISSING_WARN:
                ans.caveats.append(f"Warning: {F.share(share, 1)} of rows in the period have no {dim}; "
                                   f"results split by {dim} undercount.")
        for col, days in rep.missing_days.items():
            dim = dq.COLUMN_DIM[col]
            if days and (rep.missing.get(col, 0) <= dq.MISSING_WARN):
                worst = max(s for _, s in days)
                ans.caveats.append(f"{F.n(len(days))} day{'s' if len(days) > 1 else ''} "
                                   f"({_days_text(F, [d for d, _ in days])}) have up to {F.share(worst, 1)} "
                                   f"of rows with no {dim}; too few to change the totals.")
        for s, e, h in rep.gaps:
            ans.caveats.append(f"No records at all for {F.n(h)} hours on {F.date(s.date())} "
                               f"({F.hour(s.hour)}–{F.hour(e.hour)}), when that hour is normally busy.")
        if rep.stale:
            ans.caveats.append(f"Data may be stale: the latest record is {F.ts(rep.latest)}.")

    # ---------------------------------------------------------------- audit
    def _audit(self, question, it, ans):
        sql_ev = [e for e in ans.evidence if e.source in ("heron", "hosted", "semantic layer")]
        main = sql_ev[0] if sql_ev else (ans.evidence[0] if ans.evidence else None)
        outcome = ans.kind
        check = "passed" if ans.kind == "answer" else (ans.refusal or {}).get("code", ans.kind)
        try:
            ans.meta["audit_id"] = audit.write(
                role=self.role, question=question, kind=it.kind, outcome=outcome,
                check_result=check, sql_source=ans.meta.get("sql_source", ""),
                sql=main.sql if main else "", run_ms=ans.meta.get("elapsed_ms", 0),
                answer=ans.headline, intent=it.to_json(), n_queries=len(ans.evidence),
                cache_hits=ans.meta.get("cache_hits", 0), path=self.audit_path)
        except Exception as e:                                       # noqa: BLE001
            ans.meta["audit_error"] = str(e)


# ------------------------------------------------------------------ helpers
def results_match(hdf, cdf, unit):
    """Does Heron's result carry the same numbers as the semantic-layer query?"""
    if hdf is None or cdf is None or len(hdf) != len(cdf) or not len(cdf):
        return False
    target = np.sort(cdf.value.astype(float).to_numpy())
    for col in hdf.columns:
        s = pd.to_numeric(hdf[col], errors="coerce")
        if s.isna().any():
            continue
        vals = np.sort(s.astype(float).to_numpy())
        cands = [vals]
        if unit == "percent":
            cands.append(np.sort(vals * 100))
        for v in cands:
            tol = 0.05 if unit == "percent" else np.maximum(0.01, np.abs(target) * 1e-6)
            if np.all(np.abs(v - target) <= tol):
                return True
    return False


def _primary_value(df):
    if df is None or df.empty:
        return None
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return float(df[nums[-1]].iloc[0]) if nums else None


def _fmt_val(F, x, unit):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if unit == "percent":
        return F.pct(float(x))
    if unit == "inr":
        return F.inr(float(x))
    return F.n(float(x))


def _flt(filters):
    if not filters:
        return "all payments"
    return "; ".join(f"{d} = {', '.join(map(str, v))}" for d, v in filters.items())


def _flt_text(filters):
    if not filters:
        return "all payments"
    parts = []
    order = ["merchant", "bank", "category", "method", "device", "region", "failure_code"]
    for d, vals in sorted(filters.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99):
        vs = " or ".join(_cap(str(v)) for v in vals)
        parts.append(vs if d in ("bank", "merchant", "method") else f"{vs} {DIM_LABEL.get(d, d)}")
    return " ".join(parts)


def _cap(s):
    return {"upi": "UPI", "card": "card", "netbanking": "net banking", "wallet": "wallet",
            "ios": "iOS", "android": "Android", "web": "web", "mobile_web": "mobile web"}.get(s, s)


def _label(F, s):
    return F.label(s) if re.search(r"\d", s) else s


def _n_word(metric):
    return "orders" if metric == "order_success_rate" else (
        "payouts" if metric == "settled_net" else "attempts")


def _joins_label(sql):
    out = ""
    if "JOIN banks" in sql:
        out += " + banks"
    if "JOIN merchants" in sql:
        out += " + merchants"
    return out


def _days_text(F, days):
    days = sorted(days)
    if not days:
        return ""
    if len(days) > 1 and all((days[i + 1] - days[i]).days == 1 for i in range(len(days) - 1)) \
            and days[0].month == days[-1].month:
        return F._reg(f"{days[0].day}–{days[-1].day} {days[0]:%b}")
    return ", ".join(F.date(d) for d in days)


def _hour_spans(hours):
    hs = sorted(pd.to_datetime(pd.Series(list(hours))))
    spans = []
    start = prev = hs[0]
    for h in hs[1:] + [None]:
        if h is not None and h - prev == pd.Timedelta(hours=1):
            prev = h
            continue
        spans.append((start, prev + pd.Timedelta(hours=1)))
        if h is not None:
            start = prev = h
    return spans
