"""Text-to-SQL with Heron: prompt -> clean -> repair -> transpile -> check -> run.

Heron is prompted in its exact training format (Heron's own SYSTEM prompt +
SQLite-style DDL + "External knowledge" + question). Its SQLite output goes
through Heron's own repair guard, is transpiled to DuckDB by sqlglot, and must
pass guardrails.check() before it runs. A failed check or a runtime error goes
back to Heron with the reason, at most `max_sql_retries` times; then the free
hosted model gets one try; then the caller falls back to the compiler.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import sqlglot

from . import guardrails, llm
from .heron_bridge import HERON_SYSTEM, clean_sql, heron_user_block, repair_identifiers
from .settings import policy

DIM_HINT = {
    "bank": "issuer bank refers to banks.name (join transactions.bank_id = banks.bank_id)",
    "merchant": "merchant refers to merchants.name (join transactions.merchant_id = merchants.merchant_id)",
    "category": "merchant category refers to merchants.category",
    "method": "payment method refers to transactions.method",
    "region": "region refers to transactions.region",
    "device": "device refers to transactions.device",
    "failure_code": "failure reason refers to transactions.failure_code",
}
METRIC_HINT = {
    "success_rate": "success rate = 100.0 * COUNT(DISTINCT txn_id) with status = 'success' / "
                    "COUNT(DISTINCT txn_id), excluding status = 'user_cancelled'",
    "order_success_rate": "order success rate = 100.0 * COUNT(DISTINCT order_id) with status = 'success' "
                          "/ COUNT(DISTINCT order_id), excluding status = 'user_cancelled'",
    "attempts": "attempts = COUNT(DISTINCT txn_id)",
    "failures": "failures = COUNT(DISTINCT txn_id) with status = 'failed'",
    "successful_value": "successful payment value = SUM(amount_inr) where status = 'success'",
    "avg_ticket": "average ticket = AVG(amount_inr) where status = 'success'",
    "settled_net": "net settlement = SUM(settlements.net_inr) by settlement_date",
}
HARD_STOP = {"pii", "forbidden_table", "region_scope", "not_read_only", "blocked_function",
             "range_too_long", "too_expensive", "too_many_rows"}


@dataclass
class SQLAttempt:
    provider: str
    raw: str = ""
    sql: str = ""
    repaired: list = field(default_factory=list)
    check_code: str = ""
    reason: str = ""
    error: str = ""
    ms: float = 0.0


@dataclass
class SQLResult:
    ok: bool
    sql: str = ""
    df: object = None
    provider: str = ""
    run_ms: float = 0.0
    check: object = None
    attempts: list = field(default_factory=list)
    stop: object = None           # a guardrail Check that must be shown as a refusal


def knowledge_for(it, rdb):
    """BIRD-style 'External knowledge': definitions plus this question's mappings."""
    tcol = "created_at" if "transactions" in rdb.tables else "hour"
    parts = [METRIC_HINT.get(it.metric, "")]
    if it.period:
        a, b = it.period
        parts.append(f"{it.period_label or 'the period'} refers to {tcol} >= '{a}' AND "
                     f"{tcol} < '{b + timedelta(days=1)}'")
    for d, vals in (it.filters or {}).items():
        col = {"bank": "banks.name", "merchant": "merchants.name",
               "category": "merchants.category"}.get(d, f"transactions.{d}")
        parts.append(" or ".join(f"{v} refers to {col} = '{v}'" for v in vals))
    for d in it.group_by:
        parts.append(DIM_HINT.get(d, ""))
    if it.grain == "week":
        parts.append("weeks start on Monday; group by date(created_at, 'weekday 0', '-6 days')")
    elif it.grain == "hour":
        parts.append("hour refers to STRFTIME('%Y-%m-%d %H:00', created_at)")
    elif it.grain == "day":
        parts.append("day refers to date(created_at)")
    parts.append("status is one of 'success', 'failed', 'user_cancelled'; "
                 "method is one of 'card', 'upi', 'netbanking', 'wallet'")
    return "; ".join(p for p in parts if p)


def question_for(it):
    """Heron sees the user's words, with the resolved period made explicit."""
    q = it.question.strip()
    if it.period:
        a, b = it.period
        q = f"{q} (period: {a} to {b})"
    return q


def _literal_mask(sql):
    """True for characters inside '...', "..." or `...` (parens there don't count)."""
    mask, q = [], None
    for ch in sql:
        if q:
            mask.append(True)
            if ch == q:
                q = None
        elif ch in "'\"`":
            q = ch
            mask.append(True)
        else:
            mask.append(False)
    return mask


def balance_parens(sql, dialect="sqlite", max_tries=256):
    """Remove surplus ')' -- only at '))' sites -- until the SQL parses.

    Why this exists: the deployed Heron endpoint emits a single '))' token where
    the offline evaluation of the same model emitted ')' (e.g. COUNT(x)) FROM).
    The fault is in serving, not in the prompt; see README "Known issues".
    Deterministic and conservative like repair.py: it never adds anything, only
    tries removals at doubled parens, and gives up (returns the input) unless
    exactly the surplus is removed and sqlglot accepts the result.
    """
    import itertools
    mask = _literal_mask(sql)
    opens = sum(1 for c, m in zip(sql, mask) if c == "(" and not m)
    closes = sum(1 for c, m in zip(sql, mask) if c == ")" and not m)
    excess = closes - opens
    if excess <= 0:
        return sql, 0
    sites = [i for i in range(1, len(sql)) if sql[i] == ")" and sql[i - 1] == ")"
             and not mask[i] and not mask[i - 1]]
    for n, combo in enumerate(itertools.combinations(sites, excess)):
        if n >= max_tries:
            break
        cand = "".join(c for i, c in enumerate(sql) if i not in combo)
        try:
            sqlglot.parse_one(cand, read=dialect)
            return cand, excess
        except Exception:                                          # noqa: BLE001
            continue
    return sql, 0


def fix_date_between(sql):
    """`created_at BETWEEN '2026-09-01' AND '2026-09-30'` on a TIMESTAMP silently drops
    the last day after midnight. Rewrite to a half-open range. Deterministic."""
    from datetime import date as _d, timedelta as _td
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:                                              # noqa: BLE001
        return sql, False
    changed = False

    def tf(node):
        nonlocal changed
        if isinstance(node, exp.Between) and isinstance(node.this, exp.Column) \
                and node.this.name.lower() in ("created_at", "hour"):
            hi = node.args["high"]
            if isinstance(hi, exp.Literal) and hi.is_string and len(hi.this) == 10:
                try:
                    nxt = _d.fromisoformat(hi.this) + _td(days=1)
                except ValueError:
                    return node
                changed = True
                col = node.this
                return exp.paren(exp.and_(exp.GTE(this=col.copy(), expression=node.args["low"].copy()),
                                          exp.LT(this=col.copy(), expression=exp.Literal.string(str(nxt)))))
        return node

    out = tree.transform(tf)
    return (out.sql(dialect="duckdb"), True) if changed else (sql, False)


def semantic_lint(sql, it):
    """Known-wrong patterns for this metric -> a reason to send back to the writer."""
    from sqlglot import exp
    from .compiler import RATE_METRICS
    low = sql.lower()
    if it.metric in RATE_METRICS and "user_cancelled" not in low:
        return ("A success rate must exclude cancelled attempts: add status <> 'user_cancelled' "
                "to the WHERE clause.")
    if it.metric in RATE_METRICS and it.order and (it.group_by or it.grain):
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
            order = tree.args.get("order")
            if order is not None and not any(True for _ in order.find_all(exp.Div)) \
                    and "value" not in order.sql().lower() and not any(
                        isinstance(p, exp.Alias) and any(True for _ in p.find_all(exp.Div))
                        and p.alias.lower() in order.sql().lower() for p in tree.expressions):
                return ("Rank by the success rate (successes / attempts), not by a count; also "
                        "return the rate as a column.")
        except Exception:                                          # noqa: BLE001
            pass
    if it.metric in RATE_METRICS and "/" not in sql:
        return "Return the success rate itself (100.0 * successes / attempts) as a column."
    return None


def _schema_cols(rdb):
    return {t: list(cols) for t, cols in rdb.columns.items()}


def _to_duckdb(sql):
    return sqlglot.transpile(sql, read="sqlite", write="duckdb")[0]


def _run_loop(it, rdb, provider, gen, max_tries, timeout):
    attempts, feedback = [], None
    for _ in range(max_tries):
        text, info = gen(feedback)
        a = SQLAttempt(provider=provider, ms=info.get("ms", 0.0) if info else 0.0)
        attempts.append(a)
        if text is None:
            a.error = (info or {}).get("error", "no response")
            return None, attempts, None
        a.raw = text
        sql = clean_sql(text)
        if provider == "heron":
            sql, n_paren = balance_parens(sql)
            sql, changes = repair_identifiers(sql, _schema_cols(rdb), dialect="sqlite")
            a.repaired = changes + ([("))", ")", f"removed {n_paren} surplus ')'")] if n_paren else [])
            try:
                sql = _to_duckdb(sql)
            except Exception as e:                                 # noqa: BLE001
                a.error = f"transpile failed: {e}"
                feedback = "The query did not parse. Write one valid SQLite SELECT."
                continue
        sql, fixed = fix_date_between(sql)
        if fixed:
            a.repaired = list(a.repaired) + [("BETWEEN", ">= / <", "half-open date range")]
        a.sql = sql
        lint = semantic_lint(sql, it)
        if lint:
            a.check_code, a.reason = "semantic_lint", lint
            feedback = lint
            continue
        chk = guardrails.check(sql, rdb)
        a.check_code = chk.code or "passed"
        if not chk.ok:
            a.reason = chk.reason
            if chk.code in HARD_STOP:
                return None, attempts, chk
            feedback = chk.reason
            continue
        df, err, ms = rdb.run(chk.sql, timeout=timeout)
        if err:
            a.error = err
            feedback = f"The query failed: {err.splitlines()[0][:200]}"
            continue
        return SQLResult(ok=True, sql=chk.sql, df=df, provider=provider, run_ms=ms,
                         check=chk), attempts, None
    return None, attempts, None


def generate(it, rdb):
    lim = policy()["limits"]
    retries, timeout = lim["max_sql_retries"], lim["timeout_seconds"]
    ddl = rdb.schema_ddl()
    know = knowledge_for(it, rdb)
    q = question_for(it)
    all_attempts = []

    def heron_gen(feedback):
        k = know if not feedback else f"{know}; previous attempt was rejected: {feedback}"
        return llm.heron_sql(HERON_SYSTEM, heron_user_block(ddl, k, q))

    if llm.heron_enabled():
        res, atts, stop = _run_loop(it, rdb, "heron", heron_gen, 1 + retries, timeout)
        all_attempts += atts
        if res or stop:
            if res:
                res.attempts = all_attempts
                return res
            return SQLResult(ok=False, attempts=all_attempts, stop=stop)

    if llm.hosted_available():
        sys_prompt = ("You write one DuckDB SQL SELECT for a payments database. Respond with ONLY "
                      "the SQL. Use only the tables and columns in the schema.")

        def hosted_gen(feedback):
            u = f"Schema:\n{ddl}\n\nKnowledge: {know}\nQuestion: {q}"
            if feedback:
                u += f"\nPrevious attempt was rejected: {feedback}"
            text, prov = llm.chat_text(sys_prompt, u, max_tokens=400)
            return text, {"provider": prov}

        res, atts, stop = _run_loop(it, rdb, "hosted", hosted_gen, 1, timeout)
        all_attempts += atts
        if res:
            res.attempts = all_attempts
            return res
        if stop:
            return SQLResult(ok=False, attempts=all_attempts, stop=stop)
    return SQLResult(ok=False, attempts=all_attempts)


def heron_prompt_preview(it, rdb):
    return heron_user_block(rdb.schema_ddl(), knowledge_for(it, rdb), question_for(it))

