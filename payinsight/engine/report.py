"""Weekly KPI report: the page a payments leader opens on Monday morning.

Each week is compared with the 4 weeks before it, the same baseline the "why" analysis uses,
so the drop shown here is the drop PayInsight then explains. Queries go through the normal
checked Runner, so they pass the same guardrails and are recorded as evidence.
"""

from datetime import timedelta

import pandas as pd

from .compiler import metric_exprs
from .evidence import EvidenceLog, Runner
from .settings import data_meta

BASELINE_WEEKS = 4


def weekly(rdb):
    """-> (DataFrame week/week_end/attempts/successes/failures/value/rate, Evidence) or (None, None)."""
    if "transactions" in rdb.tables:
        table, tcol = "transactions", "created_at"
    elif "transactions_hourly" in rdb.tables:
        table, tcol = "transactions_hourly", "hour"
    else:
        return None, None
    _, n, k, where = metric_exprs("success_rate", table)
    fails, _, _, _ = metric_exprs("failures", table)
    value, _, _, _ = metric_exprs("successful_value", table)
    meta = data_meta()
    a, b = meta["start_date"], meta["end_date"]
    first = a + timedelta(days=(7 - a.weekday()) % 7)              # first Monday: whole weeks only
    last = b - timedelta(days=(b.weekday() + 1) % 7)               # last Sunday on or before the end
    sql = (f"SELECT CAST(date_trunc('week', t.{tcol}) AS DATE) AS week, {n} AS attempts, "
           f"{k} AS successes, {fails} AS failures, {value} AS value "
           f"FROM {table} t WHERE {where} AND t.{tcol} >= '{first}' "
           f"AND t.{tcol} < '{last + timedelta(days=1)}' GROUP BY 1 ORDER BY 1")
    runner = Runner(rdb, EvidenceLog())
    df, ev = runner.run(sql, title="Weekly KPIs", dataset=table,
                        window=f"{first:%d %b} – {last:%d %b %Y}")
    df["week"] = pd.to_datetime(df.week)
    df["week_end"] = df.week + pd.Timedelta(days=6)
    df["rate"] = 100.0 * df.successes / df.attempts
    return df, ev


def compare(df, i):
    """Week i against the pooled 4 weeks before it. -> dict, or None if there is no full baseline."""
    if i < BASELINE_WEEKS:
        return None
    cur, base = df.iloc[i], df.iloc[i - BASELINE_WEEKS:i]
    base_rate = 100.0 * base.successes.sum() / base.attempts.sum()
    avg = lambda c: base[c].mean()                                          # noqa: E731
    return {"rate": cur.rate, "base_rate": base_rate, "pp": cur.rate - base_rate,
            "attempts": cur.attempts, "attempts_pct": 100 * (cur.attempts / avg("attempts") - 1),
            "value": cur.value, "value_pct": 100 * (cur.value / avg("value") - 1),
            "failures": cur.failures, "failures_pct": 100 * (cur.failures / avg("failures") - 1),
            "base_start": base.week.iloc[0], "base_end": base.week_end.iloc[-1]}


def largest_drop(df):
    """Index of the week with the largest drop against its baseline (None if no drop)."""
    best, at = 0.0, None
    for i in range(BASELINE_WEEKS, len(df)):
        pp = compare(df, i)["pp"]
        if pp < best:
            best, at = pp, i
    return at
