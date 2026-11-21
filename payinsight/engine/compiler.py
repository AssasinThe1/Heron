"""Semantic layer -> exact DuckDB SQL, deterministically.

Two jobs:
  * the independent cross-check for every query Heron writes (same question,
    computed straight from the definitions in semantic_layer.yaml)
  * the fallback when Heron is unreachable, and the fixed templates behind the
    root-cause and reconciliation paths

Output columns are predictable so the answer template can read them:
  [<time bucket>] [<dimension>...] value  n  [k]
where n is the denominator (attempts / orders) and k the numerator for rates.
"""

from dataclasses import dataclass
from datetime import timedelta

RATE_METRICS = {"success_rate", "order_success_rate"}

DIM_EXPR = {
    "method": ("t.method", None), "region": ("t.region", None), "device": ("t.device", None),
    "failure_code": ("t.failure_code", None),
    "bank": ("COALESCE(b.name, '(no bank recorded)')", "bank"),
    "bank_type": ("COALESCE(b.bank_type, '(no bank recorded)')", "bank"),
    "merchant": ("m.name", "merchant"), "category": ("m.category", "merchant"),
    "tier": ("m.tier", "merchant"),
}
FILTER_EXPR = {"bank": "b.name", "bank_type": "b.bank_type", "merchant": "m.name",
               "category": "m.category", "tier": "m.tier"}
GRAIN_EXPR = {
    "hour": "date_trunc('hour', {t}) AS hour",
    "day": "CAST({t} AS DATE) AS day",
    "week": "CAST(date_trunc('week', {t}) AS DATE) AS week",
    "month": "strftime({t}, '%Y-%m') AS month",
}


class CompileError(ValueError):
    """The question can't be expressed against this role's tables."""


@dataclass
class Compiled:
    sql: str
    dims: list
    bucket: str
    metric: str
    unit: str
    table: str


def _q(v):
    return "'" + str(v).replace("'", "''") + "'"


def _source(rdb, metric):
    if metric == "settled_net":
        if "settlements" not in rdb.tables:
            raise CompileError("settlements")
        return "settlements"
    if "transactions" in rdb.tables:
        return "transactions"
    if "transactions_hourly" in rdb.tables:
        return "transactions_hourly"
    raise CompileError("transactions")


def metric_exprs(metric, table):
    """-> (value_expr, n_expr, k_expr | None, extra_where | None)."""
    if table == "transactions":
        cnt = "COUNT(DISTINCT t.txn_id)"
        cnt_if = lambda c: f"COUNT(DISTINCT t.txn_id) FILTER (WHERE {c})"   # noqa: E731
        amt_if = lambda c: f"SUM(t.amount_inr) FILTER (WHERE {c})"          # noqa: E731
    else:                                       # transactions_hourly (pre-aggregated)
        cnt = "SUM(t.attempts)"
        cnt_if = lambda c: f"SUM(t.attempts) FILTER (WHERE {c})"            # noqa: E731
        amt_if = lambda c: f"SUM(t.amount_inr) FILTER (WHERE {c})"          # noqa: E731
    ok = "t.status = 'success'"
    if metric == "success_rate":
        return (f"ROUND(100.0 * {cnt_if(ok)} / NULLIF({cnt}, 0), 4)", cnt, cnt_if(ok),
                "t.status <> 'user_cancelled'")
    if metric == "order_success_rate":
        if table != "transactions":
            raise CompileError("order-level detail")
        return ("ROUND(100.0 * COUNT(DISTINCT t.order_id) FILTER (WHERE t.status = 'success') "
                "/ NULLIF(COUNT(DISTINCT t.order_id), 0), 4)", "COUNT(DISTINCT t.order_id)",
                "COUNT(DISTINCT t.order_id) FILTER (WHERE t.status = 'success')",
                "t.status <> 'user_cancelled'")
    if metric == "attempts":
        return cnt, cnt, None, None
    if metric == "failures":
        return cnt_if("t.status = 'failed'"), cnt, None, None
    if metric == "successful_value":
        return f"ROUND({amt_if(ok)}, 2)", cnt_if(ok), None, None
    if metric == "avg_ticket":
        return f"ROUND({amt_if(ok)} / NULLIF({cnt_if(ok)}, 0), 2)", cnt_if(ok), None, None
    if metric == "settled_net":
        return "ROUND(SUM(t.net_inr), 2)", "COUNT(*)", None, None
    raise CompileError(f"metric {metric}")


def rank_floor(it):
    """Smallest segment a rate ranking considers: 100 attempts, 30 for single hours."""
    return 30 if it.grain == "hour" else 100


def compile_intent(it, rdb):
    from .settings import semantic
    metric = it.metric
    table = _source(rdb, metric)
    value, n, k, extra = metric_exprs(metric, table)
    tcol = {"transactions": "t.created_at", "transactions_hourly": "t.hour",
            "settlements": "t.settlement_date"}[table]

    joins, select, groups = set(), [], []
    if it.grain:
        select.append(GRAIN_EXPR[it.grain].format(t=tcol))
        groups.append(it.grain)
    dims = []
    for d in it.group_by:
        expr, j = DIM_EXPR[d]
        if table == "settlements":
            if d not in ("merchant", "category", "tier", "region"):
                raise CompileError(f"{d} on settlements")
            expr, j = (f"m.{'name' if d == 'merchant' else d}", "merchant")
        if table == "transactions_hourly" and d in ("bank", "bank_type", "device", "failure_code"):
            raise CompileError(f"{d} detail")
        if j:
            joins.add(j)
        select.append(f"{expr} AS {d}")
        dims.append(d)
    select += [f"{value} AS value", f"{n} AS n"]
    if k:
        select.append(f"{k} AS k")

    where = []
    if it.period:
        a, b = it.period
        where += [f"{tcol} >= {_q(a)}", f"{tcol} < {_q(b + timedelta(days=1))}"]
    if extra:
        where.append(extra)
    for d, vals in (it.filters or {}).items():
        if not vals:
            continue
        if table == "settlements":
            if d not in ("merchant", "category", "tier", "region"):
                raise CompileError(f"{d} on settlements")
            col = f"m.{'name' if d == 'merchant' else d}"
            joins.add("merchant")
        elif d in FILTER_EXPR:
            if table == "transactions_hourly" and d in ("bank", "bank_type"):
                raise CompileError("bank detail")
            col = FILTER_EXPR[d]
            joins.add("bank" if d.startswith("bank") else "merchant")
        else:
            col = f"t.{d}"
        where.append(f"{col} IN ({', '.join(_q(v) for v in vals)})")

    frm = f"{table} t"
    if "bank" in joins:
        frm += " LEFT JOIN banks b ON t.bank_id = b.bank_id"
    if "merchant" in joins:
        frm += " LEFT JOIN merchants m ON t.merchant_id = m.merchant_id"

    sql = f"SELECT {', '.join(select)} FROM {frm}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if groups or dims:
        sql += " GROUP BY " + ", ".join(str(i + 1) for i in range(len(groups) + len(dims)))
        if it.order and metric in RATE_METRICS:
            sql += f" HAVING {n} >= {rank_floor(it)}"
        if it.order:
            nulls = "NULLS LAST"
            sql += f" ORDER BY value {'ASC' if it.order == 'asc' else 'DESC'} {nulls}"
        else:
            sql += " ORDER BY " + ", ".join(str(i + 1) for i in range(len(groups) + len(dims)))
        if it.limit:
            sql += f" LIMIT {int(it.limit)}"
    unit = semantic()["metrics"][metric]["unit"]
    return Compiled(sql=sql, dims=dims, bucket=it.grain, metric=metric, unit=unit, table=table)
