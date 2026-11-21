"""Query checks before anything runs. Layer 2 of the two-layer protection.

The SQL is parsed into a tree with sqlglot and rejected unless ALL of these hold:
  1. exactly one statement, and it only reads (SELECT / set operations)
  2. no function that reads files, the network or settings
  3. every table is on the role's list (and no catalog-qualified names)
  4. no PII column unless the role may see (masked) PII
  5. no literal filter on a region outside the role's scope
  6. queries on transactions carry a date filter spanning <= 180 days
  7. every join has a join condition
  8. DuckDB's EXPLAIN estimate scans <= 5M rows; non-aggregate output <= 10k rows
A LIMIT is added (or tightened) to max_return_rows.

Each failure carries a machine code, a reason the SQL writer can act on, and
a user-facing message that never reveals what hidden data contains.
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import sqlglot
from sqlglot import exp

from .db import ALL_TABLES
from .settings import data_meta, policy

_WRITE_NODES = tuple(getattr(exp, n) for n in (
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Copy", "Pragma",
    "Set", "Attach", "Detach", "Merge", "TruncateTable", "Use", "Grant", "Transaction",
    "Commit", "Rollback", "LoadData", "Export", "Install", "Load", "Describe", "Show",
) if hasattr(exp, n))
_READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?")


@dataclass
class Check:
    ok: bool
    sql: str = ""
    code: str = ""
    reason: str = ""              # for the SQL writer's retry
    message: str = ""             # for the user
    scanned: int = 0
    est_rows: int = 0
    span_days: int = 0
    notes: list = field(default_factory=list)


def _fail(code, reason, message=None, **kw):
    return Check(ok=False, code=code, reason=reason, message=message or reason, **kw)


def _lit_date(s, end=False):
    """'2026-09-01', '2026-09-01 10:00', '2026-09' -> date. Month-only literals
    resolve to the first (or, with end=True, last) day of that month."""
    m = _DATE_RE.match(str(s).strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
    try:
        if d:
            return date(y, mo, int(d))
        first = date(y, mo, 1)
        if not end:
            return first
        nxt = date(y + (mo == 12), mo % 12 + 1, 1)
        return nxt - timedelta(days=1)
    except ValueError:
        return None


def _date_bounds(tree, date_cols=("created_at", "hour", "settlement_date", "started_at")):
    """(lo, hi) of literal dates compared against a time column; None if absent.

    Handles col >= / > / < / <= / = literal, BETWEEN, IN, LIKE '2026-09%', and
    wrappers such as CAST(col AS DATE), strftime(col, '%Y-%m'), date_trunc(...).
    """
    los, his = [], []

    def touches_time(node):
        return any(c.name.lower() in date_cols for c in node.find_all(exp.Column))

    def lits(node):
        return [l.this for l in node.find_all(exp.Literal) if l.is_string]

    for node in tree.find_all(exp.GTE, exp.GT, exp.LT, exp.LTE, exp.EQ, exp.Between,
                              exp.In, exp.Like):
        if isinstance(node, exp.Between):
            if not touches_time(node.this):
                continue
            a = _lit_date(node.args["low"].sql().strip("'"))
            b = _lit_date(node.args["high"].sql().strip("'"), end=True)
            if a:
                los.append(a)
            if b:
                his.append(b)
            continue
        if isinstance(node, exp.In):
            if not touches_time(node.this):
                continue
            ds = [_lit_date(v) for v in lits(node)]
            ds = [d for d in ds if d]
            los += ds
            his += [_lit_date(v, end=True) for v in lits(node) if _lit_date(v)]
            continue
        left, right = node.left, node.right
        col_side, lit_side = (left, right) if touches_time(left) else (right, left)
        if not touches_time(col_side) or touches_time(lit_side):
            continue
        strs = lits(lit_side)
        if not strs:
            continue
        if isinstance(node, exp.Like):
            d0 = _lit_date(strs[0].rstrip("%"))
            d1 = _lit_date(strs[0].rstrip("%"), end=True)
            if d0:
                los.append(d0)
                his.append(d1)
            continue
        d = _lit_date(strs[0])
        if not d:
            continue
        # flip the operator if the literal is on the left
        flipped = lit_side is left
        is_lo = isinstance(node, (exp.GTE, exp.GT)) != flipped
        if isinstance(node, exp.EQ):
            los.append(d)
            his.append(_lit_date(strs[0], end=True))
        elif is_lo:
            los.append(d)
        else:
            his.append(d)
    lo = min(los) if los else None
    hi = max(his) if his else None
    return lo, hi


def _cte_names(tree):
    return {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}


def check(sql, rdb, dialect="duckdb"):
    """Validate `sql` for the role database `rdb` (a db.RoleDB)."""
    pol = policy()
    limits = pol["limits"]
    blocked = {f.lower() for f in pol["blocked_functions"]}
    spec = rdb.spec

    if not sql or not sql.strip():
        return _fail("empty", "No SQL was produced.")
    try:
        stmts = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        return _fail("parse_error", f"SQL does not parse: {str(e).splitlines()[0]}",
                     "The generated query could not be read.")
    if len(stmts) != 1:
        return _fail("multi_statement", "Exactly one SQL statement is allowed.",
                     "Only a single read-only query can run.")
    tree = stmts[0]

    # 1. read-only
    if not isinstance(tree, _READ_ROOTS) or any(True for _ in tree.find_all(*_WRITE_NODES)):
        kind = type(tree).__name__.upper()
        return _fail("not_read_only", f"{kind} statements are not allowed; only SELECT.",
                     "PayInsight is read-only: it cannot change, delete or load data.")

    # 2. functions
    for fn in tree.find_all(exp.Func):
        name = (fn.sql_name() if not isinstance(fn, exp.Anonymous) else fn.name).lower()
        if name in blocked or (isinstance(fn, exp.Anonymous) and fn.name.lower() in blocked):
            return _fail("blocked_function", f"Function {name}() is not allowed.",
                         "That query uses a function that reads files or settings, which is not allowed.")
    for t in tree.find_all(exp.Table):
        if isinstance(t.this, exp.Func) or isinstance(t.this, exp.Anonymous):
            return _fail("blocked_function", "Table functions are not allowed.",
                         "That query tries to read from outside the dataset.")

    # 3. tables
    ctes = _cte_names(tree)
    allowed = {t.lower() for t in rdb.tables}
    used = set()
    for t in tree.find_all(exp.Table):
        name = (t.name or "").lower()
        if t.args.get("db") or t.args.get("catalog"):
            return _fail("unknown_table", f"Qualified table name {t.sql()} is not allowed.",
                         "That query names a table outside your workspace.")
        if name in ctes:
            continue
        if name not in allowed:
            if name in ALL_TABLES or name in ("transactions_clean",):
                label = name.replace("_", " ")
                return _fail("forbidden_table", f"Table {name} is outside this role's access.",
                             f"Your role ({spec['label']}) does not include access to {label}.")
            return _fail("unknown_table", f"Table {name} does not exist. Tables: {sorted(allowed)}.",
                         "The query referred to a table that does not exist.")
        used.add(name)

    # 4. PII
    pii = set(pol["pii_columns"])
    for c in tree.find_all(exp.Column):
        if c.name.lower() in pii and spec.get("pii", "none") == "none":
            return _fail("pii", f"Column {c.name} is customer PII.",
                         "Customer personal data (phone numbers, UPI IDs) is not available to your role.")

    # 5. region scope
    regions = spec.get("regions", "all")
    if regions != "all":
        ok_regions = {r.lower() for r in regions}
        for node in tree.find_all(exp.EQ, exp.In, exp.NEQ):
            cols = [c for c in node.find_all(exp.Column) if c.name.lower() == "region"]
            if not cols:
                continue
            vals = {l.this.lower() for l in node.find_all(exp.Literal) if l.is_string}
            bad = vals - ok_regions
            if bad and not isinstance(node, exp.NEQ):
                return _fail("region_scope", f"Region filter {sorted(bad)} is outside scope {regions}.",
                             f"Your role covers the {', '.join(regions)} region only.")

    # 6. date filter on transactions
    span = 0
    if "transactions" in used or "transactions_hourly" in used:
        lo, hi = _date_bounds(tree)
        meta = data_meta()
        if lo is None and hi is None:
            return _fail("no_date_filter",
                         "Queries on transactions must filter created_at to a date range "
                         "(created_at >= 'YYYY-MM-DD' AND created_at < 'YYYY-MM-DD').",
                         "Queries on transactions need a date range (at most "
                         f"{limits['max_range_days']} days).")
        lo = lo or meta["start_date"]
        hi = hi or meta["as_of_date"]
        span = (hi - lo).days
        if span > limits["max_range_days"]:
            return _fail("range_too_long",
                         f"Date range spans {span} days; the limit is {limits['max_range_days']}.",
                         f"That date range covers {span} days; the limit is "
                         f"{limits['max_range_days']}. Try a shorter period.", span_days=span)

    # 7. joins need conditions
    for j in tree.find_all(exp.Join):
        if j.args.get("on") or j.args.get("using"):
            continue
        if (j.args.get("kind") or "").upper() == "CROSS" or not _implicit_join_ok(tree, j):
            return _fail("cross_join", "Every join needs an ON or USING condition.",
                         "That query joins tables without a join condition.")

    # LIMIT: add or tighten
    max_rows = limits["max_return_rows"]
    final = tree
    if isinstance(tree, exp.Select):
        lim = tree.args.get("limit")
        cur = None
        if lim is not None:
            try:
                cur = int(lim.expression.this) if lim.expression is not None else int(lim.this)
            except (AttributeError, TypeError, ValueError):
                cur = None
        if cur is None or cur > max_rows:
            final = tree.limit(max_rows)
    out_sql = final.sql(dialect="duckdb")

    # 8. cost
    try:
        scanned, est = rdb.explain(out_sql)
    except Exception as e:                                       # noqa: BLE001
        return _fail("plan_error", f"DuckDB rejected the query: {str(e).splitlines()[0]}",
                     "The query refers to something that does not exist.")
    if scanned > limits["max_scan_rows"]:
        return _fail("too_expensive", f"Estimated scan {scanned:,} rows > {limits['max_scan_rows']:,}.",
                     "That query would scan too much data. A summary table or a shorter period would work.",
                     scanned=scanned)
    agg = any(True for _ in tree.find_all(exp.AggFunc)) or bool(tree.args.get("group"))
    raw_est = _raw_estimate(rdb, tree.sql(dialect="duckdb")) if not agg else est
    if not agg and raw_est > max_rows:
        return _fail("too_many_rows",
                     f"Non-aggregate query would return ~{raw_est:,} rows (> {max_rows:,}).",
                     f"That would return about {raw_est:,} rows, over the {max_rows:,}-row limit. "
                     f"I can show a summary table instead, or a sample of "
                     f"{limits['sample_rows']:,} rows.", scanned=scanned, est_rows=raw_est)
    return Check(ok=True, sql=out_sql, scanned=scanned, est_rows=est, span_days=span)


def _raw_estimate(rdb, sql):
    try:
        return rdb.explain(sql)[1]
    except Exception:                                            # noqa: BLE001
        return 0


def _implicit_join_ok(tree, join):
    """`FROM a, b WHERE a.x = b.y` is fine; `FROM a, b` alone is not."""
    alias = (join.this.alias_or_name or "").lower()
    where = tree.args.get("where")
    if not where:
        return False
    for eq in where.find_all(exp.EQ):
        if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
            tabs = {(eq.left.table or "").lower(), (eq.right.table or "").lower()}
            if alias in tabs:
                return True
    return False
