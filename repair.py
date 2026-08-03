"""Deterministic identifier repair guard. LOCAL, microseconds, no LLM call.

The failure this exists for: the model overrides correct DDL with its spelling
prior. BIRD's posts table has a column literally spelled `CreaionDate`; the
model writes the correctly-spelled `CreationDate` and fails. Laboratory has
`HGB`, the model writes `HBG`. frpm has `School Name` with a space, the model
writes `SchoolName`. A system-prompt instruction to copy names verbatim does not
fix this -- it is a one-line instruction against a pretraining-scale prior.

The critical design constraint: repair is scoped to columns of tables ALREADY
REFERENCED in the query. An unconstrained nearest-match repairer that reaches
across the whole database will convert misspelling errors into join-path errors,
which are harder to diagnose and more numerous. Scoped this way the guard cannot
touch join logic -- it can only fix a name on a table the model already chose.

Because this changes reported accuracy, score.py runs it as a separate column
(`correct_guarded`) so the final table can show guarded and unguarded rows and
nobody can claim the guard is doing the work.
"""

import difflib
import re

MIN_RATIO = 0.82        # similarity floor for accepting a rename
MAX_EDIT = 1            # or: within this Damerau-Levenshtein distance


def _damerau(a, b):
    """Edit distance WITH transpositions. Plain difflib ratio is useless on
    short identifiers -- HBG vs HGB scores 0.67, well under any sane cutoff,
    because a 3-character transposition destroys most of the matching blocks.
    Treating a swap as one edit catches HBG->HGB and CreationDate->CreaionDate
    alike, and the MAX_EDIT=1 ceiling keeps it from reaching further.
    """
    a, b = a.lower(), b.lower()
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _best_match(name, pool):
    """Unambiguous nearest column, or None.

    Two acceptance rules, unioned: high difflib ratio (good for long names and
    for space/case differences like SchoolName -> School Name) and small
    transposition-aware edit distance (good for short names). Returns None if
    zero candidates OR more than one, because an ambiguous repair is worse than
    no repair -- it invents a column the model never chose.
    """
    cands = set(difflib.get_close_matches(name, pool, n=3, cutoff=MIN_RATIO))
    cands |= {c for c in pool if _damerau(name, c) <= MAX_EDIT}
    cands.discard(name)
    return next(iter(cands)) if len(cands) == 1 else None


def _sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
        return sqlglot, exp
    except ImportError:
        return None, None


def _needs_quoting(name):
    return bool(re.search(r"[^A-Za-z0-9_]", name))


def repair_identifiers(sql, schema, dialect="sqlite"):
    """Return (repaired_sql, list_of_changes).

    schema: {table_name: [column_names]} from utils.schema_columns().
    If sqlglot is unavailable or the SQL is unparseable, returns the input
    unchanged -- the guard must never make things worse.
    """
    sqlglot, exp = _sqlglot()
    if sqlglot is None:
        return sql, []
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return sql, []

    # Tables actually referenced, matched case-insensitively against the schema.
    lower_tables = {t.lower(): t for t in schema}
    referenced = []
    for t in tree.find_all(exp.Table):
        real = lower_tables.get((t.name or "").lower())
        if real and real not in referenced:
            referenced.append(real)
    if not referenced:
        return sql, []

    # Candidate pool = columns of referenced tables only.
    pool, pool_lower = [], {}
    for t in referenced:
        for c in schema[t]:
            if c not in pool:
                pool.append(c)
                pool_lower[c.lower()] = c
    valid_lower = set(pool_lower)

    changes = []
    for col in tree.find_all(exp.Column):
        name = col.name
        if not name or name == "*":
            continue
        if name.lower() in valid_lower:
            # Correct name, but check quoting: `School Name` must stay quoted or
            # SQLite parses it as two tokens. A model that learned unquoted
            # identifiers fails these even when it copies the characters right.
            real = pool_lower[name.lower()]
            if _needs_quoting(real) and not col.this.quoted:
                col.set("this", exp.to_identifier(real, quoted=True))
                changes.append((name, real, "quoted"))
            continue

        target = _best_match(name, pool)
        if target is None:
            continue                      # zero matches, or ambiguous -> leave alone
        col.set("this", exp.to_identifier(target, quoted=_needs_quoting(target)))
        changes.append((name, target, "renamed"))

    if not changes:
        return sql, []
    try:
        return tree.sql(dialect=dialect), changes
    except Exception:
        return sql, []


def self_test():
    """Sanity checks for the cases this guard was built for."""
    schema = {
        "posts": ["Id", "CreaionDate", "ViewCount", "OwnerUserId"],
        "users": ["Id", "DisplayName", "Reputation"],
        "Laboratory": ["ID", "Date", "HGB", "GPT"],
        "frpm": ["CDSCode", "School Name", "Enrollment (K-12)"],
    }
    cases = [
        ("SELECT Id FROM posts WHERE STRFTIME('%Y', CreationDate) = '2010'",
         "CreaionDate"),
        ("SELECT HBG FROM Laboratory", "HGB"),
        ("SELECT SchoolName FROM frpm", "School Name"),
    ]
    ok = True
    for sql, expect in cases:
        out, ch = repair_identifiers(sql, schema)
        hit = expect.lower() in out.lower()
        print(f"{'PASS' if hit else 'FAIL'}  {sql}\n      -> {out}  {ch}")
        ok = ok and hit

    # Must NOT reach across to an unreferenced table.
    out, ch = repair_identifiers("SELECT Reputaton FROM posts", schema)
    crossed = "reputation" in out.lower()
    print(f"{'PASS' if not crossed else 'FAIL'}  cross-table guard: {out} {ch}")
    return ok and not crossed


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
