"""Failure taxonomy for text-to-SQL misses. LOCAL.

Only ~11% of misses are hard errors (query did not execute). The other ~89%
execute cleanly and return the wrong rows, and until you can categorize those
mechanically you are optimizing blind.

Two layers, deliberately separated by cost:

  shape_kind()  -- free. Compares result-set geometry. No parsing, no deps.
  ast_kind()    -- needs sqlglot. Structural diff of the two queries.

Cross-tabulating the two is the point: shape says how the answer was wrong, AST
says what about the query made it wrong, and the cells map onto interventions.

IMPORTANT CAVEAT on shape: the scorer compares SETS of tuples, so duplicate
rows are already collapsed before these functions see anything. "row count"
below therefore means DISTINCT row count. That is the right denominator for a
set-equality metric, but it means shape_kind cannot detect a missing DISTINCT
or a join fan-out that produced duplicates only.
"""

import re

# --------------------------------------------------------------------------
# error-string taxonomy (hard errors)
# --------------------------------------------------------------------------

ERR_KINDS = [
    ("no such column", "missing_column"),
    ("no such table", "missing_table"),
    ("no such function", "bad_function"),
    ("misuse of aggregate", "bad_aggregate"),
    ("ambiguous column name", "ambiguous_column"),
    ("syntax error", "syntax"),
    ("incomplete input", "truncated"),
    ("interrupted", "timeout"),
]


def err_kind(e):
    if not e:
        return None
    low = str(e).lower()
    for needle, label in ERR_KINDS:
        if needle in low:
            return label
    return "other"


_COL_RE = re.compile(r"no such column:\s*(?:[\w\"`\[\]]+\.)?(.+)", re.IGNORECASE)


def missing_column_subkind(err, all_cols_lower):
    """Split `missing_column` into the two classes that need opposite fixes.

    wrong_table : the column exists in this database, just not on the table the
                  model aliased it to. A join-path failure. Fix = FK edge list.
    bad_name    : the column does not exist anywhere. The model "corrected" a
                  broken identifier -- BIRD's posts table really does have a
                  column spelled CreaionDate, and Laboratory really has HGB not
                  HBG. Fix = the mechanical repair guard, not instructions.
    """
    m = _COL_RE.search(str(err or ""))
    if not m:
        return None
    name = m.group(1).strip().strip('"`[]')
    return "wrong_table" if name.lower() in all_cols_lower else "bad_name"


# --------------------------------------------------------------------------
# layer 1: result-shape triage
# --------------------------------------------------------------------------

def _ncols(s):
    if not s:
        return 0
    return len(next(iter(s)))


def shape_kind(pset, gset):
    """Mechanically categorize a silent failure by result geometry.

    pred_empty        predicted returned nothing, gold returned rows. Almost
                      always a wrong string/date literal -- this single flag
                      isolates most of the literal-format class and is the
                      direct before/after measure for the sample-values arm.
    gold_empty        gold returned nothing but prediction did. Often a gold
                      quirk; inspect these, they are cheap suspicion.
    ncol_mismatch     SELECT-clause disagreement. Counted separately because
                      BIRD's strict tuple equality fails predictions carrying
                      extra but harmless columns -- this is the quantifiable
                      part of the "unfair penalty" class.
    superset          predicted contains gold -> missed filter or join fan-out.
    subset            gold contains predicted -> over-filtered.
    same_n_diff_vals  same row count, different values -> right rows, wrong
                      column projected, or wrong aggregate.
    disjoint          no overlap at all -> wrong query, not a near miss.
    partial_overlap   some overlap, neither contains the other.
    """
    if pset is None or gset is None:
        return None
    if not pset and gset:
        return "pred_empty"
    if pset and not gset:
        return "gold_empty"
    if not pset and not gset:
        return "both_empty"
    if _ncols(pset) != _ncols(gset):
        return "ncol_mismatch"
    if pset == gset:
        return "match"
    if pset > gset:
        return "superset"
    if pset < gset:
        return "subset"
    if not (pset & gset):
        return ("same_n_diff_vals" if len(pset) == len(gset) else "disjoint")
    return "partial_overlap"


# --------------------------------------------------------------------------
# layer 2: AST diff (needs sqlglot)
# --------------------------------------------------------------------------

def _sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
        return sqlglot, exp
    except ImportError:
        return None, None


def _canonicalize_aliases(tree, exp):
    """Rewrite table aliases to positional names (a0, a1, ...) in order of
    appearance, and rewrite the column qualifiers that reference them.

    sqlglot's normalize=True handles quoting and case but NOT alias naming, so
    without this `FROM t AS T1 WHERE T1.y=1` and `FROM t AS a WHERE a.y=1`
    canonicalize to different strings and the SFT dedup keeps both as separate
    training rows. Since rejection sampling at temperature 0.8 produces exactly
    this kind of cosmetic variation, that would inflate the dataset with
    duplicates and skew the loss toward whichever questions happened to generate
    the most alias spellings.
    """
    mapping, i = {}, 0
    for t in tree.find_all(exp.Table):
        alias = t.alias
        if alias and alias.lower() not in mapping:
            mapping[alias.lower()] = f"a{i}"
            i += 1
    if not mapping:
        return tree
    for t in tree.find_all(exp.Table):
        if t.alias and t.alias.lower() in mapping:
            t.set("alias", exp.to_identifier(mapping[t.alias.lower()]))
    for c in tree.find_all(exp.Column):
        tbl = c.args.get("table")
        if tbl is not None and str(tbl.name).lower() in mapping:
            c.set("table", exp.to_identifier(mapping[str(tbl.name).lower()]))
    return tree


def normalize(sql, dialect="sqlite"):
    """Canonical form for dedup: parsed, identifiers normalized, aliases
    positionally renamed.

    Returns None if unparseable. Used by build_sft_data.py to dedupe
    rejection-sampling survivors -- two completions differing only in alias
    names or whitespace are one training example, not two.
    """
    sqlglot, exp = _sqlglot()
    if sqlglot is None:
        return None
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        tree = _canonicalize_aliases(tree, exp)
        return tree.sql(dialect=dialect, normalize=True, pretty=False,
                        comments=False).lower()
    except Exception:
        return None


def _features(sql, dialect="sqlite"):
    """Structural fingerprint: tables, join edges + types, filter columns,
    grouping + having, aggregates, projected columns, distinct, order/limit,
    literals."""
    sqlglot, exp = _sqlglot()
    if sqlglot is None:
        return None
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None

    f = {"tables": set(), "join_cols": set(), "join_types": set(),
         "where_cols": set(), "group": set(), "having": set(),
         "aggs": set(), "select": set(), "literals": set(),
         "order": (), "limit": None, "distinct": False}

    for t in tree.find_all(exp.Table):
        if t.name:
            f["tables"].add(t.name.lower())
    for j in tree.find_all(exp.Join):
        side = (j.args.get("side") or "").lower()
        kind = (j.args.get("kind") or "").lower()
        f["join_types"].add((side + " " + kind).strip() or "inner")
        on = j.args.get("on")
        if on is not None:
            for c in on.find_all(exp.Column):
                f["join_cols"].add(c.name.lower())
    where = tree.find(exp.Where)
    if where is not None:
        for c in where.find_all(exp.Column):
            f["where_cols"].add(c.name.lower())
    g = tree.find(exp.Group)
    if g is not None:
        for c in g.find_all(exp.Column):
            f["group"].add(c.name.lower())
    h = tree.find(exp.Having)
    if h is not None:
        for c in h.find_all(exp.Column):
            f["having"].add(c.name.lower())
    for a in tree.find_all(exp.AggFunc):
        f["aggs"].add(a.key.lower())
    sel = tree.find(exp.Select)
    if sel is not None:
        f["distinct"] = sel.args.get("distinct") is not None
        for e in sel.expressions:
            f["select"].add(e.sql(dialect=dialect).lower())
    for lit in tree.find_all(exp.Literal):
        f["literals"].add(str(lit.this).lower())
    o = tree.find(exp.Order)
    if o is not None:
        f["order"] = tuple(
            (od.this.sql(dialect=dialect).lower(), bool(od.args.get("desc")))
            for od in o.find_all(exp.Ordered))
    lim = tree.find(exp.Limit)
    if lim is not None and lim.expression is not None:
        f["limit"] = lim.expression.sql(dialect=dialect).lower()
    return f


def ast_kind(pred, gold, dialect="sqlite"):
    """Structural category for a miss. Each maps onto a specific intervention.

    unparseable_pred    prediction is not valid SQL
    literals_only       identical structure, different constants -> the
                        sample-values arm is the fix
    join_path           same tables, different join edges -> the FK arm
    table_set           different tables entirely -> schema-linking-hard
    grouping            GROUP BY / HAVING / aggregate disagreement -> interpretation
    order_limit         ORDER BY / LIMIT disagreement -> superlative direction
                        or tie-break (the top-k / "highest" failure class)
    projection          SELECT-clause disagreement only
    filter              WHERE columns differ
    mixed               several of the above at once
    """
    sqlglot, _ = _sqlglot()
    if sqlglot is None:
        return "sqlglot_unavailable"
    fp, fg = _features(pred, dialect), _features(gold, dialect)
    if fp is None:
        return "unparseable_pred"
    if fg is None:
        return "unparseable_gold"

    diffs = []
    if fp["tables"] != fg["tables"]:
        diffs.append("table_set")
    if fp["join_cols"] != fg["join_cols"] or fp["join_types"] != fg["join_types"]:
        diffs.append("join_path")
    if (fp["group"] != fg["group"] or fp["aggs"] != fg["aggs"]
            or fp["having"] != fg["having"]):
        diffs.append("grouping")
    if fp["order"] != fg["order"] or fp["limit"] != fg["limit"]:
        diffs.append("order_limit")
    if fp["select"] != fg["select"] or fp["distinct"] != fg["distinct"]:
        diffs.append("projection")
    if fp["where_cols"] != fg["where_cols"]:
        diffs.append("filter")

    if not diffs:
        return "literals_only" if fp["literals"] != fg["literals"] else "equivalent"
    if len(diffs) == 1:
        return diffs[0]
    # Precedence, most-upstream cause first. A difference in table choice makes
    # the join/filter differences downstream consequences rather than
    # independent errors; likewise a changed aggregate necessarily changes the
    # projection string, so `grouping` outranks `projection`. Without this
    # ordering almost everything collapses into "mixed" and the taxonomy stops
    # pointing at an intervention.
    for cause in ("table_set", "join_path", "grouping", "order_limit",
                  "filter", "projection"):
        if cause in diffs:
            return cause
    return "mixed"
