"""Shared helpers for the BIRD text-to-SQL pipeline.

Local-side module. The Colab side never imports this -- it only ever reads a
prompts JSON, so no database access and no dependency on this file.

Two things live here that are load-bearing for the whole project:

  * load_split() understands "virtual splits" (train-fit / train-val), which are
    index subsets of train.json stored under splits/. This is how the
    disjoint-database validation discipline is enforced mechanically rather than
    by remembering to filter.

  * schema_str() takes a `variant` argument. The variant string is the frozen
    format decision -- once SFT data is generated with a variant, that variant
    is baked into the model and must be used at serving time. build_prompts.py
    writes the variant into the payload so score.py can assert they match.
"""

import csv
import glob
import json
import os
import re
import sqlite3
import threading

import pandas as pd

DEFAULT_DATA_ROOT = "../Data"
SPLIT_DIR = "splits"
PROFILE_RULES_PATH = "profile_rules.json"

# Every variant is a deterministic function of the database file, so a frozen
# variant string fully determines the prompt. Do not add nondeterminism here
# (no sampling of rows, no dict iteration order dependence).
VARIANTS = ["ddl", "fk", "samples", "fk+samples", "desc", "all"]

SYSTEM = (
    "You are an expert text-to-SQL model for SQLite. Given a schema, "
    "external knowledge, and a question, respond with ONLY the SQL query "
    "- no markdown, no explanation.\n"
    "Rules:\n"
    "- Use ONLY SQLite functions. There is no YEAR(), MONTH(), or DATEDIFF(); "
    "use STRFTIME('%Y', col) and date() instead.\n"
    "- Copy table and column names EXACTLY as they appear in the schema, "
    "character for character. Some are misspelled or contain spaces or "
    "punctuation - reproduce them verbatim and wrap them in backticks. "
    "Never correct a name to what you think it should be.\n"
    "- Before using a column, confirm it belongs to the table you aliased it to."
)


# ==========================================================================
# splits
# ==========================================================================

def _parent_of(split):
    """train-fit and train-val are subsets of train; dev is its own thing."""
    return "train" if split.startswith("train") else split


def load_split(split, data_root=DEFAULT_DATA_ROOT, split_dir=SPLIT_DIR):
    """Returns (examples, db_map).

    `split` is either a physical split ('train', 'dev') or a virtual one
    ('train-fit', 'train-val', 'train-val-1k') defined by a JSON file in
    split_dir of the form {"parent": "train", "ids": [12, 47, ...]}.

    Virtual splits carry an extra key on each example: `orig_id`, the index into
    the parent split's JSON. Everything downstream keys on that, so an example
    has the same id whether you reached it via train or train-fit. Without this
    you cannot cross-reference a train-fit training row against a train-val
    eval row, and dedup across runs silently breaks.
    """
    parent = _parent_of(split)

    json_path = os.path.join(data_root, parent, f"{parent}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found. Check --data-root (currently {data_root!r})."
        )
    all_examples = json.load(open(json_path))
    for i, ex in enumerate(all_examples):
        ex["orig_id"] = i

    if split == parent:
        examples = all_examples
    else:
        spec_path = os.path.join(split_dir, f"{split.replace('-', '_')}.json")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(
                f"{spec_path} not found. Run make_splits.py first."
            )
        spec = json.load(open(spec_path))
        if spec["parent"] != parent:
            raise ValueError(f"{spec_path} has parent {spec['parent']!r}, expected {parent!r}")
        examples = [all_examples[i] for i in spec["ids"]]

    db_root = os.path.join(data_root, parent, f"{parent}_databases")
    db_map = {}
    for p in sorted(glob.glob(os.path.join(db_root, "*"))):
        if not os.path.isdir(p):
            continue
        hits = glob.glob(os.path.join(p, "*.sqlite"))
        if hits:
            db_map[os.path.basename(p)] = os.path.abspath(hits[0])
    if not db_map:
        raise FileNotFoundError(f"No .sqlite databases found under {db_root}")

    return examples, db_map


def load_tables_meta(split, data_root=DEFAULT_DATA_ROOT):
    """Parses {parent}_tables.json into {db_id: metadata dict}."""
    parent = _parent_of(split)
    path = os.path.join(data_root, parent, f"{parent}_tables.json")
    if not os.path.exists(path):
        return {}
    return {d["db_id"]: d for d in json.load(open(path))}


# ==========================================================================
# schema rendering
# ==========================================================================

# Keyed on (sqlite path, variant). Paths are stable and unique; id(db_map) is
# not -- CPython recycles ids after GC, so a freed train db_map and a fresh dev
# one can collide and hand back the wrong schema.
_schema_cache = {}

# Columns we never bother sampling values for: the value adds no information the
# type declaration doesn't already give, and on wide tables this is most of the
# token cost. Deterministic rule, so it's freeze-safe.
_ID_RE = re.compile(r"(^id$|_id$|id$)", re.IGNORECASE)
_DATE_RE = re.compile(r"(date|time|year|month|day|dob|timestamp)", re.IGNORECASE)


# ==========================================================================
# value profiler  (used by the samples/fk+samples/all variants and by
# derive_profile_rules.py -- they MUST share this code so the histogram the
# deriver shows predicts exactly what the profiler will list)
# ==========================================================================

_profile_rules = None


def _load_profile_rules(path=PROFILE_RULES_PATH):
    """Frozen profiler constants, derived on TRAIN by derive_profile_rules.py.

    The leakage boundary of this project runs right here. These constants -- K
    (the enum cutoff), the per-prompt value budget, the scan cap, truncation --
    are the *rules*: global, frozen, train-derived. The *values* they select are
    computed per-database at build time by _samples_block, including on dev, and
    that is fine: it is mechanical and label-free, the same status as the schema
    itself. Tuning K for one database would be contamination; computing which
    values a database happens to hold is not.
    """
    global _profile_rules
    if _profile_rules is not None:
        return _profile_rules
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The samples/fk+samples/all variants need frozen "
            "profiler rules; derive them on train first, e.g.:\n"
            "    python derive_profile_rules.py --k 12 --budget 60 "
            f"--emit {path}")
    _profile_rules = json.load(open(path))
    return _profile_rules


def _col_kind(name, ctype):
    """date | numeric | text, from declared type then name. Deterministic."""
    t = (ctype or "").upper()
    if any(k in t for k in ("DATE", "TIME")) or _DATE_RE.search(name or ""):
        return "date"
    if any(k in t for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "numeric"
    return "text"


def _bounded_distinct(con, table, col, scan_cap, cap):
    """Up to `cap` distinct non-null values of `col`, sorted, read from at most
    `scan_cap` rows.

    Two properties this query buys, both load-bearing:
      * bounded -- the inner LIMIT caps the scan, so a low-cardinality column in
        a multi-million-row fact table cannot trigger a full-table scan at build
        time (profiler cost #3).
      * deterministic -- ORDER BY makes the returned values a pure function of
        the file, so the variant stays freeze-safe (the naive block had no ORDER
        BY and its 3 values were whatever rowid order happened to surface).
    Returns None on error (e.g. an unorderable BLOB), which the caller skips.
    """
    q = (f'SELECT DISTINCT s.v FROM (SELECT "{col}" AS v FROM "{table}" '
         f'WHERE "{col}" IS NOT NULL LIMIT {scan_cap}) s '
         f'ORDER BY s.v LIMIT {cap}')
    try:
        return [r[0] for r in con.execute(q)]
    except sqlite3.Error:
        return None


def _ddl(con):
    ddl = pd.read_sql(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND type='table'", con
    )["sql"]
    return "\n\n".join(ddl)


def _fk_block(db_id, tables_meta):
    """Resolve foreign_keys index pairs from *_tables.json into readable edges.

    BIRD stores FKs as pairs of indices into column_names_original, which is
    itself a list of [table_index, column_name]. Index 0 is the synthetic '*'
    column, so guard against it.
    """
    meta = tables_meta.get(db_id)
    if not meta or not meta.get("foreign_keys"):
        return ""
    tnames = meta["table_names_original"]
    cnames = meta["column_names_original"]
    edges = []
    for a, b in meta["foreign_keys"]:
        try:
            ta, ca = cnames[a]
            tb, cb = cnames[b]
        except (IndexError, TypeError):
            continue
        if ta < 0 or tb < 0:
            continue
        edges.append(f"{tnames[ta]}.{ca} -> {tnames[tb]}.{cb}")
    if not edges:
        return ""
    return "Foreign keys:\n" + "\n".join(sorted(set(edges)))


def _samples_block(con):
    """Cardinality-gated example values, under a frozen per-prompt budget.

    For each non-id column: if it has <= K distinct values -- an enum / small
    categorical -- list them all, because that is the disambiguating signal a
    model can't guess from the type. If it has more, list nothing: sample values
    of a high-cardinality column (ids, prices, free text) carry no information
    the type doesn't already give, and they are the token-dense part that blew
    works_cycles' fk+samples prompt from an ~8k DDL floor to 17k tokens. Date-ish
    columns get one example, for format shape only.

    K, the value budget, the scan cap and the truncation are frozen rules from
    derive_profile_rules.py (train-derived; see _load_profile_rules). Columns are
    visited in a deterministic order -- tables by name, columns by definition
    order -- and values stop being added once the budget is spent, so a wide
    schema degrades gracefully instead of exploding. A column that crosses the
    budget is allowed to finish rather than be listed half; the budget is a soft
    cap that may overshoot by at most K.
    """
    r = _load_profile_rules()
    K, budget = r["k"], r["value_budget"]
    scan_cap, str_trunc = r["scan_cap"], r["str_trunc"]

    tables = [x[0] for x in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    out, spent = [], 0
    for t in tables:
        if spent >= budget:
            break
        try:
            cols = list(con.execute(f'PRAGMA table_info("{t}")'))
        except sqlite3.Error:
            continue
        lines = []
        for c in cols:
            if spent >= budget:
                break
            cname, ctype = c[1], c[2]
            if _ID_RE.search(cname):
                continue
            if _col_kind(cname, ctype) == "date":
                vals = _bounded_distinct(con, t, cname, scan_cap, 1)
            else:
                vals = _bounded_distinct(con, t, cname, scan_cap, K + 1)
                if vals is not None and len(vals) > K:
                    vals = None                # high-cardinality -> skip
            if not vals:
                continue
            shown = []
            for v in vals:
                s = str(v)
                if len(s) > str_trunc:
                    s = s[:str_trunc] + "..."
                shown.append(s)
            lines.append(f'  "{cname}": {", ".join(shown)}')
            spent += len(shown)
        if lines:
            out.append(f"{t}:\n" + "\n".join(lines))
    if not out:
        return ""
    return "Example values:\n" + "\n".join(out)


def _desc_block(db_path, max_len=90):
    """BIRD's database_description/*.csv column glosses.

    These files are frequently NOT utf-8 (cp1252 is common) and will raise if
    you assume otherwise -- hence the encoding ladder. Silently skipping a
    db whose CSVs won't decode is better than crashing a 9k-question build.
    """
    ddir = os.path.join(os.path.dirname(db_path), "database_description")
    if not os.path.isdir(ddir):
        return ""
    out = []
    for path in sorted(glob.glob(os.path.join(ddir, "*.csv"))):
        table = os.path.splitext(os.path.basename(path))[0]
        rows = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(path, newline="", encoding=enc) as f:
                    rows = list(csv.DictReader(f))
                break
            except (UnicodeDecodeError, csv.Error):
                continue
        if not rows:
            continue
        lines = []
        for r in rows:
            r = {(k or "").strip().lower(): (v or "").strip()
                 for k, v in r.items()}
            col = r.get("original_column_name") or r.get("column_name")
            desc = (r.get("column_description") or "").replace("\n", " ").strip()
            vdesc = (r.get("value_description") or "").replace("\n", " ").strip()
            text = "; ".join(x for x in (desc, vdesc) if x)
            if not col or not text:
                continue
            if len(text) > max_len:
                text = text[:max_len] + "..."
            lines.append(f'  "{col}": {text}')
        if lines:
            out.append(f"{table}:\n" + "\n".join(lines))
    if not out:
        return ""
    return "Column descriptions:\n" + "\n".join(out)


def schema_str(db_id, db_map, variant="ddl", tables_meta=None):
    """Render a database's schema under the given variant.

    variant:
      ddl          bare CREATE TABLE statements (the current baseline)
      fk           + resolved foreign-key edge list
      samples      + up to 3 distinct values per non-id column
      fk+samples   both (the arm most likely to win on accuracy-per-token)
      desc         + BIRD's column descriptions
      all          everything
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    path = db_map[db_id]
    key = (path, variant)
    if key in _schema_cache:
        return _schema_cache[key]

    con = sqlite3.connect(path)
    try:
        parts = [_ddl(con)]
        if variant in ("fk", "fk+samples", "all"):
            b = _fk_block(db_id, tables_meta or {})
            if b:
                parts.append(b)
        if variant in ("samples", "fk+samples", "all"):
            b = _samples_block(con)
            if b:
                parts.append(b)
        if variant in ("desc", "all"):
            b = _desc_block(path)
            if b:
                parts.append(b)
    finally:
        con.close()

    _schema_cache[key] = "\n\n".join(parts)
    return _schema_cache[key]


def schema_columns(db_id, db_map):
    """{table_name: [column_names]} -- used by the identifier repair guard."""
    con = sqlite3.connect(db_map[db_id])
    try:
        out = {}
        for (t,) in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            out[t] = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        return out
    finally:
        con.close()


def user_block(ex, db_map, variant="ddl", tables_meta=None, with_schema=True):
    parts = []
    if with_schema:
        parts.append(
            f"Schema:\n{schema_str(ex['db_id'], db_map, variant, tables_meta)}\n")
    parts.append(f"External knowledge: {ex.get('evidence', '')}")
    parts.append(f"Question: {ex['question']}")
    return "\n".join(parts)


# ==========================================================================
# execution
# ==========================================================================

def clean_sql(sql):
    sql = re.sub(r"```sql|```", "", sql or "").strip()
    return sql.split(";")[0].strip()


def run_query(db_path, sql, timeout=30.0):
    """Execute sql -> (result_set, error_string). Exactly one is None.

    The timeout fires con.interrupt() from a timer thread. sqlite3 permits
    interrupt() from another thread specifically for this; without it a
    predicted cartesian join stalls the whole scoring loop.
    """
    if not sql:
        return None, "empty query"
    con = sqlite3.connect(db_path)
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        return set(map(tuple, con.execute(sql).fetchall())), None
    except Exception as e:
        return None, str(e)
    finally:
        timer.cancel()
        con.close()


def add_common_args(parser, out_dir=None):
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="directory containing train/ and dev/")
    parser.add_argument("--split-dir", default=SPLIT_DIR,
                        help="where virtual split specs live")
    if out_dir is not None:
        parser.add_argument("--out-dir", default=out_dir)
    return parser
