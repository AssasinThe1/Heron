"""DuckDB connections: one per role, holding ONLY what that role may see.

Layer 1 of the two-layer protection. Each role gets a fresh in-memory database
with filtered copies of its allowed tables (columns dropped, rows filtered, PII
masked). The source file is then detached and external access switched off, so
even SQL that slipped past guardrails.py has nothing else to read.
"""

import threading
import time

import duckdb
import pandas as pd

from .settings import DATA_DB, policy

# Every table a user could ever name, with its source in the data file.
SOURCES = {
    "transactions": "transactions_clean",       # deduplicated (raw keeps dups for DQ)
    "merchants": "merchants",
    "banks": "banks",
    "payment_methods": "payment_methods",
    "settlements": "settlements",
    "bank_downtime_events": "bank_downtime_events",
}
ALL_TABLES = set(SOURCES) | {"transactions_hourly"}

HOURLY_SQL = """
SELECT date_trunc('hour', created_at) AS hour, merchant_id, method, region, status,
       COUNT(DISTINCT txn_id) AS attempts, SUM(amount_inr) AS amount_inr
FROM src.transactions_clean GROUP BY ALL
"""

MASK = {
    "customer_phone": "'+91-9XXXXX' || right(customer_phone, 4)",
    "customer_vpa": "CASE WHEN customer_vpa IS NULL THEN NULL ELSE "
                    "left(customer_vpa, 1) || '***@' || split_part(customer_vpa, '@', 2) END",
}

# Short descriptions for the Heron DDL (comments help a BIRD-trained model).
COLUMN_TYPES_SQLITE = {
    "VARCHAR": "TEXT", "TIMESTAMP": "DATETIME", "DATE": "DATE", "DOUBLE": "REAL",
    "BIGINT": "INTEGER", "INTEGER": "INTEGER", "BOOLEAN": "INTEGER", "HUGEINT": "INTEGER",
    "TIMESTAMP_NS": "DATETIME", "TIMESTAMP WITH TIME ZONE": "DATETIME",
}
PRIMARY_KEYS = {"transactions": "txn_id", "merchants": "merchant_id", "banks": "bank_id",
                "payment_methods": "method", "settlements": "settlement_id"}
FOREIGN_KEYS = {
    "transactions": {"merchant_id": "merchants(merchant_id)", "bank_id": "banks(bank_id)",
                     "method": "payment_methods(method)"},
    "settlements": {"merchant_id": "merchants(merchant_id)"},
    "bank_downtime_events": {"bank_id": "banks(bank_id)", "method": "payment_methods(method)"},
    "transactions_hourly": {"merchant_id": "merchants(merchant_id)"},
}


def system_connection():
    """Read-only connection to the full file. Engine-internal ONLY (data-quality
    checks and the evaluation's gold queries); never handed user SQL."""
    return duckdb.connect(DATA_DB, read_only=True)


class RoleDB:
    """An isolated database for one role."""

    def __init__(self, role):
        pol = policy()
        if role not in pol["roles"]:
            raise KeyError(f"unknown role {role!r}")
        self.role = role
        self.spec = pol["roles"][role]
        self.pii_columns = set(pol["pii_columns"])
        self.con = duckdb.connect(":memory:")
        self._lock = threading.Lock()
        self.columns = {}       # table -> [cols]
        self.types = {}         # table -> {col: duckdb type}
        self.row_counts = {}
        self._build()

    def _build(self):
        con = self.con
        con.execute(f"ATTACH '{DATA_DB}' AS src (READ_ONLY)")
        for table, tspec in self.spec["tables"].items():
            if table == "transactions_hourly":
                select = HOURLY_SQL
            else:
                src = SOURCES[table]
                cols = [r[0] for r in con.execute(
                    f"SELECT column_name FROM duckdb_columns() WHERE database_name='src' "
                    f"AND table_name='{src}' ORDER BY column_index").fetchall()]
                exprs = []
                for c in cols:
                    if c in self.pii_columns:
                        if tspec.get("columns") == "all_but_pii" or self.spec.get("pii") == "none":
                            continue
                        if self.spec.get("pii") == "masked":
                            exprs.append(f"{MASK[c]} AS {c}")
                            continue
                    exprs.append(c)
                select = f"SELECT {', '.join(exprs)} FROM src.{src}"
            if tspec.get("row_filter"):
                select = f"SELECT * FROM ({select}) WHERE {tspec['row_filter']}"
            con.execute(f"CREATE TABLE {table} AS {select}")
        con.execute("DETACH src")
        # Nothing outside this in-memory database can be read from here on.
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        for table in self.spec["tables"]:
            info = con.execute(f"DESCRIBE {table}").fetchall()
            self.columns[table] = [r[0] for r in info]
            self.types[table] = {r[0]: r[1] for r in info}
            self.row_counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ------------------------------------------------------------------ schema
    @property
    def tables(self):
        return list(self.columns)

    def schema_ddl(self):
        """SQLite-style CREATE TABLE statements, the format Heron was trained on."""
        out = []
        for t, cols in self.columns.items():
            lines = []
            for c in cols:
                ty = COLUMN_TYPES_SQLITE.get(self.types[t][c].split("(")[0], "TEXT")
                pk = " primary key" if PRIMARY_KEYS.get(t) == c else ""
                fk = FOREIGN_KEYS.get(t, {}).get(c)
                ref = f"\n        references {fk}" if fk else ""
                lines.append(f"    {c} {ty}{pk}{ref}")
            out.append(f"CREATE TABLE {t}\n(\n" + ",\n".join(lines) + "\n)")
        return "\n\n".join(out)

    # --------------------------------------------------------------- execution
    def run(self, sql, timeout=10.0, params=None):
        """Execute -> (DataFrame | None, error | None, elapsed_ms).

        Same shape as Heron's utils.run_query: a timer thread interrupts the
        query at `timeout`, so a runaway join can't stall the app.
        """
        cur = self.con.cursor()
        timer = threading.Timer(timeout, cur.interrupt)
        t0 = time.perf_counter()
        timer.start()
        try:
            df = cur.execute(sql, params or []).df()
            return df, None, (time.perf_counter() - t0) * 1000
        except duckdb.InterruptException:
            return None, f"timeout after {timeout:.0f}s", (time.perf_counter() - t0) * 1000
        except Exception as e:                                  # noqa: BLE001
            msg = str(e)
            if "INTERRUPT" in msg.upper():
                msg = f"timeout after {timeout:.0f}s"
            return None, msg, (time.perf_counter() - t0) * 1000
        finally:
            timer.cancel()
            cur.close()

    def explain(self, sql):
        """(rows_scanned_estimate, rows_out_estimate) from DuckDB's own planner."""
        import json
        cur = self.con.cursor()
        try:
            plan = json.loads(cur.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchall()[0][1])
        finally:
            cur.close()
        scanned = 0

        def walk(node):
            nonlocal scanned
            info = node.get("extra_info", {}) or {}
            if node.get("name") in ("SEQ_SCAN", "TABLE_SCAN", "SEQ_SCAN "):
                tname = str(info.get("Table", "")).split(".")[-1]
                scanned += self.row_counts.get(tname, _int(info.get("Estimated Cardinality")))
            for ch in node.get("children", []):
                walk(ch)

        roots = plan if isinstance(plan, list) else [plan]
        for r in roots:
            walk(r)
        out = _int((roots[0].get("extra_info") or {}).get("Estimated Cardinality")) if roots else 0
        return scanned, out


def _int(x):
    try:
        return int(str(x).replace("~", "").split()[0])
    except (TypeError, ValueError, IndexError):
        return 0


_role_cache = {}
_cache_lock = threading.Lock()


def role_db(role):
    """Process-wide cache: building a role database copies ~600k rows (~1 s)."""
    with _cache_lock:
        if role not in _role_cache:
            _role_cache[role] = RoleDB(role)
        return _role_cache[role]


def lookup_values():
    """Names the planner matches against (banks, merchants, categories)."""
    con = system_connection()
    try:
        banks = [r[0] for r in con.execute("SELECT name FROM banks ORDER BY name").fetchall()]
        merchants = con.execute("SELECT merchant_id, name, category, region FROM merchants").df()
        return {"banks": banks, "merchants": merchants,
                "categories": sorted(merchants.category.unique().tolist())}
    finally:
        con.close()


def as_frame(rows, cols):
    return pd.DataFrame(rows, columns=cols)
