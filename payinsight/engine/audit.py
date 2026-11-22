"""Audit log: every question, role, generated SQL, check result, run time, answer.

Kept in its own DuckDB file so the data file can stay read-only.
"""

import json
import threading
from datetime import datetime

import duckdb

from .settings import AUDIT_DB

_lock = threading.Lock()
_SCHEMA = """CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT, ts TIMESTAMP, role VARCHAR, question VARCHAR, kind VARCHAR,
    outcome VARCHAR, check_result VARCHAR, sql_source VARCHAR, sql VARCHAR,
    run_ms DOUBLE, answer VARCHAR, intent VARCHAR, n_queries INTEGER, cache_hits INTEGER)"""


def _con(path=None):
    con = duckdb.connect(path or AUDIT_DB)
    con.execute(_SCHEMA)
    return con


def write(role, question, kind, outcome, check_result, sql_source, sql, run_ms, answer,
          intent=None, n_queries=0, cache_hits=0, path=None):
    with _lock:
        con = _con(path)
        try:
            nid = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM audit_log").fetchone()[0]
            con.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                nid, datetime.now(), role, question, kind, outcome, check_result, sql_source,
                (sql or "")[:20000], float(run_ms or 0), (answer or "")[:2000],
                json.dumps(intent, default=str) if intent else None, n_queries, cache_hits])
            return nid
        finally:
            con.close()


def read(limit=200, path=None):
    with _lock:
        con = _con(path)
        try:
            return con.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", [limit]).df()
        finally:
            con.close()
