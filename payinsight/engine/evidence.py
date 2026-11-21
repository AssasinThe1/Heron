"""Evidence, cache and the checked query runner.

Every number in an answer comes from a query run through `Runner.run`, which
  1. checks the SQL with guardrails.check (templates included: no exemptions)
  2. serves it from the cache if the same SQL already ran for this role and data version
  3. runs it under the role's timeout
  4. records an Evidence item (dataset, filters, window, SQL, rows scanned, run time)
The answer cites evidence by id: [E1], [E2], ...
"""

import hashlib
import threading
from dataclasses import dataclass, field

import sqlglot

from . import guardrails
from .settings import data_meta, policy


@dataclass
class Evidence:
    id: str
    title: str
    dataset: str
    filters: str
    window: str
    sql: str
    source: str                   # heron | hosted | semantic layer | template | data-quality
    rows_scanned: int = 0
    run_ms: float = 0.0
    rows: int = 0
    cached: bool = False
    check: str = "passed"
    note: str = ""
    preview: object = None        # small DataFrame head for the panel
    chart: object = None          # plotly figure (set by the caller)
    alt_sql: list = field(default_factory=list)   # other candidate queries (e.g. Heron's)

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ("preview", "chart")}
        return d


class QueryError(RuntimeError):
    def __init__(self, check=None, error=None):
        self.check = check
        self.error = error
        super().__init__(error or (check.reason if check else "query failed"))


class Cache:
    """Results keyed by (normalised SQL, role, data version). New data -> new keys."""

    def __init__(self, max_items=512):
        self._d = {}
        self._lock = threading.Lock()
        self.max_items = max_items
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(sql, role):
        try:
            norm = sqlglot.transpile(sql, read="duckdb", write="duckdb", pretty=False)[0]
        except Exception:                                          # noqa: BLE001
            norm = " ".join(sql.split())
        raw = f"{norm.lower()}|{role}|{data_meta()['data_version']}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, k):
        with self._lock:
            v = self._d.get(k)
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
            return v

    def put(self, k, v):
        with self._lock:
            if len(self._d) >= self.max_items:
                self._d.pop(next(iter(self._d)))
            self._d[k] = v

    def clear(self):
        with self._lock:
            self._d.clear()


CACHE = Cache()


class EvidenceLog:
    def __init__(self):
        self.items = []

    def add(self, **kw):
        ev = Evidence(id=f"E{len(self.items) + 1}", **kw)
        self.items.append(ev)
        return ev

    def get(self, eid):
        return next((e for e in self.items if e.id == eid), None)


class Runner:
    """Checked, cached, timed execution for one role, recording evidence."""

    def __init__(self, rdb, log, use_cache=True):
        self.rdb = rdb
        self.log = log
        self.use_cache = use_cache
        self.timeout = policy()["limits"]["timeout_seconds"]

    def run(self, sql, title, dataset, filters="", window="", source="template",
            note="", check=None):
        chk = check or guardrails.check(sql, self.rdb)
        if not chk.ok:
            raise QueryError(check=chk)
        key = Cache.key(chk.sql, self.rdb.role)
        hit = CACHE.get(key) if self.use_cache else None
        if hit is not None:
            df, ms = hit
            cached = True
        else:
            df, err, ms = self.rdb.run(chk.sql, timeout=self.timeout)
            if err:
                raise QueryError(error=err)
            cached = False
            if self.use_cache:
                CACHE.put(key, (df, ms))
        ev = self.log.add(title=title, dataset=dataset, filters=filters, window=window,
                          sql=chk.sql, source=source, rows_scanned=chk.scanned, run_ms=ms,
                          rows=len(df), cached=cached, note=note, preview=df.head(50))
        return df.copy(), ev

    def record(self, df, sql, title, dataset, filters="", window="", source="data-quality",
               ms=0.0, scanned=0, note=""):
        """Record evidence for engine-internal queries (data-quality checks)."""
        return self.log.add(title=title, dataset=dataset, filters=filters, window=window,
                            sql=sql, source=source, rows_scanned=scanned, run_ms=ms,
                            rows=len(df), note=note, preview=df.head(50))
