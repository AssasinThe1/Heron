"""Data-quality checks, run before every analysis on the RAW transactions table.

  duplicates   same txn_id more than once (removed in every role's view; counted here)
  missing      share of blank bank / merchant / method; > 1% warns on results split by it
  gaps         hours with no rows at all where that hour normally has volume
  freshness    latest record vs the expected end of data

Engine-internal: runs on the system connection, but always inside the role's
row filter, so a regional analyst only ever sees counts for their own region.
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta

from .db import system_connection
from .settings import data_meta

MISSING_WARN = 0.01
COLUMN_DIM = {"bank_id": "bank", "merchant_id": "merchant", "method": "method"}


@dataclass
class DQReport:
    duplicates: int = 0
    duplicate_days: list = field(default_factory=list)        # [(date, n)]
    missing: dict = field(default_factory=dict)               # col -> share in window
    missing_days: dict = field(default_factory=dict)          # col -> [(date, share)]
    gaps: list = field(default_factory=list)                  # [(start_ts, end_ts, hours)]
    latest: object = None
    stale: bool = False
    queries: list = field(default_factory=list)               # (title, sql, df, ms)

    def warned_dims(self):
        return {COLUMN_DIM[c] for c, s in self.missing.items() if s > MISSING_WARN}


def _where(windows, row_filter):
    parts = [f"(created_at >= '{a}' AND created_at < '{b + timedelta(days=1)}')" for a, b in windows]
    w = "(" + " OR ".join(parts) + ")"
    if row_filter:
        w += f" AND {row_filter}"
    return w


def run(windows, row_filter=None):
    """windows: list of (date, date) inclusive ranges the analysis reads."""
    rep = DQReport()
    con = system_connection()
    w = _where(windows, row_filter)
    try:
        def q(title, sql):
            t0 = time.perf_counter()
            df = con.execute(sql).df()
            rep.queries.append((title, sql, df, (time.perf_counter() - t0) * 1000))
            return df

        d = q("Duplicate records by day",
              f"SELECT CAST(created_at AS DATE) AS day, COUNT(*) - COUNT(DISTINCT txn_id) AS duplicates "
              f"FROM transactions WHERE {w} GROUP BY 1 HAVING COUNT(*) > COUNT(DISTINCT txn_id) ORDER BY 1")
        rep.duplicates = int(d.duplicates.sum()) if len(d) else 0
        rep.duplicate_days = [(r.day.date() if hasattr(r.day, "date") else r.day, int(r.duplicates))
                              for r in d.itertuples()]

        m = q("Blank values in key columns",
              f"SELECT COUNT(*) AS n, AVG((bank_id IS NULL)::INT) AS bank_id, "
              f"AVG((merchant_id IS NULL)::INT) AS merchant_id, AVG((method IS NULL)::INT) AS method "
              f"FROM transactions WHERE {w}")
        if len(m) and m.n.iloc[0]:
            rep.missing = {c: float(m[c].iloc[0] or 0) for c in COLUMN_DIM}
        md = q("Days with more than 1% blank bank",
               f"SELECT CAST(created_at AS DATE) AS day, AVG((bank_id IS NULL)::INT) AS share "
               f"FROM transactions WHERE {w} GROUP BY 1 HAVING AVG((bank_id IS NULL)::INT) > {MISSING_WARN} ORDER BY 1")
        if len(md):
            rep.missing_days["bank_id"] = [(r.day.date() if hasattr(r.day, "date") else r.day,
                                            float(r.share)) for r in md.itertuples()]

        lo = min(a for a, _ in windows)
        hi = max(b for _, b in windows)
        g = q("Hours with no records",
              f"""WITH hours AS (
                    SELECT unnest(generate_series(TIMESTAMP '{lo}', TIMESTAMP '{hi} 23:00:00', INTERVAL 1 HOUR)) AS h),
                  counts AS (
                    SELECT date_trunc('hour', created_at) AS h, COUNT(*) AS n FROM transactions WHERE {w} GROUP BY 1),
                  typical AS (
                    SELECT hour(h) AS hod, median(n) AS usual FROM counts GROUP BY 1)
                  SELECT hours.h, COALESCE(counts.n, 0) AS n, typical.usual
                  FROM hours LEFT JOIN counts USING (h) LEFT JOIN typical ON typical.hod = hour(hours.h)
                  WHERE COALESCE(counts.n, 0) = 0 AND typical.usual >= 20 ORDER BY 1""")
        if len(g):
            hs = list(g.h)
            start = prev = hs[0]
            for h in hs[1:] + [None]:
                if h is not None and (h - prev) == timedelta(hours=1):
                    prev = h
                    continue
                rep.gaps.append((start, prev + timedelta(hours=1),
                                 int((prev - start).total_seconds() // 3600) + 1))
                if h is not None:
                    start = prev = h

        f = q("Latest record", f"SELECT MAX(created_at) AS latest FROM transactions"
              + (f" WHERE {row_filter}" if row_filter else ""))
        rep.latest = f.latest.iloc[0]
        expected = data_meta()["as_of_date"]
        rep.stale = rep.latest is not None and rep.latest.date() < expected - timedelta(days=1)
    finally:
        con.close()
    return rep
