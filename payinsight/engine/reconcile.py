"""Two sources, one number: successful payment value (transactions) vs settlements.

The gap is explained in known steps, each computed by its own query:
  1. day cut-off      settlements group by UTC day; transactions by IST day
  2. payout timing    UTC day D pays out on D+1, so a month's settlements cover
                      the UTC days from the last day of the previous month to
                      the second-last day of this one
  3. refunds          deducted from settlements, not from transactions
  4. missing payouts  merchant-days with successful payments but no payout row
Whatever is left is reported as "unexplained", with its size. Fees are shown
for completeness but excluded from the comparison (transactions carry no fees).
"""

from dataclasses import dataclass, field
from datetime import timedelta

from .planner import fmt_range


@dataclass
class Recon:
    month: tuple
    txn_value: float
    settled_gross: float
    refunds: float
    fees: float
    net: float
    steps: list = field(default_factory=list)     # (label, amount, evidence id, note)
    unexplained: float = 0.0
    missing_rows: object = None
    ev: dict = field(default_factory=dict)
    lag_days: int = 1

    @property
    def settled_before_fees(self):
        return self.settled_gross - self.refunds

    @property
    def gap(self):
        return self.settled_before_fees - self.txn_value


def _txn_source(rdb):
    if "transactions" in rdb.tables:
        return ("transactions", "t.created_at", "SUM(t.amount_inr) FILTER (WHERE t.status = 'success')",
                "t.status = 'success'")
    if "transactions_hourly" in rdb.tables:
        return ("transactions_hourly", "t.hour", "SUM(t.amount_inr) FILTER (WHERE t.status = 'success')",
                "t.status = 'success'")
    return None


def can_reconcile(rdb):
    return "settlements" in rdb.tables and _txn_source(rdb) is not None


def run(period, runner, lag_days=1, fmt=None):
    """period: (first, last) day of the month (any range works)."""
    rdb = runner.rdb
    table, tcol, val, ok = _txn_source(rdb)
    a, b = period
    b1 = b + timedelta(days=1)
    utc = f"({tcol} - INTERVAL 330 MINUTE)"
    label = fmt_range(a, b)
    reg = fmt._reg if fmt is not None else (lambda x: x)   # notes quote config values
    r = Recon(month=period, txn_value=0, settled_gross=0, refunds=0, fees=0, net=0, lag_days=lag_days)

    df, e = runner.run(
        f"SELECT {val} AS value FROM {table} t WHERE {tcol} >= '{a}' AND {tcol} < '{b1}'",
        title=f"Successful payment value, {label} (IST days)", dataset=table,
        filters="status = success", window=f"{label}, IST")
    r.txn_value = float(df.value.iloc[0] or 0)
    r.ev["txn"] = e.id

    df, e = runner.run(
        f"SELECT SUM(gross_inr) AS gross, SUM(refunds_inr) AS refunds, SUM(fees_inr) AS fees, "
        f"SUM(net_inr) AS net, COUNT(*) AS payouts FROM settlements "
        f"WHERE settlement_date >= '{a}' AND settlement_date < '{b1}'",
        title=f"Settlements paid out, {label}", dataset="settlements",
        filters="by settlement_date", window=f"{label}, settlement dates")
    row = df.iloc[0]
    r.settled_gross, r.refunds = float(row.gross or 0), float(row.refunds or 0)
    r.fees, r.net = float(row.fees or 0), float(row.net or 0)
    r.ev["settle"] = e.id

    # 1. cut-off: same calendar month, but on UTC days instead of IST days.
    # (Needs one day of data either side; the guardrail range check still applies.)
    lo, hi = a - timedelta(days=2), b1 + timedelta(days=2)
    df, e = runner.run(
        f"SELECT {val} AS value FROM {table} t WHERE {tcol} >= '{lo}' AND {tcol} < '{hi}' "
        f"AND {utc} >= '{a}' AND {utc} < '{b1}'",
        title=f"Successful value on UTC days of {label}", dataset=table,
        filters="status = success; day boundary at 00:00 UTC (05:30 IST)", window=f"{label}, UTC")
    utc_month = float(df.value.iloc[0] or 0)
    r.ev["cutoff"] = e.id
    r.steps.append(("Day cut-off (UTC instead of IST)", utc_month - r.txn_value, e.id,
                    reg("Settlements close each day at 00:00 UTC, which is 05:30 IST.")))

    # 2. payout timing: the settled UTC days are shifted back by the payout lag.
    s0, s1 = a - timedelta(days=lag_days), b1 - timedelta(days=lag_days)
    df, e = runner.run(
        f"SELECT {val} AS value FROM {table} t WHERE {tcol} >= '{lo}' AND {tcol} < '{hi}' "
        f"AND {utc} >= '{s0}' AND {utc} < '{s1}'",
        title=f"Successful value on the UTC days paid out in {label}", dataset=table,
        filters=f"status = success; UTC days {s0} to {s1 - timedelta(days=1)}",
        window=f"UTC {s0} – {s1 - timedelta(days=1)}")
    paid_window = float(df.value.iloc[0] or 0)
    r.ev["lag"] = e.id
    r.steps.append(("Next-day payout (last day pays out next month)", paid_window - utc_month, e.id,
                    reg(f"A UTC day's sales settle {lag_days} day later, so {b.day} {b:%b} pays out on "
                        f"{b1.day} {b1:%b} and the previous month's last day pays out on {a.day} {a:%b}.")))

    # 3. refunds
    r.steps.append(("Refunds deducted from settlements", -r.refunds, r.ev["settle"],
                    "Settlements net off refunds; transactions record the original sale."))

    # 4. missing payout records: merchant-days with sales but no payout row
    if "transactions" in rdb.tables or "transactions_hourly" in rdb.tables:
        df, e = runner.run(
            f"WITH sales AS (SELECT t.merchant_id, CAST({utc} AS DATE) + INTERVAL {lag_days} DAY AS pay_date, "
            f"{val} AS gross FROM {table} t WHERE {tcol} >= '{lo}' AND {tcol} < '{hi}' AND {ok} "
            f"AND {utc} >= '{s0}' AND {utc} < '{s1}' GROUP BY 1, 2) "
            f"SELECT s.merchant_id, CAST(s.pay_date AS DATE) AS expected_payout_date, ROUND(s.gross, 2) AS gross "
            f"FROM sales s LEFT JOIN settlements p ON p.merchant_id = s.merchant_id "
            f"AND p.settlement_date = CAST(s.pay_date AS DATE) WHERE p.settlement_id IS NULL ORDER BY 3 DESC",
            title=f"Merchant-days with sales but no payout record, {label}",
            dataset=f"{table} + settlements", filters="status = success; anti-join on merchant and day",
            window=f"payouts due {label}")
        r.missing_rows = df
        r.ev["missing"] = e.id
        missing = -float(df.gross.sum()) if len(df) else 0.0
        if len(df):
            r.steps.append((f"Payout records missing ({len(df)} merchant-days)", missing, e.id,
                            "Sales that should have settled in the month have no payout row."))

    explained = sum(s[1] for s in r.steps)
    r.unexplained = r.gap - explained
    return r
