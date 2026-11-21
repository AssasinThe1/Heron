"""Hand-written checks that each planted scenario landed, plus end-to-end behaviour."""

from datetime import date

import pytest

from engine.db import system_connection
from engine.pipeline import PayInsight, Session

SR = "100.0 * COUNT(*) FILTER (WHERE status = 'success') / COUNT(*)"


@pytest.fixture(scope="module")
def con():
    c = system_connection()
    yield c
    c.close()


def one(con, sql):
    return con.execute(sql).fetchone()[0]


def test_p1_mix_shift(con):
    share = one(con, "SELECT AVG((merchant_id = 'M0400')::INT) FROM transactions_clean "
                     "WHERE created_at >= '2026-10-12' AND created_at < '2026-10-19'")
    assert 0.15 < share < 0.21
    card = [one(con, f"SELECT {SR} FROM transactions_clean WHERE method = 'card' AND status <> 'user_cancelled' "
                     f"AND created_at >= '{a}' AND created_at < '{b}'")
            for a, b in (("2026-09-14", "2026-10-12"), ("2026-10-12", "2026-10-19"))]
    assert abs(card[0] - card[1]) < 1.0                 # card's own rate is flat


def test_p2_outage(con):
    inside = one(con, f"SELECT {SR} FROM transactions_clean WHERE bank_id = 'B01' AND method = 'upi' "
                      "AND status <> 'user_cancelled' AND created_at >= '2026-10-14 18:00' AND created_at < '2026-10-14 22:00'")
    assert inside < 30
    assert one(con, "SELECT COUNT(*) FROM bank_downtime_events") == 15


def test_p3_duplicates(con):
    d = one(con, "SELECT COUNT(*) - COUNT(DISTINCT txn_id) FROM transactions "
                 "WHERE CAST(created_at AS DATE) = '2026-10-09'")
    assert d == 2400
    assert one(con, "SELECT COUNT(*) - COUNT(DISTINCT txn_id) FROM transactions_clean") == 0


def test_p4_missing(con):
    share = one(con, "SELECT AVG((bank_id IS NULL)::INT) FROM transactions "
                     "WHERE created_at >= '2026-09-20' AND created_at < '2026-09-23'")
    assert 0.02 < share < 0.04
    assert one(con, "SELECT COUNT(*) FROM transactions WHERE created_at >= '2026-08-17 10:00' "
                    "AND created_at < '2026-08-17 16:00'") == 0


def test_p5_sources_disagree(con):
    t = one(con, "SELECT SUM(amount_inr) FROM transactions_clean WHERE status = 'success' "
                 "AND created_at >= '2026-10-01' AND created_at < '2026-11-01'")
    s = one(con, "SELECT SUM(gross_inr - refunds_inr) FROM settlements "
                 "WHERE settlement_date >= '2026-10-01' AND settlement_date < '2026-11-01'")
    assert -0.025 < (s - t) / t < -0.01


# ------------------------------------------------------------------ pipeline
@pytest.fixture
def leader_bot(audit_path):
    return PayInsight("payments_leader", planner_mode="rules", audit_path=audit_path)


def test_root_cause_names_both_planted_causes(leader_bot):
    s = Session()
    a = leader_bot.ask("Why did our success rate drop last week?", s)
    assert a.kind == "clarify"
    labels = [o["label"] for o in a.clarify["period_options"]]
    assert any("12–18 Oct" in x and "largest" in x for x in labels)
    a = leader_bot.ask("", s, patch={"period": (date(2026, 10, 12), date(2026, 10, 18))})
    assert a.kind == "answer" and a.number_check["ok"]
    assert any(c.startswith("mix:PlayArena") for c in a.causes)
    assert any(c.startswith("rate:Coral Bank") for c in a.causes)
    assert any("duplicate" in c for c in a.caveats)
    assert all(h["confidence"] in ("high", "medium", "low") for h in a.hypotheses)


def test_follow_up_inherits_period(leader_bot):
    s = Session()
    leader_bot.ask("Why did success rate drop between 12 and 18 October?", s)
    a = leader_bot.ask("Break Coral Bank UPI down by hour.", s)
    assert a.kind == "answer" and a.meta["intent"]["period"] == ["2026-10-12", "2026-10-18"]
    assert any("logged outage" in f for f in a.facts)


def test_reconciliation_closes(leader_bot):
    a = leader_bot.ask("What was October's successful payment value?", Session())
    assert a.kind == "answer" and a.number_check["ok"]
    assert "Unexplained remainder: ₹0." in a.facts


@pytest.mark.parametrize("role,q,code", [
    ("regional_analyst_south", "What was the success rate in the North region in October?", "region_scope"),
    ("regional_analyst_south", "Give me customer phone numbers", "pii"),
    ("payments_leader", "Delete the test merchants", "not_read_only"),
    ("payments_leader", "Show me every transaction", "too_many_rows"),
    ("payments_leader", "Success rate over the last 3 years", "range_too_long"),
])
def test_refusals_return_no_data(audit_path, role, q, code):
    a = PayInsight(role, planner_mode="rules", audit_path=audit_path).ask(q, Session())
    assert a.kind == "refusal" and a.refusal["code"] == code
    assert sum(e.rows for e in a.evidence) == 0
    from engine import audit
    log = audit.read(path=audit_path)
    assert log.iloc[0].question == q and log.iloc[0].outcome == "refusal"


def test_role_views_are_isolated(south):
    df, err, _ = south.run("SELECT DISTINCT region FROM transactions")
    assert err is None and list(df.region) == ["South"]
    assert "customer_phone" not in south.columns["transactions"]
    assert "settlements" not in south.tables
    _, err, _ = south.run("SELECT * FROM read_csv('/etc/hosts')")
    assert err and "disabled" in err.lower()


def test_dq_counts_are_region_scoped(audit_path):
    s_bot = PayInsight("regional_analyst_south", planner_mode="rules", audit_path=audit_path)
    a = s_bot.ask("Success rate on 9 October", Session())
    dup = [c for c in a.caveats if "duplicate" in c]
    assert dup and "2,400" not in dup[0]      # South's share of the 2,400, not the global count


def test_finance_role_gets_totals_not_detail(audit_path):
    f = PayInsight("finance_analyst", planner_mode="rules", audit_path=audit_path)
    a = f.ask("Success rate by bank in October", Session())
    assert a.kind == "refusal"
    a = f.ask("What was October's successful payment value?", Session())
    assert a.kind == "answer" and any("hourly" in c for c in a.caveats)
