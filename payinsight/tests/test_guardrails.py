import pytest

from engine.guardrails import check

OK_RANGE = "created_at >= '2026-09-01' AND created_at < '2026-10-01'"


@pytest.mark.parametrize("sql,code", [
    ("DELETE FROM merchants WHERE name LIKE 'test%'", "not_read_only"),
    ("DROP TABLE transactions", "not_read_only"),
    ("UPDATE merchants SET tier = 'x'", "not_read_only"),
    ("INSERT INTO banks VALUES ('x', 'y', 'z')", "not_read_only"),
    ("COPY transactions TO 'out.csv'", "not_read_only"),
    ("SET threads = 1", "not_read_only"),
    ("SELECT 1; SELECT 2", "multi_statement"),
    ("SELECT * FROM read_csv('/etc/passwd')", "blocked_function"),
    ("SELECT getenv('HOME')", "blocked_function"),
    ("SELECT * FROM src.transactions", "unknown_table"),
    ("SELECT * FROM nonexistent", "unknown_table"),
    ("SELECT COUNT(*) FROM transactions", "no_date_filter"),
    ("SELECT COUNT(*) FROM transactions WHERE created_at >= '2023-01-01'", "range_too_long"),
    ("SELECT COUNT(*) FROM transactions WHERE created_at BETWEEN '2025-01-01' AND '2026-10-31'",
     "range_too_long"),
    (f"SELECT customer_phone FROM transactions WHERE {OK_RANGE}", "pii"),
    (f"SELECT * FROM transactions WHERE {OK_RANGE}", "too_many_rows"),
    (f"SELECT COUNT(*) FROM transactions t CROSS JOIN banks b WHERE t.{OK_RANGE}", "cross_join"),
    (f"SELECT COUNT(*) FROM transactions t, banks b WHERE t.{OK_RANGE}", "cross_join"),
])
def test_blocked(leader, sql, code):
    c = check(sql, leader)
    assert not c.ok and c.code == code, (c.code, c.reason)


@pytest.mark.parametrize("sql", [
    f"SELECT COUNT(*) FROM transactions WHERE {OK_RANGE}",
    f"SELECT b.name, COUNT(*) FROM transactions t JOIN banks b ON t.bank_id = b.bank_id "
    f"WHERE t.{OK_RANGE} GROUP BY 1",
    f"SELECT COUNT(*) FROM transactions t, banks b WHERE t.bank_id = b.bank_id AND t.{OK_RANGE}",
    "SELECT COUNT(*) FROM transactions WHERE STRFTIME(created_at, '%Y-%m') = '2026-09'",
    "SELECT COUNT(*) FROM transactions WHERE CAST(created_at AS DATE) BETWEEN '2026-09-01' AND '2026-09-30'",
    "WITH x AS (SELECT * FROM transactions WHERE created_at >= '2026-10-01') SELECT COUNT(*) FROM x",
    "SELECT * FROM banks",
])
def test_allowed(leader, sql):
    c = check(sql, leader)
    assert c.ok, (c.code, c.reason)


def test_limit_added_and_tightened(leader):
    assert "LIMIT 10000" in check("SELECT * FROM banks", leader).sql
    assert "LIMIT 10000" in check("SELECT * FROM banks LIMIT 50000", leader).sql
    assert "LIMIT 5" in check("SELECT * FROM banks LIMIT 5", leader).sql


def test_region_scope_and_tables(south):
    c = check("SELECT COUNT(*) FROM transactions WHERE region = 'North' AND created_at >= '2026-10-01'", south)
    assert c.code == "region_scope"
    c = check("SELECT SUM(net_inr) FROM settlements", south)
    assert c.code == "forbidden_table" and "settlements" in c.message
    assert "North" not in c.message          # refusals never describe hidden data
    assert check("SELECT COUNT(*) FROM transactions WHERE region = 'South' "
                 "AND created_at >= '2026-10-01'", south).ok
