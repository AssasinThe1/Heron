from datetime import date

import pytest

from engine.answer import Fmt, number_check
from engine.db import lookup_values
from engine.planner import last_full_week, parse_period, rules_plan


@pytest.fixture(scope="module")
def lk():
    return lookup_values()


def test_number_check_catches_invented_numbers():
    F = Fmt()
    text = f"Success rate fell to {F.pct(76.42)} ({F.pp(-2.5)}) on {F.date(date(2026, 10, 14))} [E1]."
    assert number_check(text, F) == (True, [])
    ok, bad = number_check(text + " About 3,000 customers were affected.", F)
    assert not ok and bad == ["3,000"]


def test_number_check_ignores_tags_and_labels():
    F = Fmt()
    text = f"{F.label('3DS_AUTH_FAILED')} rose to {F.share(0.31)} [E12, E13]."
    assert number_check(text, F)[0]


def test_inr_format():
    F = Fmt()
    assert F.inr(389794321) == "₹38.98 crore"
    assert F.inr(-471253) == "−₹4.71 lakh"
    assert F.inr(-0.0) == "₹0"


@pytest.mark.parametrize("text,expect", [
    ("in September", (date(2026, 9, 1), date(2026, 9, 30))),
    ("between 12 and 18 October", (date(2026, 10, 12), date(2026, 10, 18))),
    ("12–18 Oct", (date(2026, 10, 12), date(2026, 10, 18))),
    ("mid-October", (date(2026, 10, 12), date(2026, 10, 18))),
    ("the week of 12 October", (date(2026, 10, 12), date(2026, 10, 18))),
    ("on 14 October", (date(2026, 10, 14), date(2026, 10, 14))),
    ("last month", (date(2026, 10, 1), date(2026, 10, 31))),
    ("last full week", (date(2026, 10, 19), date(2026, 10, 25))),
    ("q3", (date(2026, 7, 1), date(2026, 9, 30))),
    ("from 2026-09-03 to 2026-09-09", (date(2026, 9, 3), date(2026, 9, 9))),
])
def test_periods(text, expect):
    p, _label, vague, _ = parse_period(f"success rate {text}")
    assert p == expect and not vague


@pytest.mark.parametrize("text", ["last week", "lately", "how are we doing"])
def test_vague_periods(text):
    p, _, vague, cands = parse_period(text)
    assert p is None and vague and len(cands) >= 3


def test_last_full_week_is_monday_to_sunday():
    a, b = last_full_week(date(2026, 11, 1))
    assert a.weekday() == 0 and b.weekday() == 6 and b < date(2026, 11, 1)


def test_long_ranges_parse_so_they_can_be_refused():
    p, *_ = parse_period("success rate over the last 3 years")
    assert (p[1] - p[0]).days > 1000


@pytest.mark.parametrize("q,kind", [
    ("Why did our success rate drop last week?", "why"),
    ("Delete the test merchants", "write"),
    ("drop table transactions", "write"),
    ("Give me customer phone numbers", "pii"),
    ("Show me every transaction", "dump"),
    ("How many transactions in September?", "metric"),
    ("Reconcile October settlements with transactions", "reconcile"),
])
def test_kinds(lk, q, kind):
    assert rules_plan(q, lk).kind == kind


def test_grouping_and_filters(lk):
    it = rules_plan("Which merchant category had the lowest success rate in October?", lk)
    assert it.group_by == ["category"] and it.order == "asc" and it.limit == 1
    it = rules_plan("Break Coral Bank UPI down by hour", lk)
    assert it.filters == {"method": ["upi"], "bank": ["Coral Bank"]} and it.grain == "hour"
    it = rules_plan("success rate for unique orders in September", lk)
    assert it.metric == "order_success_rate"
