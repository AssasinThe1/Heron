import math

import pandas as pd
import pytest

from engine.analysis import confidence, decompose, wilson


def frame(rows):
    return pd.DataFrame(rows, columns=["period", "segment", "attempts", "successes"])


def test_decomposition_is_exact():
    df = frame([("baseline", "card", 300, 195), ("baseline", "upi", 600, 516), ("baseline", "wallet", 100, 90),
                ("current", "card", 500, 320), ("current", "upi", 550, 440), ("current", "wallet", 90, 81)])
    seg, t = decompose(df)
    assert t["mix"] + t["rate"] == pytest.approx(t["delta"], abs=1e-12)
    assert seg.contrib.sum() == pytest.approx(t["delta"], abs=1e-12)
    assert t["R0"] == pytest.approx((195 + 516 + 90) / 1000)


def test_pure_mix_shift_simpson():
    """No segment changes its rate, the total still falls: all of it is mix."""
    df = frame([("baseline", "card", 200, 130), ("baseline", "upi", 800, 720),
                ("current", "card", 600, 390), ("current", "upi", 400, 360)])
    seg, t = decompose(df)
    assert t["delta"] < -0.09
    assert t["rate"] == pytest.approx(0, abs=1e-12)
    assert t["mix"] == pytest.approx(t["delta"])


def test_pure_rate_change():
    df = frame([("baseline", "a", 500, 450), ("baseline", "b", 500, 400),
                ("current", "a", 500, 350), ("current", "b", 500, 400)])
    seg, t = decompose(df)
    assert t["mix"] == pytest.approx(0, abs=1e-12)
    assert seg.loc["a", "rate"] == pytest.approx(t["delta"])


def test_new_segment_counts_as_mix():
    df = frame([("baseline", "a", 1000, 800), ("current", "a", 800, 640), ("current", "new", 200, 120)])
    seg, t = decompose(df)
    assert seg.loc["new", "rate"] == 0
    assert t["mix"] + t["rate"] == pytest.approx(t["delta"])


def test_wilson():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_confidence_rules():
    assert confidence(0.8, 10000, True, False, False)[0] == "high"
    assert confidence(0.8, 10000, False, False, False)[0] == "medium"   # one signal missing
    assert confidence(0.5, 10000, True, False, False)[0] == "medium"
    assert confidence(0.2, 10000, True, False, False)[0] == "low"
    assert confidence(0.8, 10000, True, True, False)[0] == "low"        # data-quality warning
    assert confidence(0.8, 10000, True, False, True)[0] == "low"        # ranges overlap
