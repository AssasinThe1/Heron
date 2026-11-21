"""The Heron cross-check: full-result match, and BIRD-style label-only answers."""
import pandas as pd

from engine.pipeline import results_match, top_label_match

RANKED = pd.DataFrame({"region": ["West", "South", "North"], "value": [12.5, 10.0, 9.0]})


def test_label_only_answer_matches_top_segment():
    assert top_label_match(pd.DataFrame({"region": ["West"]}), RANKED, ["region"])
    assert top_label_match(pd.DataFrame({"r": [" west "]}), RANKED, ["region"])


def test_wrong_label_is_rejected():
    assert not top_label_match(pd.DataFrame({"region": ["South"]}), RANKED, ["region"])
    assert not top_label_match(pd.DataFrame({"region": []}), RANKED, ["region"])


def test_full_listing_is_not_a_label_answer():
    """A 6+ row result is a listing, judged by results_match, not by its first row."""
    big = pd.DataFrame({"region": ["West"] * 6})
    assert not top_label_match(big, RANKED, ["region"])


def test_hour_labels_compare_as_timestamps():
    ranked = pd.DataFrame({"hour": pd.to_datetime(["2026-10-15 20:00", "2026-10-15 19:00"]),
                           "value": [40.0, 55.0]})
    assert top_label_match(pd.DataFrame({"hour": ["2026-10-15 20:00:00"]}), ranked, ["hour"])
    assert not top_label_match(pd.DataFrame({"hour": ["2026-10-15 19:00:00"]}), ranked, ["hour"])


def test_label_only_does_not_satisfy_value_match():
    assert not results_match(pd.DataFrame({"region": ["West"]}), RANKED, "percent")
