from dstmatchup.explain import why_text


def test_why_text_lists_two_largest_drivers_with_direction():
    # z-weighted units times display_scale 25: 20, 10, -5, 0.25 points. Top two by size win.
    contribs = {"implied_total": 0.8, "qb_status": 0.4, "turnovers": -0.2, "sack_rate": 0.01}
    assert why_text(contribs, 25) == "▲ implied pts  ▲ QB"


def test_why_text_marks_negative_drivers_down():
    # -5 and -2.5 points, then rest at +0.25 (dropped as the third driver).
    contribs = {"turnovers": -0.2, "sack_rate": -0.1, "rest": 0.01}
    assert why_text(contribs, 25) == "▼ turnovers  ▼ sacks"


def test_why_text_is_dash_when_no_driver_reaches_one_point():
    # 0.03 and 0.02 z at scale 25 are 0.75 and 0.5 points: below the 1.0 point floor.
    assert why_text({"implied_total": 0.03, "turnovers": -0.02}, 25) == "-"


def test_why_text_breaks_equal_sizes_by_signal_order():
    # Both are 5 points. implied_total comes before turnovers in SIGNAL_NAMES, whatever the dict order.
    assert why_text({"turnovers": -0.2, "implied_total": 0.2}, 25) == "▲ implied pts  ▼ turnovers"


def test_why_text_skips_nan_contributions():
    nan = float("nan")
    contribs = {"implied_total": nan, "qb_status": 0.4, "turnovers": nan}
    assert why_text(contribs, 25) == "▲ QB"


def test_why_text_shows_only_one_driver_when_only_one_qualifies():
    assert why_text({"implied_total": 0.01, "qb_status": 0.4}, 25) == "▲ QB"


def test_labels_follow_signal_names_order():
    from dstmatchup.pipeline import SIGNAL_NAMES
    from dstmatchup.explain import LABELS

    assert tuple(LABELS) == SIGNAL_NAMES
