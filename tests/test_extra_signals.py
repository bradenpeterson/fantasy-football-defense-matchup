import importlib.util
import inspect
import math

import pandas as pd
import pytest

from dstmatchup import signals
from dstmatchup.signals import build_signal_frame, rest_signal

CFG = {"scoring": {"blend_recent": 0.5, "last_n": 4}}
SIGNAL_NAMES = [
    "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
    "opp_ppg", "own_defense", "home_away", "rest",
]


# --- rest_signal -----------------------------------------------------------

def test_rest_signal_bands():
    assert rest_signal(4) == 1.0
    assert rest_signal(5) == 1.0
    assert rest_signal(14) == -1.0
    assert rest_signal(10) == -1.0
    assert rest_signal(7) == 0.0


def test_rest_signal_nan_is_nan():
    assert math.isnan(rest_signal(float("nan")))


# --- weather and OL-injury signals are gone --------------------------------

def test_weather_and_ol_injury_code_is_gone():
    assert importlib.util.find_spec("dstmatchup.weather") is None
    for name in ("ol_missing", "weather_signal", "roof_signal", "INJURY_WEIGHTS", "OL_POSITIONS"):
        assert not hasattr(signals, name), name
    params = set(inspect.signature(build_signal_frame).parameters)
    assert params.isdisjoint({"injuries", "depth", "forecast", "week"})


# --- build_signal_frame ----------------------------------------------------

def _matchups():
    return pd.DataFrame([
        dict(team="DEN", opponent="KC", home=True, rest_days=7, opp_rest_days=4, played=False),
        dict(team="KC", opponent="DEN", home=False, rest_days=4, opp_rest_days=7, played=False),
        dict(team="BUF", opponent="MIA", home=True, rest_days=14, opp_rest_days=7, played=False),
        dict(team="MIA", opponent="BUF", home=False, rest_days=7, opp_rest_days=14, played=False),
    ])


def _lines():
    return pd.DataFrame([
        dict(team="DEN", opponent="KC", home=True, implied_total=24.5, opp_implied_total=21.0),
        dict(team="KC", opponent="DEN", home=False, implied_total=21.0, opp_implied_total=24.5),
        dict(team="BUF", opponent="MIA", home=True, implied_total=20.0, opp_implied_total=30.0),
        dict(team="MIA", opponent="BUF", home=False, implied_total=30.0, opp_implied_total=20.0),
    ])


def _rates():
    rows = {}
    for team in ("DEN", "KC", "BUF", "MIA"):
        rows[team] = {"games": 4}
        for name, val in (("turnovers_pg", 1.0), ("sack_rate_allowed", 0.05), ("ppg", 22.0),
                          ("def_sacks_pg", 2.0), ("def_takeaways_pg", 1.0)):
            rows[team][f"{name}_season"] = val
            rows[team][f"{name}_recent"] = val
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("team")


def _allowed():
    return pd.DataFrame([
        dict(offense=t, games=4, dst_pts_allowed_per_game=5.0) for t in ("DEN", "KC", "BUF", "MIA")
    ])


QB_TABLE = pd.DataFrame([
    dict(team="KC", qb_name="Patrick Mahomes", status="out"),
    dict(team="DEN", qb_name="Bo Nix", status="starter"),
    dict(team="MIA", qb_name="Tua", status="unknown"),
    dict(team="BUF", qb_name="Josh Allen", status="backup"),
])


def _frame(**kw):
    return build_signal_frame(_matchups(), _rates(), _allowed(), _lines(), CFG, **kw)


def test_frame_has_exactly_the_nine_signal_names():
    frame = _frame(qb_table=QB_TABLE)
    assert [c for c in frame.columns if c in SIGNAL_NAMES] == SIGNAL_NAMES
    assert not {"weather", "ol_injuries"} & set(frame.columns)


def test_frame_qb_status_uses_opponent_qb():
    frame = _frame(qb_table=QB_TABLE)
    # DEN faces KC, whose QB is out -> 1.0; BUF faces MIA, status unknown -> NaN.
    assert frame.loc["DEN", "qb_status"] == pytest.approx(1.0)
    assert frame.loc["KC", "qb_status"] == pytest.approx(0.0)  # faces DEN starter
    assert math.isnan(frame.loc["BUF", "qb_status"])


def test_frame_rest_uses_opponent_rest_days():
    frame = _frame()
    assert frame.loc["DEN", "rest"] == 1.0   # opp KC had 4 days
    assert frame.loc["KC", "rest"] == 0.0    # opp DEN had 7 days
    assert frame.loc["BUF", "rest"] == 0.0   # opp MIA had 7 days
    assert frame.loc["MIA", "rest"] == -1.0  # opp BUF had 14 days


def test_frame_missing_qb_table_is_nan_not_zero():
    assert _frame()["qb_status"].isna().all()
