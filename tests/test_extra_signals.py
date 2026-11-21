import math
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from dstmatchup import weather
from dstmatchup.signals import build_signal_frame, ol_missing, rest_signal
from dstmatchup.teams import ALL_TEAMS
from dstmatchup.weather import STADIUM_COORDS, fetch_forecast, roof_signal, weather_signal

CFG = {"scoring": {"blend_recent": 0.5, "last_n": 4}}
SIGNAL_NAMES = [
    "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
    "opp_ppg", "own_defense", "home_away", "weather", "rest", "ol_injuries",
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


# --- ol_missing ------------------------------------------------------------

def _depth(team, rows):
    return pd.DataFrame(
        [dict(team=team, player_name=n, gsis_id=g, pos_abb=p, pos_rank=r) for n, g, p, r in rows]
    )


def _injury(team, week, gsis_id, status, position="T"):
    return dict(team=team, week=week, gsis_id=gsis_id, position=position, report_status=status)


BUF_DEPTH = _depth("BUF", [
    ("Dion Dawkins", "G1", "LT", 1),
    ("Spencer Brown", "G2", "RT", 1),
    ("Alec Anderson", "G3", "LG", 1),
    ("Backup Guard", "G4", "LG", 2),
    ("Connor McGovern", "G5", "C", 1),
    ("O'Cyrus Torrence", "G6", "RG", 1),
    ("Reserve Tackle", "G7", "RT", 6),
])


def test_ol_missing_weights():
    injuries = pd.DataFrame([
        _injury("BUF", 11, "G1", "Out"),
        _injury("BUF", 11, "G3", "Questionable", position="G"),
    ])
    assert ol_missing(injuries, BUF_DEPTH, "BUF", 11) == pytest.approx(1.25)


def test_ol_missing_doubtful_weight():
    injuries = pd.DataFrame([_injury("BUF", 11, "G5", "Doubtful", position="C")])
    assert ol_missing(injuries, BUF_DEPTH, "BUF", 11) == pytest.approx(0.75)


def test_ol_missing_rank_two_and_non_ol_add_zero():
    injuries = pd.DataFrame([
        _injury("BUF", 11, "G4", "Out", position="G"),  # starter-rank 2
        _injury("BUF", 11, "G7", "Out"),                # rank 6
    ])
    assert ol_missing(injuries, BUF_DEPTH, "BUF", 11) == 0.0


def test_ol_missing_no_injury_rows_is_nan_not_healthy():
    empty = pd.DataFrame(columns=["team", "week", "gsis_id", "position", "report_status"])
    assert math.isnan(ol_missing(empty, BUF_DEPTH, "BUF", 11))


def test_ol_missing_no_rows_for_week_or_team_is_nan():
    injuries = pd.DataFrame([
        _injury("BUF", 10, "G1", "Out"),
        _injury("MIA", 11, "G1", "Out"),
    ])
    assert math.isnan(ol_missing(injuries, BUF_DEPTH, "BUF", 11))


def test_ol_missing_practice_only_rows_is_nan():
    # nflverse practice reports have a NaN report_status; game statuses come later.
    injuries = pd.DataFrame([_injury("BUF", 11, "G1", None), _injury("BUF", 11, "G5", float("nan"))])
    assert math.isnan(ol_missing(injuries, BUF_DEPTH, "BUF", 11))


def test_ol_missing_published_out_starter_counts_despite_practice_rows():
    injuries = pd.DataFrame([
        _injury("BUF", 11, "G1", "Out"),
        _injury("BUF", 11, "G5", None),
    ])
    assert ol_missing(injuries, BUF_DEPTH, "BUF", 11) == pytest.approx(1.0)


def test_ol_missing_published_report_with_no_ol_hurt_is_zero():
    injuries = pd.DataFrame([_injury("BUF", 11, "WR1", "Out", position="WR")])
    assert ol_missing(injuries, BUF_DEPTH, "BUF", 11) == 0.0


def test_ol_missing_no_depth_rows_for_team_is_nan():
    injuries = pd.DataFrame([_injury("BUF", 11, "G1", "Out")])
    assert math.isnan(ol_missing(injuries, BUF_DEPTH, "SEA", 11))


def test_ol_missing_uses_only_latest_depth_snapshot():
    old = BUF_DEPTH.assign(dt="2026-11-10T00:00:00Z")
    old.loc[old["gsis_id"] == "G1", "gsis_id"] = "OLD1"
    new = BUF_DEPTH.assign(dt="2026-11-20T00:00:00Z")
    depth = pd.concat([old, new], ignore_index=True)
    # The player who started in the old snapshot only must not count.
    injuries = pd.DataFrame([_injury("BUF", 11, "OLD1", "Out")])
    assert ol_missing(injuries, depth, "BUF", 11) == 0.0
    injuries = pd.DataFrame([_injury("BUF", 11, "G1", "Out")])
    assert ol_missing(injuries, depth, "BUF", 11) == pytest.approx(1.0)


# --- roof_signal / weather_signal -----------------------------------------

def test_roof_signal():
    assert roof_signal("dome") == 0.0
    assert roof_signal("closed") == 0.0
    assert roof_signal("outdoors") == 0.5
    assert roof_signal("open") == 0.5
    assert roof_signal(None) is None
    assert roof_signal(float("nan")) is None


def test_weather_signal_adds_bonus_for_wind_or_cold_outdoors():
    assert weather_signal("outdoors", {"temp": 60.0, "wind": 10.0}) == pytest.approx(0.5)
    assert weather_signal("outdoors", {"temp": 60.0, "wind": 16.0}) == pytest.approx(1.0)
    assert weather_signal("open", {"temp": 20.0, "wind": 5.0}) == pytest.approx(1.0)


def test_weather_signal_dome_ignores_forecast_and_missing_forecast_is_roof_only():
    assert weather_signal("dome", {"temp": 10.0, "wind": 30.0}) == 0.0
    assert weather_signal("outdoors", None) == pytest.approx(0.5)
    assert weather_signal(None, {"temp": 10.0, "wind": 30.0}) is None


# --- fetch_forecast --------------------------------------------------------

def test_forecast_failure_returns_none():
    with mock.patch("requests.get", side_effect=requests.ConnectionError("offline")):
        assert fetch_forecast(42.0, -71.0, "2026-11-22T13:00") is None


def test_forecast_parses_kickoff_hour():
    payload = {"hourly": {
        "time": ["2026-11-22T17:00", "2026-11-22T18:00", "2026-11-22T19:00"],
        "temperature_2m": [30.0, 28.5, 27.0],
        "wind_speed_10m": [9.0, 17.5, 12.0],
    }}
    resp = mock.Mock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    with mock.patch("requests.get", return_value=resp):
        # 13:00 Eastern on 2026-11-22 is 18:00 UTC.
        out = fetch_forecast(42.0, -71.0, "2026-11-22T13:00")
    assert out == {"temp": 28.5, "wind": 17.5}


def test_stadium_coords_cover_all_teams():
    assert len(STADIUM_COORDS) == 32
    assert set(STADIUM_COORDS) == set(ALL_TEAMS)
    for lat, lon in STADIUM_COORDS.values():
        assert 25 < lat < 50 and -125 < lon < -70


# --- build_signal_frame extras --------------------------------------------

def _matchups():
    return pd.DataFrame([
        dict(team="DEN", opponent="KC", home=True, rest_days=7, opp_rest_days=4, roof="outdoors",
             played=False),
        dict(team="KC", opponent="DEN", home=False, rest_days=4, opp_rest_days=7, roof="outdoors",
             played=False),
        dict(team="BUF", opponent="MIA", home=True, rest_days=14, opp_rest_days=7, roof="dome",
             played=False),
        dict(team="MIA", opponent="BUF", home=False, rest_days=7, opp_rest_days=14, roof="dome",
             played=False),
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


def test_frame_has_exactly_the_eleven_signal_names():
    frame = _frame(qb_table=QB_TABLE, depth=BUF_DEPTH,
                   injuries=pd.DataFrame([_injury("BUF", 11, "G1", "Out")]), week=11,
                   forecast={"DEN": {"temp": 20.0, "wind": 5.0}})
    assert [c for c in frame.columns if c in SIGNAL_NAMES] == SIGNAL_NAMES
    assert set(SIGNAL_NAMES) <= set(frame.columns)


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


def test_frame_ol_injuries_is_opponent_missing_weight():
    depth = pd.concat([BUF_DEPTH, _depth("KC", [("KC Tackle", "K1", "LT", 1)])], ignore_index=True)
    injuries = pd.DataFrame([_injury("BUF", 11, "G1", "Out")])
    # KC has a published report with no OL hurt, so DEN (facing KC) gets 0.0.
    injuries = pd.concat([injuries, pd.DataFrame([_injury("KC", 11, "K9", "Out", position="WR")])])
    frame = _frame(depth=depth, injuries=injuries, week=11)
    # MIA faces BUF: BUF starter out -> 1.0. DEN faces KC: published, no OL hurt -> 0.0.
    assert frame.loc["MIA", "ol_injuries"] == pytest.approx(1.0)
    assert frame.loc["DEN", "ol_injuries"] == pytest.approx(0.0)
    assert math.isnan(frame.loc["BUF", "ol_injuries"])  # MIA has no depth rows


def test_frame_missing_optional_inputs_are_nan_not_zero():
    frame = _frame()
    for col in ("qb_status", "ol_injuries"):
        assert frame[col].isna().all(), col


def test_frame_weather_uses_roof_and_forecast_for_defense():
    frame = _frame(forecast={"DEN": {"temp": 20.0, "wind": 5.0}})
    assert frame.loc["DEN", "weather"] == pytest.approx(1.0)  # outdoors, cold
    assert frame.loc["KC", "weather"] == pytest.approx(0.5)   # no forecast -> roof only
    assert frame.loc["BUF", "weather"] == pytest.approx(0.0)  # dome
