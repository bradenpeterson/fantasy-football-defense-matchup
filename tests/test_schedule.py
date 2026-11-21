import math

import pandas as pd

from dstmatchup.schedule import bye_teams, current_week, get_lines, matchups
from dstmatchup.teams import ALL_TEAMS


def _schedule():
    rows = [
        # Week 10: fully played.
        dict(week=10, gameday="2026-11-08", gametime="13:00", away_team="BUF", home_team="MIA",
             spread_line=-2.5, total_line=44.0, result=3.0, away_rest=7, home_rest=7,
             away_qb_name="J. Allen", home_qb_name="T. Tagovailoa"),
        # Week 11: one played, two unplayed.
        dict(week=11, gameday="2026-11-15", gametime="13:00", away_team="KC", home_team="DEN",
             spread_line=3.5, total_line=45.5, result=-7.0, away_rest=7, home_rest=6,
             away_qb_name="P. Mahomes", home_qb_name="B. Nix"),
        dict(week=11, gameday="2026-11-15", gametime="16:25", away_team="DAL", home_team="NYG",
             spread_line=3.5, total_line=45.5, result=None, away_rest=7, home_rest=7,
             away_qb_name="D. Prescott", home_qb_name="D. Jones"),
        dict(week=11, gameday="2026-11-16", gametime="20:20", away_team="SEA", home_team="SF",
             spread_line=6.0, total_line=40.0, result=None, away_rest=7, home_rest=7,
             away_qb_name="G. Smith", home_qb_name="B. Purdy"),
    ]
    return pd.DataFrame(rows)


def test_implied_totals():
    lines = get_lines(_schedule(), 11)
    home_row = lines[(lines.team == "DEN")].iloc[0]
    away_row = lines[(lines.team == "KC")].iloc[0]
    assert bool(home_row.home) is True
    assert math.isclose(home_row.implied_total, 24.5)
    assert math.isclose(away_row.implied_total, 21.0)
    assert math.isclose(home_row.opp_implied_total, 21.0)
    assert math.isclose(away_row.opp_implied_total, 24.5)
    assert math.isclose(away_row.spread_for_team, -3.5)
    assert math.isclose(home_row.spread_for_team, 3.5)
    assert away_row.opponent == "DEN"
    assert not bool(away_row.home)


def test_current_week_is_earliest_with_unplayed():
    assert current_week(_schedule()) == 11


def test_current_week_all_played_returns_next():
    played = _schedule()
    played["result"] = 1.0
    assert current_week(played) == 11  # no week 12 in fixture -> stays at max week
    extra = pd.concat([played, played.assign(week=12)], ignore_index=True)
    assert current_week(extra) == 12


def test_matchups_marks_played_and_bye():
    m = matchups(_schedule(), 11)
    assert len(m) == 6  # 3 games x 2 teams
    kc = m[m.team == "KC"].iloc[0]
    assert bool(kc.played) is True
    dal = m[m.team == "DAL"].iloc[0]
    assert bool(dal.played) is False
    assert "roof" not in m.columns
    assert dal.rest_days == 7
    den = m[m.team == "DEN"].iloc[0]
    assert den.rest_days == 6
    assert den.opp_rest_days == 7

    byes = bye_teams(_schedule(), 11)
    assert "BUF" in byes and "MIA" in byes
    assert "KC" not in byes
    assert len(byes) == len(ALL_TEAMS) - 6
