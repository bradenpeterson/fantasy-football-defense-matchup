import numpy as np
import pandas as pd
import pytest

from dstmatchup.qb import names_match, qb_signal, qb_status, qb_status_table


def test_starter_equal_to_leader_is_starter():
    assert qb_status("Josh Allen", "J.Allen", None) == "starter"


def test_starter_differing_from_leader_is_backup():
    assert qb_status("Tua Tagovailoa", "R.Wilson", None) == "backup"


def test_out_injury_is_out():
    assert qb_status("Josh Allen", "J.Allen", "Out") == "out"
    assert qb_status("Josh Allen", "J.Allen", "Doubtful") == "out"


def test_questionable_injury_is_questionable():
    assert qb_status("Josh Allen", "J.Allen", "Questionable") == "questionable"


def test_missing_starter_and_leader_is_unknown():
    assert qb_status(None, None, None) == "unknown"
    assert qb_status(np.nan, None, "Out") == "unknown"


def test_injury_overrides_leader_match():
    # Starter equals leader, but the injury report says Out: injury wins.
    assert qb_status("Josh Allen", "J.Allen", "Out") == "out"


def test_missing_leader_uses_named_starter():
    assert qb_status("Bo Nix", None, None) == "starter"


def test_missing_starter_uses_leader():
    assert qb_status(None, "R.Wilson", None) == "starter"


def test_qb_signal_values():
    assert qb_signal("backup") == 1.0
    assert qb_signal("out") == 1.0
    assert qb_signal("questionable") == 0.5
    assert qb_signal("starter") == 0.0
    assert qb_signal("unknown") is None


def test_names_match_suffix_and_initial_forms():
    assert names_match("Joe Milton III", "Joe Milton")
    assert names_match("Marvin Harrison Jr.", "Marvin Harrison")
    assert names_match("J.Milton", "Joe Milton")
    assert names_match("joe milton", "JOE MILTON")
    assert not names_match("Joe Milton", "Jake Milton")
    assert not names_match("Joe Milton", "Joe Burrow")
    assert not names_match(None, "Joe Milton")


def test_suffix_name_is_starter_against_abbreviated_leader():
    assert qb_status("Joe Milton III", "J.Milton", None) == "starter"


def _plays(team, name, week, n, season_type="REG"):
    return [
        dict(season_type=season_type, week=week, posteam=team,
             passer_player_name=name, qb_dropback=1)
        for _ in range(n)
    ]


def test_qb_status_table_week_three():
    schedule = pd.DataFrame({
        "week": [3, 3, 4],
        "home_team": ["BUF", "KC", "BUF"],
        "away_team": ["MIA", "DEN", "NYJ"],
        "home_qb_name": ["Josh Allen", None, "Josh Allen"],
        "away_qb_name": ["Tua Tagovailoa", "Bo Nix", "Aaron Rodgers"],
    })
    rows = []
    rows += _plays("BUF", "J.Allen", 1, 3)
    rows += _plays("BUF", "T.Heinicke", 2, 1)
    rows += _plays("MIA", "T.Tagovailoa", 1, 2)
    rows += _plays("MIA", "R.Wilson", 2, 5)
    rows += _plays("MIA", "J.Milton", 2, 50, season_type="POST")
    rows += _plays("MIA", "J.Milton", 3, 50)
    pbp = pd.DataFrame(rows)
    injuries = pd.DataFrame([
        dict(team="BUF", week=3, position="QB", full_name="Josh Allen", report_status="Questionable"),
        dict(team="DEN", week=3, position="QB", full_name="Bo Nix", report_status="Out"),
        dict(team="KC", week=3, position="RB", full_name="Isiah Pacheco", report_status="Out"),
        dict(team="MIA", week=2, position="QB", full_name="Tua Tagovailoa", report_status="Out"),
    ])

    table = qb_status_table(schedule, pbp, injuries, 3)

    assert list(table.columns) == ["team", "qb_name", "status"]
    got = dict(zip(table["team"], zip(table["qb_name"], table["status"])))
    assert set(got) == {"BUF", "MIA", "KC", "DEN"}
    assert got["BUF"] == ("Josh Allen", "questionable")
    assert got["MIA"] == ("Tua Tagovailoa", "backup")
    assert got["DEN"] == ("Bo Nix", "out")
    assert pd.isna(got["KC"][0])
    assert got["KC"][1] == "unknown"
