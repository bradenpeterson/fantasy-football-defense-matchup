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


def test_missing_leader_is_unknown():
    assert qb_status("Bo Nix", None, None) == "unknown"


def test_missing_starter_is_unknown():
    assert qb_status(None, "R.Wilson", None) == "unknown"


def test_one_sided_input_with_out_injury_is_out():
    assert qb_status("Bo Nix", None, "Out") == "out"
    assert qb_status(None, "R.Wilson", "Questionable") == "questionable"


def test_week_one_empty_pbp_makes_every_team_unknown():
    schedule = pd.DataFrame({
        "week": [1, 1],
        "home_team": ["BUF", "KC"],
        "away_team": ["MIA", "DEN"],
        "home_qb_name": ["Josh Allen", "Patrick Mahomes"],
        "away_qb_name": ["Tua Tagovailoa", "Bo Nix"],
    })
    pbp = pd.DataFrame(columns=["season_type", "week", "posteam", "passer_player_name", "qb_dropback"])
    table = qb_status_table(schedule, pbp, pd.DataFrame(), 1)
    assert len(table) == 4
    assert set(table["status"]) == {"unknown"}


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


def _games(team, name, weeks, plays=3):
    return [row for week in weeks for row in _plays(team, name, week, plays)]


def _week_eleven(named_away, injuries=None):
    schedule = pd.DataFrame({
        "week": [11],
        "home_team": ["NYG"],
        "away_team": ["JAX"],
        "home_qb_name": [None],
        "away_qb_name": [named_away],
    })
    if injuries is None:
        injuries = pd.DataFrame(columns=["team", "week", "position", "full_name", "report_status"])
    return schedule, injuries


def test_recent_leader_over_last_two_games_is_starter():
    # A.Alpha led weeks 1-6 and B.Beta led weeks 7-10. A is the season leader,
    # but B led each of the last two games, so the named B is a starter.
    pbp = pd.DataFrame(_games("JAX", "A.Alpha", range(1, 7)) + _games("JAX", "B.Beta", range(7, 11)))
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    got = dict(zip(table["team"], table["status"]))
    assert got["JAX"] == "starter"


def test_one_game_fill_in_stays_backup():
    # A.Alpha led weeks 1-9; B.Beta led only week 10 (one game) and is named.
    pbp = pd.DataFrame(_games("JAX", "A.Alpha", range(1, 10)) + _games("JAX", "B.Beta", [10]))
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "backup"


def test_season_leader_named_is_starter():
    # A.Alpha leads weeks 1-6 and B.Beta leads weeks 7-10; A is the season leader.
    pbp = pd.DataFrame(_games("JAX", "A.Alpha", range(1, 7)) + _games("JAX", "B.Beta", range(7, 11)))
    schedule, injuries = _week_eleven("Al Alpha")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "starter"


def test_team_with_one_prior_game_skips_recent_rule():
    # Only one prior game, led by A.Alpha. Named B.Beta is not the season leader,
    # and with fewer than two prior games the recent rule cannot make B a starter.
    pbp = pd.DataFrame(_games("JAX", "A.Alpha", [1]))
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "backup"


def test_recent_games_use_own_played_games_across_a_bye():
    # JAX has a bye in week 10. Its last two played games are weeks 8 and 9, both led
    # by B.Beta, so B is a starter for week 11 even though week 10 has no pbp.
    pbp = pd.DataFrame(_games("JAX", "A.Alpha", range(1, 8)) + _games("JAX", "B.Beta", [8, 9]))
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "starter"


def test_injury_overrides_recent_leader_rule():
    pbp = pd.DataFrame(_games("JAX", "A.Alpha", range(1, 7)) + _games("JAX", "B.Beta", range(7, 11)))
    schedule, injuries = _week_eleven("Bob Beta")
    injuries = pd.DataFrame([
        dict(team="JAX", week=11, position="QB", full_name="Bob Beta", report_status="Out"),
    ])
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "out"


def test_week_one_with_recent_rule_is_unknown():
    schedule = pd.DataFrame({
        "week": [1], "home_team": ["NYG"], "away_team": ["JAX"],
        "home_qb_name": [None], "away_qb_name": ["Bob Beta"],
    })
    pbp = pd.DataFrame(columns=["season_type", "week", "posteam", "passer_player_name", "qb_dropback"])
    table = qb_status_table(schedule, pbp, pd.DataFrame(), 1, recent_games=2)
    assert set(table["status"]) == {"unknown"}


def test_window_ignores_week_of_and_after_the_week_being_scored():
    # A.Alpha leads weeks 1-9; B.Beta leads week 10 and then, in the scored week 11
    # and a later week 12. Only weeks before 11 may count: last two are 9 (A), 10 (B).
    pbp = pd.DataFrame(
        _games("JAX", "A.Alpha", range(1, 10))
        + _games("JAX", "B.Beta", [10, 11, 12])
    )
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "backup"


def test_recent_games_are_regular_season_only():
    # A.Alpha leads REG weeks 1-8. B.Beta leads only POST weeks 9-10, which must not
    # count as the last two games. Last two REG games are 7 and 8, both A.
    pbp = pd.DataFrame(
        _games("JAX", "A.Alpha", range(1, 9))
        + [row for week in (9, 10) for row in _plays("JAX", "B.Beta", week, 3, season_type="POST")]
    )
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "backup"


def test_game_without_named_passer_counts_as_a_game_with_no_leader():
    # Week 9 has dropbacks but no named passer. It is still one of the last two
    # played games, with no leader, so the recent rule fails. Week 8 and week 10
    # both led by B.Beta would wrongly pass if week 9 were ignored.
    unnamed = [dict(season_type="REG", week=9, posteam="JAX", passer_player_name=None, qb_dropback=1)
               for _ in range(3)]
    pbp = pd.DataFrame(
        _games("JAX", "A.Alpha", range(1, 8))
        + _games("JAX", "B.Beta", [8])
        + unnamed
        + _games("JAX", "B.Beta", [10])
    )
    schedule, injuries = _week_eleven("Bob Beta")
    table = qb_status_table(schedule, pbp, injuries, 11)
    assert dict(zip(table["team"], table["status"]))["JAX"] == "backup"
