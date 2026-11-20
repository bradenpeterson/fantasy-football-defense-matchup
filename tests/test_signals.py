import math

import numpy as np
import pandas as pd
import pytest

from dstmatchup.signals import blend, build_signal_frame, team_rates

CFG = {"scoring": {"blend_recent": 0.5, "last_n": 4}}


def _stat(team, week, **kw):
    row = dict(
        team=team, week=week, season_type="REG", opponent_team="OPP",
        attempts=30, sacks_suffered=2, passing_interceptions=0, sack_fumbles_lost=0,
        rushing_fumbles_lost=0, receiving_fumbles_lost=0, def_sacks=2,
        def_interceptions=0, fumble_recovery_opp=0,
    )
    row.update(kw)
    return row


def _stats(rows):
    return pd.DataFrame(rows)


def test_rates_only_use_prior_weeks():
    rows = [_stat("BUF", w, passing_interceptions=1) for w in range(1, 11)]
    # Week 11 is the week being scored; its turnovers must not leak in.
    rows.append(_stat("BUF", 11, passing_interceptions=5))
    rates = team_rates(_stats(rows), upto_week=11)
    assert rates.loc["BUF", "games"] == 10
    assert math.isclose(rates.loc["BUF", "turnovers_pg_season"], 1.0)


def test_turnovers_and_defense_per_game():
    rows = [
        _stat("BUF", 1, passing_interceptions=1, sack_fumbles_lost=1, rushing_fumbles_lost=0,
              receiving_fumbles_lost=1, def_sacks=3, def_interceptions=1, fumble_recovery_opp=2),
        _stat("BUF", 2, passing_interceptions=0, def_sacks=1, def_interceptions=0,
              fumble_recovery_opp=0),
    ]
    rates = team_rates(_stats(rows), upto_week=3)
    assert math.isclose(rates.loc["BUF", "turnovers_pg_season"], 1.5)
    assert math.isclose(rates.loc["BUF", "def_sacks_pg_season"], 2.0)
    assert math.isclose(rates.loc["BUF", "def_takeaways_pg_season"], 1.5)


def test_sack_rate_is_pooled_and_zero_dropbacks_is_nan():
    rows = [
        _stat("BUF", 1, attempts=30, sacks_suffered=3),
        _stat("BUF", 2, attempts=10, sacks_suffered=1),
        _stat("NYJ", 1, attempts=0, sacks_suffered=0),
        _stat("NYJ", 2, attempts=0, sacks_suffered=0),
    ]
    rates = team_rates(_stats(rows), upto_week=3)
    assert math.isclose(rates.loc["BUF", "sack_rate_allowed_season"], 4 / 44)
    assert math.isnan(rates.loc["NYJ", "sack_rate_allowed_season"])
    assert math.isnan(rates.loc["NYJ", "sack_rate_allowed_recent"])


def test_recent_window_uses_last_n_games():
    rows = [_stat("BUF", w, passing_interceptions=w) for w in range(1, 7)]
    rates = team_rates(_stats(rows), upto_week=7, last_n=2)
    assert math.isclose(rates.loc["BUF", "turnovers_pg_season"], 3.5)
    assert math.isclose(rates.loc["BUF", "turnovers_pg_recent"], 5.5)


def test_ppg_from_schedule_and_nan_without_schedule():
    rows = [_stat("BUF", 1), _stat("BUF", 2)]
    schedule = pd.DataFrame([
        dict(week=1, game_type="REG", home_team="BUF", away_team="MIA", home_score=24, away_score=10),
        dict(week=2, game_type="REG", home_team="NYJ", away_team="BUF", home_score=7, away_score=17),
        dict(week=11, game_type="REG", home_team="BUF", away_team="KC", home_score=40, away_score=0),
    ])
    with_sched = team_rates(_stats(rows), upto_week=11, schedule=schedule)
    assert math.isclose(with_sched.loc["BUF", "ppg_season"], 20.5)
    no_sched = team_rates(_stats(rows), upto_week=11)
    assert math.isnan(no_sched.loc["BUF", "ppg_season"])


def test_team_with_zero_games_is_nan_not_zero():
    rows = [_stat("BUF", 1)]
    schedule = pd.DataFrame([
        dict(week=5, game_type="REG", home_team="SEA", away_team="SF", home_score=3, away_score=3),
    ])
    rates = team_rates(_stats(rows), upto_week=2, schedule=schedule)
    assert rates.loc["SEA", "games"] == 0
    for col in rates.columns:
        if col != "games":
            assert math.isnan(rates.loc["SEA", col]), col


def test_blend_half_half():
    season = pd.Series([10.0], index=["BUF"])
    recent = pd.Series([20.0], index=["BUF"])
    assert math.isclose(blend(season, recent, 0.5)["BUF"], 15.0)


def test_blend_recent_nan_falls_back_to_season():
    season = pd.Series([10.0, 4.0], index=["BUF", "KC"])
    recent = pd.Series([np.nan, 8.0], index=["BUF", "KC"])
    out = blend(season, recent, 0.5)
    assert math.isclose(out["BUF"], 10.0)
    assert math.isclose(out["KC"], 6.0)


def _rates_frame(values: dict) -> pd.DataFrame:
    """values: team -> dict of metric -> value (season and recent set equal)."""
    rows = {}
    for team, metrics in values.items():
        row = {"games": metrics.pop("games", 4)}
        for name, val in metrics.items():
            row[f"{name}_season"] = val
            row[f"{name}_recent"] = val
        rows[team] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("team")


@pytest.fixture
def signal_inputs():
    matchups = pd.DataFrame([
        dict(team="DEN", opponent="KC", home=True),
        dict(team="KC", opponent="DEN", home=False),
        dict(team="BUF", opponent="MIA", home=True),
        dict(team="MIA", opponent="BUF", home=False),
        dict(team="SEA", opponent="SF", home=True),
    ])
    lines = pd.DataFrame([
        dict(team="DEN", opponent="KC", home=True, implied_total=24.5, opp_implied_total=21.0),
        dict(team="KC", opponent="DEN", home=False, implied_total=21.0, opp_implied_total=24.5),
        dict(team="BUF", opponent="MIA", home=True, implied_total=20.0, opp_implied_total=30.0),
        dict(team="MIA", opponent="BUF", home=False, implied_total=30.0, opp_implied_total=20.0),
    ])
    rates = _rates_frame({
        "DEN": dict(turnovers_pg=1.0, sack_rate_allowed=0.05, ppg=22.0,
                    def_sacks_pg=2.5, def_takeaways_pg=1.0),
        "KC": dict(turnovers_pg=2.5, sack_rate_allowed=0.09, ppg=28.0,
                   def_sacks_pg=2.0, def_takeaways_pg=1.0),
        "BUF": dict(turnovers_pg=1.0, sack_rate_allowed=0.05, ppg=20.0,
                    def_sacks_pg=3.0, def_takeaways_pg=2.0),
        "MIA": dict(turnovers_pg=0.5, sack_rate_allowed=0.04, ppg=18.0,
                    def_sacks_pg=1.0, def_takeaways_pg=0.5),
    })
    allowed = pd.DataFrame([
        dict(offense="KC", games=4, dst_pts_allowed_per_game=6.0),
        dict(offense="DEN", games=4, dst_pts_allowed_per_game=2.0),
        dict(offense="MIA", games=4, dst_pts_allowed_per_game=9.0),
        dict(offense="BUF", games=4, dst_pts_allowed_per_game=4.0),
    ])
    return matchups, rates, allowed, lines


def test_signal_orientation_higher_is_better(signal_inputs):
    matchups, rates, allowed, lines = signal_inputs
    frame = build_signal_frame(matchups, rates, allowed, lines, CFG)
    # DEN faces KC (implied 21.0); BUF faces MIA (implied 30.0).
    assert frame.loc["DEN", "implied_total"] > frame.loc["BUF", "implied_total"]
    assert frame.loc["DEN", "implied_total"] == pytest.approx(-21.0)
    # DEN faces KC (2.5 turnovers/g); BUF faces MIA (0.5).
    assert frame.loc["DEN", "turnovers"] == pytest.approx(2.5)
    assert frame.loc["BUF", "turnovers"] == pytest.approx(0.5)
    # opp_ppg is negated: DEN faces KC (28 ppg) -> -28.
    assert frame.loc["DEN", "opp_ppg"] == pytest.approx(-28.0)
    # Own defense: DEN 2.5 sacks + 1.0 takeaways.
    assert frame.loc["DEN", "own_defense"] == pytest.approx(3.5)
    assert frame.loc["DEN", "dst_pts_allowed"] == pytest.approx(6.0)
    assert frame.loc["DEN", "home_away"] == 1.0
    assert frame.loc["KC", "home_away"] == 0.0
    assert frame.loc["DEN", "opp_games"] == 4
    assert frame.loc["DEN", "own_games"] == 4


def test_team_missing_from_rates_gets_nan_signals(signal_inputs):
    matchups, rates, allowed, lines = signal_inputs
    frame = build_signal_frame(matchups, rates, allowed, lines, CFG)
    assert math.isnan(frame.loc["SEA", "turnovers"])
    assert math.isnan(frame.loc["SEA", "dst_pts_allowed"])
    assert math.isnan(frame.loc["SEA", "own_defense"])
