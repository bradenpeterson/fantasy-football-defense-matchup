from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dstmatchup import pipeline
from dstmatchup.config import load_config

REPO_CONFIG = str(Path(__file__).resolve().parent.parent / "config.toml")
SIGNALS = {
    "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
    "opp_ppg", "own_defense", "home_away", "weather", "rest", "ol_injuries",
}

# (home, away) pairings. Weeks 1-4 are played; week 5 is the week under test.
PAIRS = [("BUF", "DEN"), ("KC", "LA"), ("MIA", "NE"), ("NYJ", "PIT")]
ROOF = {"BUF": "outdoors", "KC": "outdoors", "MIA": "dome", "NYJ": "outdoors"}
QB = {
    "BUF": "Josh Allen", "KC": None, "LA": "Matthew Stafford", "MIA": "Tua Tagovailoa",
    "NE": "Drake Maye", "NYJ": "Aaron Rodgers", "PIT": "Russell Wilson",
}
# DEN's named week-5 starter differs from the pbp dropback leader, so DEN is "backup".
DEN_NAMED = "Jarrett Stidham"
DEN_LEADER = "Bo Nix"


@pytest.fixture
def cfg():
    return load_config(REPO_CONFIG)


def _game(week, home, away, played=False, total=44.0, spread=0.0, home_qb=None, away_qb=None):
    if played:
        home_score, away_score, result = 20.0, 17.0, 3.0
    else:
        home_score = away_score = result = np.nan
    return dict(
        season=2026, game_type="REG", week=week, gameday=f"2026-09-{week:02d}", gametime="13:00",
        home_team=home, away_team=away, home_score=home_score, away_score=away_score, result=result,
        roof=ROOF.get(home, "dome"), home_rest=7, away_rest=7,
        total_line=total, spread_line=spread,
        home_qb_name=home_qb, away_qb_name=away_qb,
    )


def make_schedule():
    rows = []
    for week in (1, 2, 3, 4):
        for home, away in PAIRS:
            rows.append(_game(week, home, away, played=True, home_qb=QB[home], away_qb=_away_qb(away)))
    # Week 5: BUF-DEN has the lowest implied total for DEN (18.5 vs 25.5 for BUF).
    rows.append(_game(5, "BUF", "DEN", total=44.0, spread=7.0, home_qb=QB["BUF"], away_qb=DEN_NAMED))
    rows.append(_game(5, "KC", "LA", total=50.0, spread=0.0, home_qb=QB["KC"], away_qb=QB["LA"]))
    rows.append(_game(5, "MIA", "NE", total=40.0, spread=0.0, home_qb=QB["MIA"], away_qb=QB["NE"]))
    rows.append(_game(5, "NYJ", "PIT", total=46.0, spread=3.0, home_qb=QB["NYJ"], away_qb=QB["PIT"]))
    rows.append(_game(5, "CHI", "GB", played=True, total=45.0, spread=0.0))
    return pd.DataFrame(rows)


def _away_qb(away):
    return DEN_LEADER if away == "DEN" else QB[away]


def make_team_stats():
    rows = []
    for week in (1, 2, 3, 4):
        for home, away in PAIRS:
            for team, opp in ((home, away), (away, home)):
                rows.append(dict(
                    team=team, opponent_team=opp, week=week, season_type="REG",
                    sacks_suffered=2, attempts=30,
                    passing_interceptions=2 if team == "DEN" else 0,
                    def_sacks=2, def_interceptions=0, fumble_recovery_opp=1, def_tds=0,
                ))
    return pd.DataFrame(rows)


def make_pbp():
    rows = []
    for week in (1, 2, 3, 4):
        for home, away in PAIRS:
            for team in (home, away):
                leader = DEN_LEADER if team == "DEN" else QB.get(team)
                if leader is None:
                    continue
                rows.append(dict(
                    season_type="REG", week=week, posteam=team,
                    passer_player_name=leader, qb_dropback=30,
                ))
    return pd.DataFrame(rows, columns=["season_type", "week", "posteam", "passer_player_name", "qb_dropback"])


def make_injuries():
    return pd.DataFrame(columns=["team", "week", "position", "full_name", "report_status", "gsis_id"])


def make_data():
    return {
        "schedule": make_schedule(),
        "team_stats": make_team_stats(),
        "pbp": make_pbp(),
        "injuries": make_injuries(),
    }


def test_pipeline_ranks_weak_opponent_first(cfg):
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())
    ranked = result["ranked"]
    assert ranked.iloc[0]["defense"] == "BUF"
    assert ranked.iloc[0]["opponent"] == "DEN"
    assert ranked.iloc[0]["opp_qb_status"] == "backup"
    assert ranked.iloc[0]["opp_turnovers_pg"] == ranked["opp_turnovers_pg"].max()
    assert ranked.iloc[0]["score"] > ranked.iloc[-1]["score"]


def test_played_games_excluded_and_byes_listed(cfg):
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())
    ranked_teams = set(result["ranked"]["defense"])
    assert "CHI" not in ranked_teams and "GB" not in ranked_teams
    assert {"CHI", "GB"} <= set(result["played"]["team"])
    assert "BAL" in result["byes"]
    assert "CHI" not in result["byes"] and "BUF" not in result["byes"]


def test_teams_filter_normalizes(cfg):
    result = pipeline.run(week=5, teams=["lar", "buf"], cfg=cfg, data=make_data())
    assert set(result["ranked"]["defense"]) == {"LA", "BUF"}


def test_team_without_qb_info_lists_qb_status_missing(cfg):
    # KC has no named QB and no dropback history, so its QB status is unknown.
    # LA faces KC, so LA's qb_status signal is missing.
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())
    row = result["ranked"].set_index("defense").loc["LA"]
    assert row["opp_qb_status"] == "unknown"
    assert "qb_status" in row["missing"]


def test_contributions_cover_every_weight(cfg):
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())
    contrib_names = {c[len("c_"):] for c in result["ranked"].columns if c.startswith("c_")}
    assert contrib_names == set(cfg["weights"]) == SIGNALS


def test_optional_inputs_missing_do_not_crash(cfg):
    schedule = make_schedule().drop(columns=["home_rest", "away_rest", "roof"])
    data = {"schedule": schedule, "team_stats": make_team_stats()}
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=data)
    ranked = result["ranked"]
    assert len(ranked) == 8
    assert ranked["c_rest"].eq(0.0).all()
    assert ranked["c_weather"].eq(0.0).all()
    assert ranked["opp_qb_status"].isna().all()
    assert ranked["c_ol_injuries"].eq(0.0).all()
    assert all("ol_injuries" in m for m in ranked["missing"])


def test_weather_note_roof_only_without_forecast(cfg):
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())
    notes = result["ranked"].set_index("defense")["notes"]
    assert notes["BUF"] == "weather: roof only"
    assert notes["MIA"] == ""


def test_forecast_clears_roof_only_note(cfg):
    data = make_data()
    data["forecast"] = {
        "BUF": {"temp": 30.0, "wind": 5.0},
        "DEN": {"temp": 30.0, "wind": 5.0},
    }
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=data)
    ranked = result["ranked"].set_index("defense")
    assert ranked.loc["BUF", "notes"] == ""
    assert "weather" not in ranked.loc["BUF", "missing"]


def test_nan_scores_sort_last(cfg):
    data = make_data()
    extra = _game(5, "ARI", "ATL", total=46.0, spread=0.0)
    data["schedule"] = pd.concat([data["schedule"], pd.DataFrame([extra])], ignore_index=True)
    only_turnovers = dict(cfg)
    only_turnovers["weights"] = {name: (1.0 if name == "turnovers" else 0.0) for name in cfg["weights"]}
    ranked = pipeline.run(week=5, teams=None, cfg=only_turnovers, data=data)["ranked"]
    assert ranked["score"].iloc[-2:].isna().all()
    assert set(ranked["defense"].iloc[-2:]) == {"ARI", "ATL"}
    assert ranked["score"].iloc[:-2].notna().all()


def test_teams_filter_keeps_played_game_for_away_side(cfg):
    # CHI hosted GB in week 5 and the game is final. GB is the away side.
    result = pipeline.run(week=5, teams=["GB"], cfg=cfg, data=make_data())
    played = result["played"]
    assert set(played["team"]) == {"CHI", "GB"}
    assert set(played["opponent"]) == {"CHI", "GB"}
