from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dstmatchup import pipeline
from dstmatchup.config import load_config

REPO_CONFIG = str(Path(__file__).resolve().parent.parent / "config.toml")
SIGNALS = {
    "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
    "opp_ppg", "own_defense", "home_away", "rest",
}

# (home, away) pairings. Weeks 1-4 are played; week 5 is the week under test.
PAIRS = [("BUF", "DEN"), ("KC", "LA"), ("MIA", "NE"), ("NYJ", "PIT")]
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
        home_rest=7, away_rest=7,
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
                    sack_fumbles_lost=0, rushing_fumbles_lost=0, receiving_fumbles_lost=0,
                    def_sacks=2, def_interceptions=0, fumble_recovery_opp=1, def_tds=0,
                    def_safeties=0, def_fg_blocks=0, def_pat_blocks=0, def_punt_blocks=0,
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
    schedule = make_schedule().drop(columns=["home_rest", "away_rest"])
    data = {"schedule": schedule, "team_stats": make_team_stats()}
    result = pipeline.run(week=5, teams=None, cfg=cfg, data=data)
    ranked = result["ranked"]
    assert len(ranked) == 8
    assert ranked["c_rest"].eq(0.0).all()
    assert ranked["opp_qb_status"].isna().all()


def test_signal_names_are_the_nine_remaining_signals():
    assert tuple(pipeline.SIGNAL_NAMES) == (
        "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
        "opp_ppg", "own_defense", "home_away", "rest",
    )


def test_ranked_table_has_no_notes_or_weather_or_ol_columns(cfg):
    ranked = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())["ranked"]
    assert "notes" not in ranked.columns
    assert not [c for c in ranked.columns if "weather" in c or "ol_" in c or "roof" in c]


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


def test_zscores_pool_whole_week_so_played_games_do_not_move_unplayed_teams(cfg):
    # Same week, but BUF-DEN and KC-LA are final. z is over the whole week, so the
    # teams still unplayed (MIA, NE, NYJ, PIT) must get exactly the same signals.
    all_unplayed = make_data()
    some_played = make_data()
    sched = some_played["schedule"]
    final = (sched["week"] == 5) & sched["home_team"].isin(["BUF", "KC"])
    sched.loc[final, ["home_score", "away_score", "result"]] = [20.0, 17.0, 3.0]

    base = pipeline.run(week=5, teams=None, cfg=cfg, data=all_unplayed)["ranked"].set_index("defense")
    mixed_result = pipeline.run(week=5, teams=None, cfg=cfg, data=some_played)
    mixed = mixed_result["ranked"].set_index("defense")

    assert set(mixed.index) == {"MIA", "NE", "NYJ", "PIT"}
    assert {"BUF", "DEN", "KC", "LA"}.isdisjoint(mixed.index)  # played rows stay out of ranked
    assert {"BUF", "DEN", "KC", "LA"} <= set(mixed_result["played"]["team"])
    columns = [c for c in base.columns if c.startswith("c_")] + ["score"]
    for team in mixed.index:
        for col in columns:
            assert mixed.loc[team, col] == pytest.approx(base.loc[team, col]), (team, col)


def _only(cfg, name):
    return {**cfg, "weights": {n: (1.0 if n == name else 0.0) for n in cfg["weights"]}}


def test_only_turnovers_matches_hand_calculation(cfg):
    # Week 5, turnovers the only weight. CHI and GB have no team stats, so the pool is
    # the 8 teams in PAIRS. DEN has 2 interceptions a game in weeks 1-4, everyone else 0.
    # BUF faces DEN: raw 2.0; mean 0.25; population var 0.4375, std 0.661438.
    # BUF z = 1.75 / 0.661438 = 2.645751. DEN has 4 games, shrink 4/(4+4) = 0.5,
    # c_turnovers = 1.322876, score = 50 + 25 * 1.322876 = 83.0719.
    # Other defenses: z = -0.25 / 0.661438 = -0.377964, shrunk -0.188982, score 45.2754.
    ranked = pipeline.run(week=5, teams=None, cfg=_only(cfg, "turnovers"), data=make_data())
    rows = ranked["ranked"].set_index("defense")
    assert rows.loc["BUF", "c_turnovers"] == pytest.approx(1.322876, abs=1e-6)
    assert rows.loc["BUF", "score"] == pytest.approx(83.0719, abs=1e-4)
    others = [t for t in rows.index if t != "BUF"]
    assert len(others) == 7
    for team in others:
        assert rows.loc[team, "c_turnovers"] == pytest.approx(-0.188982, abs=1e-6), team
        assert rows.loc[team, "score"] == pytest.approx(45.2754, abs=1e-4), team


def test_only_implied_total_matches_hand_calculation(cfg):
    # Week 5, implied_total the only weight. The defense signal is minus the opponent's
    # implied total. Pool is all 10 teams playing week 5 (CHI and GB are final but still
    # in the pool, both at 22.5): BUF -18.5, DEN -25.5, KC -25, LA -25, MIA -20, NE -20,
    # NYJ -21.5, PIT -24.5, CHI -22.5, GB -22.5. Mean -22.5, population var 5.5, std 2.345208.
    # BUF: dev +4 -> z 1.705606 -> score 92.6401. MIA and NE: dev +2.5 -> z 1.066004 -> 76.6501.
    # DEN: dev -3 -> z -1.279204 -> 18.0199. KC: 23.3499. NYJ: 60.6600. PIT: 28.6799.
    # (Before the whole-week z-pool fix the pool was 8 teams and BUF was 88.1385.)
    rows = pipeline.run(week=5, teams=None, cfg=_only(cfg, "implied_total"), data=make_data())["ranked"]
    rows = rows.set_index("defense")
    assert rows.loc["BUF", "c_implied_total"] == pytest.approx(1.705606, abs=1e-6)
    assert rows.loc["BUF", "score"] == pytest.approx(92.6401, abs=1e-3)
    assert rows.loc["MIA", "score"] == pytest.approx(76.6501, abs=1e-3)
    assert rows.loc["NE", "score"] == pytest.approx(76.6501, abs=1e-3)
    assert rows.loc["DEN", "score"] == pytest.approx(18.0199, abs=1e-3)
    assert rows.loc["KC", "score"] == pytest.approx(23.3499, abs=1e-3)
    assert rows.loc["NYJ", "score"] == pytest.approx(60.6600, abs=1e-3)
    assert rows.loc["PIT", "score"] == pytest.approx(28.6799, abs=1e-3)


def test_contributions_sum_to_score_minus_50_over_25(cfg):
    # Holds row by row even where a signal is missing (missing contributions are 0),
    # because the rescaled weights of the present signals sum to 1.
    ranked = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())["ranked"]
    contrib = [f"c_{name}" for name in cfg["weights"]]
    assert ranked["missing"].map(len).gt(0).any()  # make_data has missing signals on purpose
    for _, row in ranked.iterrows():
        assert sum(row[c] for c in contrib) == pytest.approx((row["score"] - 50) / 25, abs=1e-9), row["defense"]


def test_sack_rate_orientation_offense_that_gives_up_sacks_helps_defense(cfg):
    # DEN allows 10 sacks a game in weeks 1-4 (rate 10/40 = 0.25 vs 2/32 = 0.0625 for others).
    # BUF faces DEN, so BUF's sack_rate signal must be the highest and positive.
    data = make_data()
    stats = data["team_stats"]
    stats.loc[stats["team"] == "DEN", "sacks_suffered"] = 10
    rows = pipeline.run(week=5, teams=None, cfg=_only(cfg, "sack_rate"), data=data)["ranked"]
    rows = rows.set_index("defense")
    assert rows["c_sack_rate"].idxmax() == "BUF"
    assert rows.loc["BUF", "c_sack_rate"] > 0
    assert rows.loc["BUF", "opp_sack_rate"] == pytest.approx(0.25)


def test_rank_is_one_to_n_in_descending_score_order(cfg):
    ranked = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())["ranked"]
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["score"].is_monotonic_decreasing


def test_nan_score_rows_have_no_rank(cfg):
    data = make_data()
    extra = _game(5, "ARI", "ATL", total=46.0, spread=0.0)
    data["schedule"] = pd.concat([data["schedule"], pd.DataFrame([extra])], ignore_index=True)
    ranked = pipeline.run(week=5, teams=None, cfg=_only(cfg, "turnovers"), data=data)["ranked"]
    unscored = ranked[ranked["score"].isna()]
    assert set(unscored["defense"]) == {"ARI", "ATL"}
    assert unscored["rank"].isna().all()
    assert ranked["rank"].dropna().tolist() == list(range(1, 9))
    assert str(ranked["rank"].dtype) == "Int64"


def test_teams_filter_keeps_league_wide_ranks(cfg):
    full = pipeline.run(week=5, teams=None, cfg=cfg, data=make_data())["ranked"].set_index("defense")
    kept = pipeline.run(week=5, teams=["BUF", "KC"], cfg=cfg, data=make_data())["ranked"].set_index("defense")
    assert set(kept.index) == {"BUF", "KC"}
    assert kept.loc["BUF", "rank"] == full.loc["BUF", "rank"] == 1
    assert kept.loc["KC", "rank"] == full.loc["KC", "rank"]
    assert kept.loc["KC", "rank"] != 2


def test_why_column_names_the_turnover_driver_by_sign(cfg):
    # Only turnovers weighted: BUF c_turnovers 1.322876 = 33.07 points (up), others -0.188982 = -4.72 (down).
    rows = pipeline.run(week=5, teams=None, cfg=_only(cfg, "turnovers"), data=make_data())["ranked"]
    rows = rows.set_index("defense")
    assert rows.loc["BUF", "why"] == "▲ turnovers"
    assert rows.loc["MIA", "c_turnovers"] * 25 == pytest.approx(-4.72, abs=0.01)
    assert rows.loc["MIA", "why"] == "▼ turnovers"
