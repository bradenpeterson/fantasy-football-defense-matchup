from pathlib import Path

import math

import numpy as np
import pandas as pd
import pytest

from dstmatchup.config import load_config
from dstmatchup.dst_points import (
    dst_points_allowed_to,
    dst_points_by_game,
    points_allowed_score,
)

REPO_CONFIG = str(Path(__file__).resolve().parent.parent / "config.toml")


@pytest.fixture
def yahoo():
    return load_config(REPO_CONFIG)["yahoo"]


def test_points_allowed_tiers(yahoo):
    tiers = yahoo["tiers"]
    assert points_allowed_score(0, tiers) == 10
    assert points_allowed_score(6, tiers) == 7
    assert points_allowed_score(7, tiers) == 4
    assert points_allowed_score(20, tiers) == 1
    assert points_allowed_score(27, tiers) == 0
    assert points_allowed_score(34, tiers) == -1
    assert points_allowed_score(35, tiers) == -4


def test_dst_points_by_game(yahoo):
    team_stats = pd.DataFrame(
        [
            {
                "team": "BUF",
                "week": 1,
                "opponent_team": "KC",
                "season_type": "REG",
                "def_sacks": 3,
                "def_interceptions": 1,
                "def_safeties": 0,
                "def_tds": 0,
                "def_fg_blocks": 0,
                "def_pat_blocks": 0,
                "def_punt_blocks": 0,
                "fumble_recovery_opp": 0,
                "fumble_recovery_tds": 0,
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "away_team": "KC",
                "home_team": "BUF",
                "away_score": 13,
                "home_score": 20,
                "week": 1,
                "game_type": "REG",
            }
        ]
    )
    out = dst_points_by_game(team_stats, schedule, yahoo)
    assert list(out.columns) == ["team", "week", "opponent", "dst_points"]
    assert len(out) == 1
    assert out.iloc[0]["team"] == "BUF"
    assert out.iloc[0]["opponent"] == "KC"
    assert out.iloc[0]["dst_points"] == 9


def test_dst_points_by_game_missing_required_column_raises(yahoo):
    # Every D/ST stat column is required; drop one and the error must name it.
    team_stats = pd.DataFrame([{
        "team": "BUF", "week": 1, "opponent_team": "KC", "season_type": "REG",
        "def_sacks": 3, "def_interceptions": 0, "fumble_recovery_opp": 0, "def_safeties": 0,
        "def_tds": 0, "def_pat_blocks": 0, "def_punt_blocks": 0,
    }])
    schedule = pd.DataFrame(
        [{"away_team": "KC", "home_team": "BUF", "away_score": 13, "home_score": 20, "week": 1}]
    )
    with pytest.raises(KeyError, match="def_fg_blocks"):
        dst_points_by_game(team_stats, schedule, yahoo)


def test_dst_points_nan_game_is_excluded_from_allowed_mean(yahoo):
    base = {"team": "BUF", "opponent_team": "KC", "season_type": "REG", "def_safeties": 0,
            "def_tds": 0, "def_fg_blocks": 0, "def_pat_blocks": 0, "def_punt_blocks": 0,
            "def_interceptions": 0, "fumble_recovery_opp": 0}
    team_stats = pd.DataFrame([
        {**base, "week": 1, "def_sacks": np.nan},  # missing stat, not zero sacks
        {**base, "week": 2, "def_sacks": 3},
    ])
    schedule = pd.DataFrame([
        {"away_team": "KC", "home_team": "BUF", "away_score": 13, "home_score": 20, "week": w, "game_type": "REG"}
        for w in (1, 2)
    ])
    by_game = dst_points_by_game(team_stats, schedule, yahoo).set_index("week")
    assert math.isnan(by_game.loc[1, "dst_points"])
    assert by_game.loc[2, "dst_points"] == 3 + 4  # 3 sacks + 13 points allowed (tier 4)
    allowed = dst_points_allowed_to(dst_points_by_game(team_stats, schedule, yahoo))
    row = allowed.set_index("offense").loc["KC"]
    assert row["dst_pts_allowed_per_game"] == pytest.approx(7.0)
    assert row["games"] == 1


def test_dst_points_by_game_skips_unscored_games(yahoo):
    team_stats = pd.DataFrame(
        [{"team": "BUF", "week": 2, "opponent_team": "MIA", "season_type": "REG", "def_sacks": 1}]
    )
    schedule = pd.DataFrame(
        [
            {
                "away_team": "MIA",
                "home_team": "BUF",
                "away_score": None,
                "home_score": None,
                "week": 2,
            }
        ]
    )
    out = dst_points_by_game(team_stats, schedule, yahoo)
    assert len(out) == 0


def test_allowed_to_averages_per_offense():
    df = pd.DataFrame(
        [
            {"team": "BUF", "week": 1, "opponent": "X", "dst_points": 9},
            {"team": "MIA", "week": 2, "opponent": "X", "dst_points": 3},
            {"team": "BUF", "week": 3, "opponent": "Y", "dst_points": 5},
        ]
    )
    out = dst_points_allowed_to(df).set_index("offense")
    assert out.loc["X", "games"] == 2
    assert out.loc["X", "dst_pts_allowed_per_game"] == pytest.approx(6.0)
    assert out.loc["Y", "games"] == 1
    assert "Z" not in out.index
