import numpy as np
import pandas as pd
import pytest

from dstmatchup import backtest, dst_points
from dstmatchup.config import load_config
from dstmatchup.pipeline import SIGNAL_NAMES
from test_pipeline import REPO_CONFIG, make_data

SIGNALS = (
    "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
    "opp_ppg", "own_defense", "home_away", "weather", "rest", "ol_injuries",
)


def _frame(signal_values: dict, actual: list) -> pd.DataFrame:
    df = pd.DataFrame({f"c_{name}": v for name, v in signal_values.items()})
    df["actual_dst_points"] = actual
    df["score"] = actual
    return df


def test_signal_correlations_known_values():
    actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _frame(
        {
            "implied_total": [10.0, 20.0, 30.0, 40.0, 50.0],
            "turnovers": [5.0, 4.0, 3.0, 2.0, 1.0],
            "sack_rate": [np.nan] * 5,
        },
        actual,
    )
    corr = backtest.signal_correlations(df)
    assert corr["implied_total"] == pytest.approx(1.0)
    assert corr["turnovers"] == pytest.approx(-1.0)
    assert np.isnan(corr["sack_rate"])


def test_suggest_weights_sum_and_nonnegative():
    corr = pd.Series({name: 0.0 for name in SIGNALS})
    corr["implied_total"] = 0.4
    corr["turnovers"] = 0.2
    corr["sack_rate"] = -0.3
    corr["score"] = 0.5
    weights = backtest.suggest_weights(corr)
    assert set(weights) == set(SIGNALS)
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(100.0)
    assert weights["sack_rate"] == 0.0
    assert weights["implied_total"] == pytest.approx(100 * 0.4 / 0.6)


def test_suggest_weights_all_nonpositive_returns_current_weights():
    corr = pd.Series({name: -0.1 for name in SIGNALS})
    current = {name: float(i + 1) for i, name in enumerate(SIGNALS)}
    assert backtest.suggest_weights(corr, current=current) == current


@pytest.fixture
def cfg():
    return load_config(REPO_CONFIG)


def test_actual_points_are_the_defenses_own_points(cfg):
    data = make_data()
    stats = data["team_stats"]
    stats.loc[(stats["team"] == "BUF") & (stats["week"] == 4), "def_sacks"] = 10
    out = backtest.run([2026], cfg, min_week=4, data={2026: data})

    expected = dst_points.dst_points_by_game(data["team_stats"], data["schedule"], cfg["yahoo"])
    own = expected[(expected["team"] == "BUF") & (expected["week"] == 4)]["dst_points"].iloc[0]
    other = expected[(expected["team"] == "DEN") & (expected["week"] == 4)]["dst_points"].iloc[0]
    assert own != other

    row = out[(out["team"] == "BUF") & (out["week"] == 4)]
    assert len(row) == 1
    assert row["actual_dst_points"].iloc[0] == pytest.approx(own)


def _poison_future(data: dict) -> dict:
    """Make every week-4+ input absurd: stats, scores, and a fake dropback leader."""
    poisoned = {k: v.copy() for k, v in data.items()}
    stats = poisoned["team_stats"]
    numeric = [c for c in stats.columns if c not in ("team", "opponent_team", "week", "season_type")]
    stats.loc[stats["week"] >= 4, numeric] = 999.0
    sched = poisoned["schedule"]
    sched.loc[sched["week"] >= 4, ["home_score", "away_score"]] = 999.0
    extra = pd.DataFrame([
        dict(season_type="REG", week=w, posteam=team, passer_player_name="Poison Passer", qb_dropback=9999)
        for w in (4, 5) for team in ("BUF", "DEN", "KC", "LA")
    ])
    poisoned["pbp"] = pd.concat([poisoned["pbp"], extra], ignore_index=True)
    return poisoned


def _with_week5_results(data: dict) -> dict:
    """Add week-5 team stats for CHI and GB, the one final week-5 game in the fixture."""
    stats = data["team_stats"]
    extra = pd.DataFrame([
        dict(team=team, opponent_team=opp, week=5, season_type="REG", sacks_suffered=2, attempts=30,
             passing_interceptions=0, def_sacks=2, def_interceptions=0, fumble_recovery_opp=1, def_tds=0)
        for team, opp in (("CHI", "GB"), ("GB", "CHI"))
    ])
    data["team_stats"] = pd.concat([stats, extra], ignore_index=True)
    return data


def test_no_lookahead_future_poison_does_not_change_scores(cfg):
    clean = backtest.run([2026], cfg, min_week=4, data={2026: _with_week5_results(make_data())})
    poisoned = backtest.run([2026], cfg, min_week=4, data={2026: _poison_future(_with_week5_results(make_data()))})
    assert set(clean["week"]) == {4, 5}
    key = ["week", "team"]
    merged = clean.merge(poisoned, on=key, suffixes=("_clean", "_poison"))
    assert len(merged) == len(clean) == len(poisoned)
    pd.testing.assert_series_equal(
        merged["score_clean"], merged["score_poison"], check_names=False
    )
    for name in SIGNAL_NAMES:
        pd.testing.assert_series_equal(
            merged[f"c_{name}_clean"], merged[f"c_{name}_poison"], check_names=False
        )
