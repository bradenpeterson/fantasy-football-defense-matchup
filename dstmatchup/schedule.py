"""Schedule-derived views: current week, betting lines, matchups, byes."""

import numpy as np
import pandas as pd

from dstmatchup.teams import ALL_TEAMS


def _played(schedule: pd.DataFrame) -> pd.Series:
    return schedule["result"].notna()


def current_week(schedule: pd.DataFrame) -> int:
    """Earliest week with an unplayed game; otherwise the next week, or the last week."""
    unplayed = schedule.loc[~_played(schedule), "week"]
    if not unplayed.empty:
        return int(unplayed.min())
    last = int(schedule["week"].max())
    if (schedule["week"] == last + 1).any():
        return last + 1
    return last


def _week_games(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    return schedule[schedule["week"] == week]


def get_lines(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    """One row per team playing in `week`.

    spread_line is positive when the home team is favored, so the home team's
    spread_for_team is +spread_line and the away team's is -spread_line.
    """
    rows = []
    for g in _week_games(schedule, week).itertuples(index=False):
        total = float(g.total_line)
        spread = float(g.spread_line)
        home_implied = (total + spread) / 2.0
        away_implied = (total - spread) / 2.0
        rows.append(dict(
            team=g.home_team, opponent=g.away_team, home=True,
            implied_total=home_implied, opp_implied_total=away_implied,
            total=total, spread_for_team=spread,
        ))
        rows.append(dict(
            team=g.away_team, opponent=g.home_team, home=False,
            implied_total=away_implied, opp_implied_total=home_implied,
            total=total, spread_for_team=-spread,
        ))
    return pd.DataFrame(rows, columns=[
        "team", "opponent", "home", "implied_total", "opp_implied_total", "total", "spread_for_team",
    ])


def matchups(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    """One row per team playing in `week`, with rest, venue and played status."""
    rows = []
    for g in _week_games(schedule, week).itertuples(index=False):
        played = pd.notna(g.result)
        home_rest = getattr(g, "home_rest", np.nan)
        away_rest = getattr(g, "away_rest", np.nan)
        for team, opp, home, rest, opp_rest in (
            (g.home_team, g.away_team, True, home_rest, away_rest),
            (g.away_team, g.home_team, False, away_rest, home_rest),
        ):
            rows.append(dict(
                team=team, opponent=opp, home=home, gameday=g.gameday,
                played=bool(played), rest_days=rest, opp_rest_days=opp_rest,
            ))
    return pd.DataFrame(rows, columns=[
        "team", "opponent", "home", "gameday", "played", "rest_days", "opp_rest_days",
    ])


def bye_teams(schedule: pd.DataFrame, week: int) -> list[str]:
    """Teams from ALL_TEAMS with no game in `week`."""
    playing = set(_week_games(schedule, week)[["home_team", "away_team"]].to_numpy().ravel())
    return [t for t in ALL_TEAMS if t not in playing]
