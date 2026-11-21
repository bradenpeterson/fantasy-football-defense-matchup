"""Thin cached wrappers around nflreadpy loaders.

Each loader returns a pandas DataFrame with team-code columns normalized to
nflverse codes. Results are cached as parquet via cached_frame.
"""

from typing import Callable

import nflreadpy
import pandas as pd

from dstmatchup.cache import cached_frame
from dstmatchup.teams import normalize_team

CURRENT_SEASON = 2026
TEAM_COLUMNS = ("team", "away_team", "home_team", "opponent_team")


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEAM_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: v if v is None or (isinstance(v, float) and v != v) else normalize_team(v)
            )
    return df


def _load(name: str, season: int, fetch: Callable[[int], object]) -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        return _normalize_teams(fetch(season).to_pandas())

    return cached_frame(name, season, _fetch, current_season=CURRENT_SEASON)


def load_schedule(season: int) -> pd.DataFrame:
    return _load("schedule", season, lambda s: nflreadpy.load_schedules([s]))


def load_team_stats(season: int) -> pd.DataFrame:
    return _load("team_stats", season, lambda s: nflreadpy.load_team_stats([s]))


def load_pbp(season: int) -> pd.DataFrame:
    return _load("pbp", season, lambda s: nflreadpy.load_pbp([s]))


def load_injuries(season: int) -> pd.DataFrame:
    return _load("injuries", season, lambda s: nflreadpy.load_injuries([s]))
