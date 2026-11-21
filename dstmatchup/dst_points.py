"""Yahoo D/ST points proxy built from team stats and schedule scores.

No network I/O: callers pass in frames (see dstmatchup.loaders).
A missing stat column raises KeyError; a NaN stat stays NaN (that game's
points are NaN and drop out of averages). Only regular-season games with
scores contribute.
"""

import pandas as pd

KEY_COLS = ["team", "week", "opponent_team"]
OUT_COLS = ["team", "week", "opponent", "dst_points"]


def points_allowed_score(points: int, tiers: list) -> int:
    """Map points allowed to a Yahoo points value using inclusive min/max tiers."""
    for tier in tiers:
        if tier["min"] <= points <= tier["max"]:
            return int(tier["points"])
    raise ValueError(f"no points-allowed tier covers {points}")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric stat column. A missing column raises; a NaN cell stays NaN."""
    if name not in df.columns:
        raise KeyError(f"team_stats is missing required D/ST column {name!r}")
    return pd.to_numeric(df[name], errors="coerce")


def _regular_season(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        return df[df[col] == "REG"]
    return df


def _points_allowed_frame(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, week, opponent) with the points `opponent` scored on `team`."""
    games = _regular_season(schedule, "game_type")
    home = pd.DataFrame(
        {
            "team": games["home_team"],
            "week": games["week"],
            "opponent_team": games["away_team"],
            "points_allowed": games["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team"],
            "week": games["week"],
            "opponent_team": games["home_team"],
            "points_allowed": games["home_score"],
        }
    )
    both = pd.concat([home, away], ignore_index=True)
    return both.dropna(subset=["points_allowed"])


def dst_points_by_game(
    team_stats: pd.DataFrame, schedule: pd.DataFrame, yahoo: dict
) -> pd.DataFrame:
    """Yahoo D/ST points for each team in each regular-season game with a score.

    Fumble recoveries use fumble_recovery_opp. Blocked kicks are fg + pat + punt
    blocks. Only def_tds is counted as TDs; fumble_recovery_tds is not added
    separately to avoid double counting.
    """
    stats = _regular_season(team_stats, "season_type").copy()
    allowed = _points_allowed_frame(schedule)
    merged = stats.merge(
        allowed,
        on=KEY_COLS,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=OUT_COLS)

    blocked = _col(merged, "def_fg_blocks") + _col(merged, "def_pat_blocks") + _col(
        merged, "def_punt_blocks"
    )
    tier_points = merged["points_allowed"].astype(int).map(
        lambda p: points_allowed_score(p, yahoo["tiers"])
    )
    points = (
        yahoo["sack"] * _col(merged, "def_sacks")
        + yahoo["interception"] * _col(merged, "def_interceptions")
        + yahoo["fumble_recovery"] * _col(merged, "fumble_recovery_opp")
        + yahoo["safety"] * _col(merged, "def_safeties")
        + yahoo["td"] * _col(merged, "def_tds")
        + yahoo["blocked_kick"] * blocked
        + tier_points
    )
    return pd.DataFrame(
        {
            "team": merged["team"].values,
            "week": merged["week"].values,
            "opponent": merged["opponent_team"].values,
            "dst_points": points.astype(float).values,
        }
    )


def dst_points_allowed_to(df: pd.DataFrame) -> pd.DataFrame:
    """Average D/ST points each offense has given up to opposing defenses.

    Input is the output of dst_points_by_game. Offenses with no games are absent.
    """
    grouped = df.groupby("opponent").agg(
        games=("dst_points", "count"),
        dst_pts_allowed_per_game=("dst_points", "mean"),
    )
    grouped = grouped.reset_index().rename(columns={"opponent": "offense"})
    return grouped[["offense", "games", "dst_pts_allowed_per_game"]]
