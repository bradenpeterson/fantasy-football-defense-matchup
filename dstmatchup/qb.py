"""Opponent QB status for one week.

A team's QB status comes from three sources: the schedule's named starter
(a hint only), the dropback leader from prior regular-season pbp, and the
injury report. Injury status overrides the name comparison. When neither a
named starter nor a dropback leader exists the status is `unknown` and the
signal is excluded (qb_signal returns None).
"""

import re

import numpy as np
import pandas as pd

from dstmatchup.teams import normalize_team

STATUSES = ("starter", "backup", "questionable", "out", "unknown")
_SIGNALS = {"backup": 1.0, "out": 1.0, "questionable": 0.5, "starter": 0.0, "unknown": None}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _name_tokens(name: str) -> list[str]:
    letters = re.sub(r"[^a-z\s]", " ", name.lower())
    return [t for t in letters.split() if t not in _SUFFIXES]


def names_match(a, b) -> bool:
    """True for equal normalized names, or same last name with matching first initial.

    Suffixes (Jr, Sr, II, III, IV) and punctuation are ignored, so "J.Milton"
    matches "Joe Milton III". Two full, different first names never match.
    """
    a, b = _clean(a), _clean(b)
    if a is None or b is None:
        return False
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    if len(ta) < 2 or len(tb) < 2 or ta[-1] != tb[-1]:
        return False
    first_a, first_b = ta[0], tb[0]
    return first_a[0] == first_b[0] and (len(first_a) == 1 or len(first_b) == 1)


def _injury_bucket(injury_status) -> str | None:
    text = _clean(injury_status)
    if text is None:
        return None
    text = text.lower()
    if text in ("out", "doubtful"):
        return "out"
    if text == "questionable":
        return "questionable"
    return None


def qb_status(named_starter, dropback_leader, injury_status) -> str:
    starter = _clean(named_starter)
    leader = _clean(dropback_leader)
    if starter is None and leader is None:
        return "unknown"
    bucket = _injury_bucket(injury_status)
    if bucket is not None:
        return bucket
    if starter is None or leader is None:
        return "unknown"
    return "starter" if names_match(starter, leader) else "backup"


def qb_signal(status: str) -> float | None:
    if status not in _SIGNALS:
        raise ValueError(f"unknown QB status {status!r}; valid: {', '.join(STATUSES)}")
    return _SIGNALS[status]


def _team_or_none(value):
    text = _clean(value)
    if text is None:
        return None
    try:
        return normalize_team(text)
    except ValueError:
        return None


def _dropback_leaders(pbp: pd.DataFrame, week: int) -> dict[str, str]:
    df = pbp
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["week"] < week].dropna(subset=["posteam", "passer_player_name"])
    dropbacks = pd.to_numeric(df["qb_dropback"], errors="coerce").fillna(0.0)
    totals = df.assign(_db=dropbacks.to_numpy()).groupby(["posteam", "passer_player_name"])["_db"].sum()
    leaders: dict[str, tuple[str, float]] = {}
    for (posteam, name), total in totals.items():
        team = _team_or_none(posteam)
        if team is None or total <= 0:
            continue
        if team not in leaders or total > leaders[team][1]:
            leaders[team] = (str(name), float(total))
    return {team: name for team, (name, _) in leaders.items()}


def _recent_leaders(pbp: pd.DataFrame, week: int, n_games: int) -> dict[str, list]:
    """Per team, the dropback leader of each of its last `n_games` REG games before `week`.

    Games are the team's own played games (byes are skipped), oldest first. A game
    with no dropbacks has a None leader. Teams with fewer than `n_games` prior games
    are omitted, so the recent-leader rule does not apply to them.
    """
    df = pbp
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["week"] < week].dropna(subset=["posteam"])
    teams = df["posteam"].map(_team_or_none)
    df = df.assign(_team=teams.to_numpy()).dropna(subset=["_team"])
    dropbacks = pd.to_numeric(df["qb_dropback"], errors="coerce").fillna(0.0)
    df = df.assign(_db=dropbacks.to_numpy())
    result: dict[str, list] = {}
    for team, group in df.groupby("_team"):
        # Game weeks come from every row, so a game with no named passer still
        # counts as a played game (with no leader).
        weeks = sorted(group["week"].unique())
        if len(weeks) < n_games:
            continue
        leaders = []
        for game_week in weeks[-n_games:]:
            game = group[group["week"] == game_week].dropna(subset=["passer_player_name"])
            totals = game.groupby("passer_player_name")["_db"].sum()
            totals = totals[totals > 0]
            leaders.append(str(totals.idxmax()) if len(totals) else None)
        result[str(team)] = leaders
    return result


def _led_every_recent_game(named: str, leaders: list | None) -> bool:
    if not leaders:
        return False
    return all(leader is not None and names_match(named, leader) for leader in leaders)


def _injury_for(injuries: pd.DataFrame, team: str, week: int, name: str | None):
    if name is None or injuries is None or injuries.empty:
        return None
    rows = injuries[
        (injuries["team"] == team) & (injuries["week"] == week) & (injuries["position"] == "QB")
    ]
    for row in rows.itertuples(index=False):
        if names_match(row.full_name, name) and _injury_bucket(row.report_status):
            return row.report_status
    return None


def qb_status_table(
    schedule: pd.DataFrame, pbp: pd.DataFrame, injuries: pd.DataFrame, week: int, recent_games: int = 2,
) -> pd.DataFrame:
    leaders = _dropback_leaders(pbp, week)
    recent = _recent_leaders(pbp, week, recent_games) if recent_games >= 1 else {}
    games = schedule[schedule["week"] == week]
    rows = []
    for game in games.itertuples(index=False):
        for team, named in (
            (_team_or_none(game.home_team), _clean(game.home_qb_name)),
            (_team_or_none(game.away_team), _clean(game.away_qb_name)),
        ):
            if team is None:
                continue
            leader = leaders.get(team)
            if named is not None and leader is not None and _led_every_recent_game(named, recent.get(team)):
                leader = named
            evaluated = named if named is not None else leader
            injury = _injury_for(injuries, team, week, evaluated)
            rows.append(dict(
                team=team,
                qb_name=evaluated,
                status=qb_status(named, leader, injury),
            ))
    return pd.DataFrame(rows, columns=["team", "qb_name", "status"])
