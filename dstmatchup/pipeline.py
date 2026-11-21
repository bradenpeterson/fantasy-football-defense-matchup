"""Week-level DST matchup ranking: data frames in, ranked frame out.

No network I/O. Callers (the CLI) load data and pass it in.
"""

import numpy as np
import pandas as pd

from dstmatchup import dst_points, schedule as sched, signals, scoring
from dstmatchup.teams import normalize_team
from dstmatchup.qb import qb_status_table
from dstmatchup.weather import roof_signal

SIGNAL_NAMES = (
    "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
    "opp_ppg", "own_defense", "home_away", "weather", "rest", "ol_injuries",
)
OPP_SHRUNK = ("turnovers", "sack_rate", "dst_pts_allowed", "opp_ppg")
OWN_SHRUNK = ("own_defense",)


def _shrink_column(z: pd.Series, n: pd.Series, k: float) -> pd.Series:
    return scoring.shrink(z, n, k)


def run(week, teams, cfg: dict, data: dict) -> dict:
    schedule = data["schedule"]
    if week is None:
        week = sched.current_week(schedule)

    matchups = sched.matchups(schedule, week)
    played_mask = matchups["played"].to_numpy(dtype=bool)
    lines = sched.get_lines(schedule, week)

    team_stats = data["team_stats"]
    rates = signals.team_rates(team_stats, week, last_n=cfg["scoring"]["last_n"], schedule=schedule)
    allowed = dst_points.dst_points_allowed_to(
        dst_points.dst_points_by_game(team_stats, schedule, cfg["yahoo"])
    )
    pbp = data.get("pbp")
    qb_table = None
    if pbp is not None:
        qb_table = qb_status_table(schedule, pbp, data.get("injuries"), week)

    # z-scores are over every team playing this week (played games included), so a
    # team's z does not depend on how many of the week's games are already final.
    # Inputs only use weeks before `week`, so the played rows leak nothing.
    raw = signals.build_signal_frame(
        matchups, rates, allowed, lines, cfg,
        qb_table=qb_table,
        injuries=data.get("injuries"),
        depth=data.get("depth"),
        forecast=data.get("forecast"),
        week=week,
    )

    k = float(cfg["scoring"]["shrink_k"])
    z = pd.DataFrame(index=raw.index)
    for name in SIGNAL_NAMES:
        z[name] = scoring.zscore(raw[name].astype(float))
    for name in OPP_SHRUNK:
        z[name] = _shrink_column(z[name], raw["opp_games"].astype(float), k)
    for name in OWN_SHRUNK:
        z[name] = _shrink_column(z[name], raw["own_games"].astype(float), k)

    scored = scoring.score_teams(z, cfg["weights"], scale=float(cfg["scoring"]["display_scale"]))

    everyone = pd.DataFrame({
        "defense": raw.index.to_numpy(),
        "opponent": matchups["opponent"].to_numpy(),
        "home": matchups["home"].to_numpy(),
        "opp_implied_total": -raw["implied_total"].to_numpy(dtype=float),
        "opp_turnovers_pg": raw["turnovers"].to_numpy(dtype=float),
        "opp_sack_rate": raw["sack_rate"].to_numpy(dtype=float),
        "opp_qb_status": _qb_status_strings(qb_table, matchups["opponent"]),
        "score": scored["score"].to_numpy(dtype=float),
        "missing": list(scored["missing"]),
        "notes": _notes(matchups, data.get("forecast")),
    })
    for name in SIGNAL_NAMES:
        everyone[f"c_{name}"] = scored[name].to_numpy(dtype=float)

    ranked = everyone[~played_mask]
    ranked = ranked.sort_values("score", ascending=False, na_position="last", kind="stable")
    ranked = ranked.reset_index(drop=True)

    played = matchups[matchups["played"]].reset_index(drop=True)
    byes = sched.bye_teams(schedule, week)
    if teams is not None:
        wanted = {normalize_team(t) for t in teams}
        ranked = ranked[ranked["defense"].isin(wanted)].reset_index(drop=True)
        keep = played["team"].isin(wanted) | played["opponent"].isin(wanted)
        played = played[keep].reset_index(drop=True)
        byes = [t for t in byes if t in wanted]

    return {
        "week": week,
        "ranked": ranked,
        "played": played,
        "byes": byes,
    }


def _qb_status_strings(qb_table, opponents: pd.Series) -> list:
    if qb_table is None:
        return [None] * len(opponents)
    statuses = qb_table.set_index("team")["status"]
    return [statuses.get(t) for t in opponents]


def _notes(unplayed: pd.DataFrame, forecast: dict | None) -> list[str]:
    notes = []
    for team, roof in zip(unplayed["team"], unplayed["roof"]):
        note = ""
        if roof_signal(roof) == 0.5 and not (forecast or {}).get(team):
            note = "weather: roof only"
        notes.append(note)
    return notes
