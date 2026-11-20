"""Command-line entry point: load data, run the pipeline, print the ranking."""

import argparse
import sys

from dstmatchup import pipeline
from dstmatchup import schedule as sched
from dstmatchup.config import load_config
from dstmatchup.loaders import (
    load_depth_charts,
    load_injuries,
    load_pbp,
    load_schedule,
    load_team_stats,
)
from dstmatchup.teams import normalize_team
from dstmatchup.weather import STADIUM_COORDS, fetch_forecast, roof_signal
from tabulate import tabulate

BASE_HEADERS = ["Defense", "Opp", "H/A", "Opp Implied", "Opp TO/g", "Opp Sack%", "Opp QB", "Score"]


def _parse_teams(text: str) -> list[str]:
    return [normalize_team(part) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dstmatchup", description="Rank NFL defenses for fantasy matchups.")
    parser.add_argument("--week", type=int, default=None, help="week to rank (default: current week)")
    parser.add_argument("--teams", type=str, default=None, help="comma-separated defense codes to show")
    parser.add_argument("--season", type=int, default=2026, help="season year (default: 2026)")
    parser.add_argument("--breakdown", action="store_true", help="show per-signal contribution columns")
    return parser


def _load_data(season: int) -> dict:
    return {
        "schedule": load_schedule(season),
        "team_stats": load_team_stats(season),
        "pbp": load_pbp(season),
        "injuries": load_injuries(season),
        "depth": load_depth_charts(season),
    }


def _forecasts(schedule, week: int) -> dict:
    """Forecast for each outdoor unplayed game, keyed by both teams in the game.

    The stadium is the home team's. A failed fetch leaves that game out.
    """
    forecast: dict = {}
    games = schedule[(schedule["week"] == week) & (schedule["result"].isna())]
    for g in games.itertuples(index=False):
        if roof_signal(getattr(g, "roof", None)) != 0.5:
            continue
        coords = STADIUM_COORDS.get(g.home_team)
        if coords is None:
            continue
        kickoff = f"{g.gameday} {g.gametime}"
        try:
            result = fetch_forecast(coords[0], coords[1], kickoff)
        except Exception:
            result = None
        if result is not None:
            forecast[g.home_team] = result
            forecast[g.away_team] = result
    return forecast


def _num(value, pattern: str) -> str:
    if value is None or value != value:
        return "n/a"
    return pattern.format(value)


def _text(value) -> str:
    return "n/a" if value is None or value != value else str(value)


def format_result(result: dict, breakdown: bool = False) -> str:
    ranked = result["ranked"]
    contrib_cols = [c for c in ranked.columns if c.startswith("c_")]
    headers = list(BASE_HEADERS)
    if breakdown:
        headers += [c[len("c_"):] for c in contrib_cols]

    rows = []
    for row in ranked.itertuples(index=False):
        under = []
        if row.missing:
            under.append(f"  missing: {', '.join(row.missing)}")
        if row.notes:
            under.append(f"  notes: {row.notes}")
        defense = "\n".join([row.defense] + under)
        cells = [
            defense,
            row.opponent,
            "H" if row.home else "A",
            _num(row.opp_implied_total, "{:.1f}"),
            _num(row.opp_turnovers_pg, "{:.1f}"),
            _num(row.opp_sack_rate, "{:.1%}"),
            _text(row.opp_qb_status),
            _num(row.score, "{:.1f}"),
        ]
        if breakdown:
            cells += [_num(getattr(row, c), "{:+.2f}") for c in contrib_cols]
        rows.append(cells)

    lines = [f"Week {result['week']} DST matchup ranking", ""]
    lines.append(tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True))

    lines.append("")
    lines.append("Already played this week:")
    played = result["played"]
    home_games = played[played["home"]] if len(played) else played
    if len(home_games):
        for g in home_games.itertuples(index=False):
            lines.append(f"  {g.opponent} @ {g.team}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("On bye:")
    lines.append(f"  {', '.join(result['byes']) if result['byes'] else 'none'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    teams = None
    try:
        if args.teams is not None:
            teams = _parse_teams(args.teams)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cfg = load_config()
    if teams is None and cfg["defaults"]["available_teams"]:
        teams = [normalize_team(t) for t in cfg["defaults"]["available_teams"]]

    data = _load_data(args.season)
    week = args.week if args.week is not None else sched.current_week(data["schedule"])
    data["forecast"] = _forecasts(data["schedule"], week)

    result = pipeline.run(week, teams, cfg, data)
    print(format_result(result, breakdown=args.breakdown))
    return 0
