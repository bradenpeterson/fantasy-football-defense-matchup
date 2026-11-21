"""Walk-forward backtest of the DST ranking against realized D/ST points.

Limitations (by design):
- Only week W's ranking is computed from weeks < W. Schedule results, scores and
  team stats from week W onward are hidden from pipeline.run.
- Depth charts and injury reports are single snapshots, so they would leak the
  future. ol_injuries is therefore always NaN (excluded) in the backtest, and
  QB status uses only the schedule's named starter versus the prior-weeks
  dropback leader (no injury report).
- The weather signal is roof-only; no forecast is fetched for past games.
- qb_status uses the schedule's actual starter for the week (hindsight): for past
  games the schedule names who really started, so its correlation and suggested
  weight are optimistic.
- Samples are small (a few hundred defense-weeks at most), so correlations are
  noisy and the suggested weights are not evidence of predictive power.
"""

import argparse
import math

import pandas as pd
from tabulate import tabulate

from dstmatchup import dst_points, pipeline
from dstmatchup.config import load_config

SIGNAL_NAMES = pipeline.SIGNAL_NAMES
ROW_COLS = ["season", "week", "team", "score", "actual_dst_points"]
_RESULT_COLS = ("result", "home_score", "away_score")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    # Spearman is Pearson on average ranks; scipy is not a dependency.
    rx = x[mask].rank(method="average")
    ry = y[mask].rank(method="average")
    return float(rx.corr(ry))


def signal_correlations(df: pd.DataFrame) -> pd.Series:
    """Spearman correlation of each signal's contribution with actual D/ST points.

    Indexed by the 11 signal names, then `score`. Uses rows where both values
    are non-NaN. NaN when there are fewer than two usable rows or no variation.
    """
    actual = pd.to_numeric(df["actual_dst_points"], errors="coerce")
    values = {}
    for name in SIGNAL_NAMES:
        col = f"c_{name}"
        values[name] = _spearman(df[col], actual) if col in df.columns else float("nan")
    values["score"] = _spearman(df["score"], actual)
    return pd.Series(values, dtype=float)


def suggest_weights(corr: pd.Series, current: dict | None = None) -> dict[str, float]:
    """Weights proportional to positive correlations, summing to 100.

    Only the 11 signal names are used. Negative and NaN correlations get weight 0.
    If nothing is positive, returns `current` unchanged (all zeros when no
    current weights are given). Never writes config.toml.
    """
    clipped = {}
    for name in SIGNAL_NAMES:
        value = corr.get(name, float("nan"))
        clipped[name] = float(value) if pd.notna(value) and value > 0 else 0.0
    total = sum(clipped.values())
    if total == 0:
        return dict(current) if current is not None else {name: 0.0 for name in SIGNAL_NAMES}
    return {name: 100.0 * value / total for name, value in clipped.items()}


def _empty_injuries() -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "week", "position", "full_name", "report_status", "gsis_id"])


def _season_data(season: int) -> dict:
    from dstmatchup.loaders import load_pbp, load_schedule, load_team_stats

    return {
        "schedule": load_schedule(season),
        "team_stats": load_team_stats(season),
        "pbp": load_pbp(season),
    }


def _hide_from_week(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    """Copy of `schedule` with results and scores of `week` onward set to NaN."""
    hidden = schedule.copy()
    mask = hidden["week"] >= week
    for col in _RESULT_COLS:
        if col in hidden.columns:
            hidden.loc[mask, col] = float("nan")
    return hidden


def _weeks_with_scores(schedule: pd.DataFrame, min_week: int, max_week: int | None) -> list[int]:
    """Regular-season weeks in range that have at least one final score."""
    games = schedule
    if "game_type" in games.columns:
        games = games[games["game_type"] == "REG"]
    weeks = sorted({int(w) for w in games.loc[games["result"].notna(), "week"]})
    return [w for w in weeks if w >= min_week and (max_week is None or w <= max_week)]


def _season_rows(season: int, cfg: dict, min_week: int, max_week: int | None, season_data: dict) -> pd.DataFrame:
    schedule = season_data["schedule"]
    team_stats = season_data["team_stats"]
    pbp = season_data.get("pbp")

    actual = dst_points.dst_points_by_game(team_stats, schedule, cfg["yahoo"])
    actual = actual[["team", "week", "dst_points"]].rename(columns={"dst_points": "actual_dst_points"})
    actual["week"] = actual["week"].astype(int)

    frames = []
    for week in _weeks_with_scores(schedule, min_week, max_week):
        week_data = {
            "schedule": _hide_from_week(schedule, week),
            "team_stats": team_stats,
            "pbp": pbp,
            "injuries": _empty_injuries(),
        }
        ranked = pipeline.run(week, None, cfg, week_data)["ranked"]
        frame = ranked.rename(columns={"defense": "team"})
        # score_teams contributes 0.0 for a missing signal; here it becomes NaN so
        # missing data is excluded from correlations instead of counted as average.
        for name in SIGNAL_NAMES:
            is_missing = ranked["missing"].apply(lambda m, n=name: n in m)
            frame[f"c_{name}"] = frame[f"c_{name}"].where(~is_missing)
        frame = frame.drop(columns=["missing"])
        frame["week"] = week
        frame["season"] = season
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=ROW_COLS)

    scored = pd.concat(frames, ignore_index=True)
    return scored.merge(actual, on=["team", "week"], how="inner")


def run(
    seasons: list[int],
    cfg: dict,
    min_week: int = 4,
    max_week: int | None = None,
    data: dict | None = None,
) -> pd.DataFrame:
    """One row per defense-week: pipeline score from prior weeks plus actual D/ST points.

    `data` maps season -> {schedule, team_stats, pbp}; when None, real data is
    loaded per season. Defense-weeks with no actual outcome (byes, unplayed)
    are dropped. Rows with a NaN score are kept.
    """
    frames = []
    for season in seasons:
        season_data = data[season] if data is not None else _season_data(season)
        frames.append(_season_rows(season, cfg, min_week, max_week, season_data))
    if not frames:
        return pd.DataFrame(columns=ROW_COLS)
    out = pd.concat(frames, ignore_index=True)
    contrib = [f"c_{name}" for name in SIGNAL_NAMES]
    return out[ROW_COLS + contrib]


DEFAULT_SEASONS = [2025, 2026]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dstmatchup backtest", description="Backtest DST signals against actual points.")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS, help="seasons (default: 2025 2026)")
    parser.add_argument("--min-week", type=int, default=4, help="first week to score (default: 4)")
    parser.add_argument("--max-week", type=int, default=None, help="last week to score (default: all)")
    return parser


def _fmt(value, pattern: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return pattern.format(value)


def format_report(df: pd.DataFrame, cfg: dict, seasons: list[int]) -> str:
    corr = signal_correlations(df)
    current = {name: float(cfg["weights"][name]) for name in SIGNAL_NAMES}
    suggested = suggest_weights(corr, current=current)
    scored = int(df["score"].notna().sum()) if len(df) else 0
    lines = [
        f"Backtest seasons {', '.join(str(s) for s in seasons)} "
        "(each week scored only from earlier weeks)",
        f"Sample size: {len(df)} defense-weeks ({scored} with a score)",
        "Caveat: correlations are on a small sample and are noisy. ol_injuries is excluded by design "
        "(no historical depth or injury snapshots), so its correlation is n/a.",
        "qb_status uses the schedule's actual starter for the week (hindsight), so its correlation and "
        "suggested weight are optimistic.",
        "",
        "Spearman correlation with actual D/ST points:",
    ]
    for name in list(SIGNAL_NAMES) + ["score"]:
        label = f"{name} (hindsight)" if name == "qb_status" else name
        lines.append(f"  {label:<26} {_fmt(corr.get(name), '{:+.3f}')}")
    lines.append("")
    lines.append("Suggested weights (from positive correlations, sum 100) vs current config.toml weights.")
    lines.append("Printed only; config.toml is not changed.")
    rows = [[name, _fmt(current[name], "{:.1f}"), _fmt(suggested[name], "{:.1f}")] for name in SIGNAL_NAMES]
    lines.append(tabulate(rows, headers=["signal", "current", "suggested"], tablefmt="simple", disable_numparse=True))
    config_sum = sum(current.values())
    lines.append(
        f"Note: ol_injuries is forced to 0 in the backtest, so its suggested weight is 0. The suggested "
        f"weights sum to 100 while config.toml's sum to {config_sum:.0f}."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    df = run(args.seasons, cfg, min_week=args.min_week, max_week=args.max_week)
    print(format_report(df, cfg, args.seasons))
    return 0
