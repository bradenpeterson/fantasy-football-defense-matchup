"""Opponent and own-defense signals per defense for one week.

Pure functions: callers pass frames (see dstmatchup.loaders and
dstmatchup.dst_points). Only games strictly before `upto_week` are used.
Every signal column is oriented so higher is better for the defense.
No z-scoring or shrinkage happens here.
"""

import numpy as np
import pandas as pd

from dstmatchup.qb import qb_signal
from dstmatchup.weather import weather_signal

RATE_NAMES = ("turnovers_pg", "sack_rate_allowed", "ppg", "def_sacks_pg", "def_takeaways_pg")
TURNOVER_COLS = ("passing_interceptions", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
OL_POSITIONS = ("LT", "LG", "C", "RG", "RT")
INJURY_WEIGHTS = {"out": 1.0, "doubtful": 0.75, "questionable": 0.25}


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric stat column. A missing column raises; a NaN cell stays NaN."""
    if name not in df.columns:
        raise KeyError(f"team_stats is missing required column {name!r}")
    return pd.to_numeric(df[name], errors="coerce")


def _prior_regular_season(df: pd.DataFrame, upto_week: int) -> pd.DataFrame:
    out = df[df["week"] < upto_week]
    if "season_type" in out.columns:
        out = out[out["season_type"] == "REG"]
    return out


def _team_week_stats(stats: pd.DataFrame) -> pd.DataFrame:
    sacks = _num(stats, "sacks_suffered")
    attempts = _num(stats, "attempts")
    # min_count=1: a game with every component NaN is NaN, not 0 turnovers.
    turnovers = pd.concat([_num(stats, c) for c in TURNOVER_COLS], axis=1).sum(axis=1, min_count=1)
    return pd.DataFrame({
        "team": stats["team"].to_numpy(),
        "week": stats["week"].to_numpy(),
        "turnovers": turnovers.to_numpy(dtype=float),
        "sack_num": sacks.to_numpy(dtype=float),
        "sack_den": (attempts + sacks).to_numpy(dtype=float),
        "def_sacks": _num(stats, "def_sacks").to_numpy(dtype=float),
        "def_takeaways": pd.concat(
            [_num(stats, "def_interceptions"), _num(stats, "fumble_recovery_opp")], axis=1,
        ).sum(axis=1, min_count=1).to_numpy(dtype=float),
    })


def _points_scored(schedule: pd.DataFrame, upto_week: int) -> pd.DataFrame:
    games = schedule[schedule["week"] < upto_week]
    if "game_type" in games.columns:
        games = games[games["game_type"] == "REG"]
    games = games.dropna(subset=["home_score", "away_score"])
    home = pd.DataFrame({
        "team": games["home_team"].to_numpy(),
        "week": games["week"].to_numpy(),
        "pts": games["home_score"].astype(float).to_numpy(),
    })
    away = pd.DataFrame({
        "team": games["away_team"].to_numpy(),
        "week": games["week"].to_numpy(),
        "pts": games["away_score"].astype(float).to_numpy(),
    })
    return pd.concat([home, away], ignore_index=True)


def _mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else np.nan


def _pooled_ratio(num: pd.Series, den: pd.Series) -> float:
    mask = den > 0
    if not mask.any():
        return np.nan
    return float(num[mask].sum() / den[mask].sum())


def _summary(g: pd.DataFrame, last_n: int) -> dict:
    recent = g.tail(last_n)
    out = {"games": len(g)}
    for label, window in (("season", g), ("recent", recent)):
        out[f"turnovers_pg_{label}"] = _mean(window["turnovers"])
        out[f"sack_rate_allowed_{label}"] = _pooled_ratio(window["sack_num"], window["sack_den"])
        out[f"ppg_{label}"] = _mean(window["pts"])
        out[f"def_sacks_pg_{label}"] = _mean(window["def_sacks"])
        out[f"def_takeaways_pg_{label}"] = _mean(window["def_takeaways"])
    return out


def team_rates(
    team_stats: pd.DataFrame,
    upto_week: int,
    last_n: int = 4,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Season and last-`last_n` rates per team, from weeks strictly before `upto_week`.

    Indexed by team. `games` counts prior regular-season team-games. Teams with
    no games are NaN in every rate. `ppg` is NaN when `schedule` is None.
    """
    stats = _prior_regular_season(team_stats, upto_week)
    tw = _team_week_stats(stats)
    if schedule is not None:
        pts = _points_scored(schedule, upto_week)
        tw = tw.merge(pts, on=["team", "week"], how="left")
        universe = pd.unique(pd.concat([
            stats["team"], schedule["home_team"], schedule["away_team"],
        ]).dropna())
    else:
        tw["pts"] = np.nan
        universe = pd.unique(stats["team"].dropna())

    rows = {}
    for team in universe:
        g = tw[tw["team"] == team].sort_values("week", kind="stable")
        rows[team] = _summary(g, last_n)
    columns = ["games"] + [f"{n}_{w}" for n in RATE_NAMES for w in ("season", "recent")]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    out["games"] = out["games"].fillna(0).astype(int)
    out.index.name = "team"
    return out


def blend(season: pd.Series, recent: pd.Series, w_recent: float) -> pd.Series:
    """w_recent * recent + (1 - w_recent) * season; NaN recent falls back to season."""
    recent_filled = recent.fillna(season)
    return w_recent * recent_filled + (1.0 - w_recent) * season


def _at(series: pd.Series, keys) -> np.ndarray:
    return series.reindex(keys).to_numpy(dtype=float)


def rest_signal(opp_rest_days: float) -> float:
    """Opponent rest: short rest (<= 5 days) helps the defense, long rest (>= 10) hurts it."""
    if opp_rest_days is None or np.isnan(opp_rest_days):
        return np.nan
    if opp_rest_days <= 5:
        return 1.0
    if opp_rest_days >= 10:
        return -1.0
    return 0.0


def _is_published(report_status) -> bool:
    return isinstance(report_status, str) and report_status.strip() != ""


def _status_weight(report_status) -> float:
    if not isinstance(report_status, str):
        return 0.0
    return INJURY_WEIGHTS.get(report_status.strip().lower(), 0.0)


def ol_missing(injuries: pd.DataFrame, depth: pd.DataFrame, team: str, week: int) -> float:
    """Weighted count of `team`'s starting offensive linemen who are injured in `week`.

    Starters are depth rank 1 at LT, LG, C, RG, RT, taken from the team's latest
    depth snapshot. Out 1.0, Doubtful 0.75, Questionable 0.25; backups add 0.
    Only a published report counts: rows with a NULL report_status are practice
    reports, and a team with none for `week` is unknown (NaN) until game statuses
    are published. Returns NaN when the team has no depth rows or no published
    report, 0.0 when a published report hurts no starter.
    """
    team_depth = depth[depth["team"] == team]
    if team_depth.empty:
        return np.nan
    if "dt" in team_depth.columns:
        team_depth = team_depth[team_depth["dt"] == team_depth["dt"].max()]
    starters = team_depth[team_depth["pos_abb"].isin(OL_POSITIONS) & (team_depth["pos_rank"] == 1)]
    if starters.empty:
        return np.nan
    if injuries is None or injuries.empty:
        return np.nan
    team_week = injuries[(injuries["team"] == team) & (injuries["week"] == week)]
    published = team_week[team_week["report_status"].map(_is_published)]
    if published.empty:
        return np.nan
    hurt = published[published["gsis_id"].isin(set(starters["gsis_id"].dropna()))]
    if hurt.empty:
        return 0.0
    per_player = hurt.assign(_w=hurt["report_status"].map(_status_weight)).groupby("gsis_id")["_w"].max()
    return float(per_player.sum())


def _qb_value(status) -> float:
    if not isinstance(status, str):
        return np.nan
    value = qb_signal(status)
    return np.nan if value is None else value


def build_signal_frame(
    matchups: pd.DataFrame,
    rates: pd.DataFrame,
    allowed: pd.DataFrame,
    lines: pd.DataFrame,
    cfg: dict,
    qb_table: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    depth: pd.DataFrame | None = None,
    forecast: dict | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    """One row per defense team (matchups.team) with raw, higher-is-better signals.

    Signal columns (exactly the [weights] names): implied_total, qb_status,
    turnovers, sack_rate, dst_pts_allowed, opp_ppg, own_defense, home_away,
    weather, rest, ol_injuries. Also opp_games and own_games for shrinkage.

    Optional inputs: `qb_table` (team, qb_name, status) gives qb_status from the
    opponent's QB; `injuries` + `depth` + `week` give ol_injuries from the
    opponent's missing OL; `forecast` maps defense team to {"temp", "wind"} or
    None. A missing qb_table, or injuries/depth/week, leaves that column NaN.
    A missing forecast falls back to the roof-only weather signal.
    rest and weather use matchups' opp_rest_days and roof when present.
    """
    w = float(cfg["scoring"]["blend_recent"])
    blended = pd.DataFrame(index=rates.index)
    for name in RATE_NAMES:
        blended[name] = blend(rates[f"{name}_season"], rates[f"{name}_recent"], w)

    dst_allowed = allowed.set_index("offense")["dst_pts_allowed_per_game"]
    opp_implied = lines.set_index("team")["opp_implied_total"]

    defense = matchups["team"].to_numpy()
    opponent = matchups["opponent"].to_numpy()
    b_opp = blended.reindex(opponent)
    b_own = blended.reindex(defense)
    n = len(defense)

    if qb_table is not None:
        statuses = qb_table.set_index("team")["status"]
        qb_col = np.array([_qb_value(statuses.get(t)) for t in opponent], dtype=float)
    else:
        qb_col = np.full(n, np.nan)

    if "opp_rest_days" in matchups.columns:
        rest_col = np.array([rest_signal(x) for x in matchups["opp_rest_days"].to_numpy(dtype=float)])
    else:
        rest_col = np.full(n, np.nan)

    roof = matchups["roof"].to_numpy() if "roof" in matchups.columns else np.full(n, None, dtype=object)
    weather_col = np.array([
        np.nan if (value := weather_signal(r, (forecast or {}).get(d))) is None else value
        for r, d in zip(roof, defense)
    ], dtype=float)

    if injuries is not None and depth is not None and week is not None:
        ol_col = np.array([ol_missing(injuries, depth, t, week) for t in opponent], dtype=float)
    else:
        ol_col = np.full(n, np.nan)

    out = pd.DataFrame({
        "implied_total": -_at(opp_implied, defense),
        "qb_status": qb_col,
        "turnovers": b_opp["turnovers_pg"].to_numpy(dtype=float),
        "sack_rate": b_opp["sack_rate_allowed"].to_numpy(dtype=float),
        "dst_pts_allowed": _at(dst_allowed, opponent),
        "opp_ppg": -b_opp["ppg"].to_numpy(dtype=float),
        "own_defense": (
            b_own["def_sacks_pg"].to_numpy(dtype=float)
            + b_own["def_takeaways_pg"].to_numpy(dtype=float)
        ),
        "home_away": matchups["home"].astype(float).to_numpy(),
        "weather": weather_col,
        "rest": rest_col,
        "ol_injuries": ol_col,
        "opp_games": rates["games"].reindex(opponent).fillna(0).astype(int).to_numpy(),
        "own_games": rates["games"].reindex(defense).fillna(0).astype(int).to_numpy(),
    }, index=pd.Index(defense, name="team"))
    return out
