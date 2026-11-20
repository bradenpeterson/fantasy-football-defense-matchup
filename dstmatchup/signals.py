"""Opponent and own-defense signals per defense for one week.

Pure functions: callers pass frames (see dstmatchup.loaders and
dstmatchup.dst_points). Only games strictly before `upto_week` are used.
Every signal column is oriented so higher is better for the defense.
No z-scoring or shrinkage happens here.
"""

import numpy as np
import pandas as pd

RATE_NAMES = ("turnovers_pg", "sack_rate_allowed", "ppg", "def_sacks_pg", "def_takeaways_pg")
TURNOVER_COLS = ("passing_interceptions", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _prior_regular_season(df: pd.DataFrame, upto_week: int) -> pd.DataFrame:
    out = df[df["week"] < upto_week]
    if "season_type" in out.columns:
        out = out[out["season_type"] == "REG"]
    return out


def _team_week_stats(stats: pd.DataFrame) -> pd.DataFrame:
    sacks = _num(stats, "sacks_suffered")
    attempts = _num(stats, "attempts")
    turnovers = sum(_num(stats, c) for c in TURNOVER_COLS)
    return pd.DataFrame({
        "team": stats["team"].to_numpy(),
        "week": stats["week"].to_numpy(),
        "turnovers": turnovers.to_numpy(dtype=float),
        "sack_num": sacks.to_numpy(dtype=float),
        "sack_den": (attempts + sacks).to_numpy(dtype=float),
        "def_sacks": _num(stats, "def_sacks").to_numpy(dtype=float),
        "def_takeaways": (
            _num(stats, "def_interceptions") + _num(stats, "fumble_recovery_opp")
        ).to_numpy(dtype=float),
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


def build_signal_frame(
    matchups: pd.DataFrame,
    rates: pd.DataFrame,
    allowed: pd.DataFrame,
    lines: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """One row per defense team (matchups.team) with raw, higher-is-better signals.

    Columns: implied_total, turnovers, sack_rate, dst_pts_allowed, opp_ppg,
    own_defense, home_away, opp_games, own_games.
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

    out = pd.DataFrame({
        "implied_total": -_at(opp_implied, defense),
        "turnovers": b_opp["turnovers_pg"].to_numpy(dtype=float),
        "sack_rate": b_opp["sack_rate_allowed"].to_numpy(dtype=float),
        "dst_pts_allowed": _at(dst_allowed, opponent),
        "opp_ppg": -b_opp["ppg"].to_numpy(dtype=float),
        "own_defense": (
            b_own["def_sacks_pg"].to_numpy(dtype=float)
            + b_own["def_takeaways_pg"].to_numpy(dtype=float)
        ),
        "home_away": matchups["home"].astype(float).to_numpy(),
        "opp_games": rates["games"].reindex(opponent).fillna(0).astype(int).to_numpy(),
        "own_games": rates["games"].reindex(defense).fillna(0).astype(int).to_numpy(),
    }, index=pd.Index(defense, name="team"))
    return out
