"""Pure scoring functions for DST matchup ranking.

No I/O and no config reading: callers pass in data and weights.
"""

import numpy as np
import pandas as pd


def zscore(values: pd.Series) -> pd.Series:
    """Population z-score across teams.

    NaN entries stay NaN. If the non-NaN values have zero variance, every
    non-NaN entry maps to 0.0.
    """
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values.where(values.isna(), 0.0).astype(float)
    return (values - mean) / std


def shrink(z: pd.Series, n: pd.Series, k: float) -> pd.Series:
    """Shrink z toward 0 by sample size: z * n / (n + k).

    Where n == 0 the result is NaN (no evidence), not 0.
    """
    result = z * n / (n + k)
    return result.where(n != 0, np.nan)


def score_teams(
    z_by_signal: pd.DataFrame,
    weights: dict[str, float],
    scale: float = 25.0,
) -> pd.DataFrame:
    """Combine per-signal z-scores into a 0-100 display score per team.

    Missing signals (NaN z, or a signal column absent from ``z_by_signal``)
    are dropped for that team and the remaining weights are rescaled to sum
    to 1. Missing signal names are listed in ``missing``.

    Returns a DataFrame indexed like ``z_by_signal`` with columns:
      - ``score``: clip(50 + scale * weighted_mean_z, 0, 100); NaN when every
        signal is missing.
      - ``missing``: list of signal names with NaN z for that team.
      - one contribution column per signal: rescaled_weight * z. Missing
        signals contribute 0.0 (not NaN), so the contribution columns sum to
        the weighted mean z.
    """
    signals = list(weights)
    z = z_by_signal.reindex(columns=signals).astype(float)
    present = z.notna()

    w = pd.Series(weights, dtype=float)
    present_weight = present.mul(w, axis=1).sum(axis=1)
    rescaled = present.mul(w, axis=1).div(present_weight.replace(0.0, np.nan), axis=0)

    contrib = (rescaled * z).fillna(0.0)
    weighted_mean_z = contrib.sum(axis=1).where(present.any(axis=1), np.nan)

    score = (50.0 + scale * weighted_mean_z).clip(lower=0.0, upper=100.0)

    missing = z.apply(lambda row: [s for s in signals if pd.isna(row[s])], axis=1)

    out = pd.DataFrame(index=z_by_signal.index)
    out["score"] = score
    out["missing"] = missing
    for s in signals:
        out[s] = contrib[s]
    return out
