"""Parquet-backed cache for fetched DataFrames.

Current-season files expire after ttl_hours (judged by file mtime).
Past-season files never expire.
"""

import os
import time
from typing import Callable

import pandas as pd


def cached_frame(
    name: str,
    season: int,
    fetch: Callable[[], pd.DataFrame],
    current_season: int,
    ttl_hours: float = 6,
    cache_dir: str = ".cache",
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_{season}.parquet")

    if os.path.exists(path):
        fresh = True
        if season == current_season:
            age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
            fresh = age_hours < ttl_hours
        if fresh:
            return pd.read_parquet(path)

    df = fetch()
    df.to_parquet(path)
    return df
