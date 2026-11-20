import os
import time

import pandas as pd

from dstmatchup.cache import cached_frame


def _fetcher(counter):
    def fetch():
        counter["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    return fetch


def test_cache_hit_skips_fetch(tmp_path):
    counter = {"n": 0}
    cache_dir = str(tmp_path / "cache")
    fetch = _fetcher(counter)
    first = cached_frame("sched", 2026, fetch, current_season=2026, ttl_hours=6, cache_dir=cache_dir)
    second = cached_frame("sched", 2026, fetch, current_season=2026, ttl_hours=6, cache_dir=cache_dir)
    assert counter["n"] == 1
    assert len(first) == 3
    assert len(second) == 3


def test_cache_expires_for_current_season(tmp_path):
    counter = {"n": 0}
    cache_dir = str(tmp_path / "cache")
    fetch = _fetcher(counter)
    cached_frame("sched", 2026, fetch, current_season=2026, ttl_hours=0, cache_dir=cache_dir)
    time.sleep(0.01)
    cached_frame("sched", 2026, fetch, current_season=2026, ttl_hours=0, cache_dir=cache_dir)
    assert counter["n"] == 2


def test_past_season_never_expires(tmp_path):
    counter = {"n": 0}
    cache_dir = str(tmp_path / "cache")
    fetch = _fetcher(counter)
    cached_frame("sched", 2024, fetch, current_season=2026, ttl_hours=0, cache_dir=cache_dir)
    # Make the file look very old; past seasons must still be served from cache.
    for root, _, files in os.walk(cache_dir):
        for f in files:
            p = os.path.join(root, f)
            os.utime(p, (0, 0))
    cached_frame("sched", 2024, fetch, current_season=2026, ttl_hours=0, cache_dir=cache_dir)
    assert counter["n"] == 1
