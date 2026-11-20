import math

import numpy as np
import pandas as pd
import pytest

from dstmatchup.scoring import score_teams, shrink, zscore


def test_zscore_basic():
    z = zscore(pd.Series([1.0, 2.0, 3.0]))
    assert list(z.round(4)) == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)


def test_zscore_zero_variance_is_zero():
    z = zscore(pd.Series([5.0, 5.0, 5.0]))
    assert list(z) == [0.0, 0.0, 0.0]


def test_zscore_keeps_nan():
    z = zscore(pd.Series([1.0, np.nan, 3.0]))
    assert math.isnan(z.iloc[1])
    assert z.iloc[0] == pytest.approx(-1.0)
    assert z.iloc[2] == pytest.approx(1.0)


def test_shrink():
    assert shrink(pd.Series([2.0]), pd.Series([4.0]), 4.0).iloc[0] == pytest.approx(1.0)


def test_shrink_zero_n_is_nan():
    assert math.isnan(shrink(pd.Series([2.0]), pd.Series([0.0]), 4.0).iloc[0])


def test_missing_signal_rescales_weights():
    z = pd.DataFrame({"a": [1.0], "b": [np.nan]}, index=["X"])
    out = score_teams(z, {"a": 30, "b": 10})
    assert out.loc["X", "a"] == pytest.approx(1.0)
    assert out.loc["X", "missing"] == ["b"]
    assert out.loc["X", "score"] == pytest.approx(75.0)


def test_missing_contribution_is_zero():
    z = pd.DataFrame({"a": [1.0], "b": [np.nan]}, index=["X"])
    out = score_teams(z, {"a": 30, "b": 10})
    assert out.loc["X", "b"] == 0.0


def test_all_missing_gives_nan_score():
    z = pd.DataFrame({"a": [np.nan], "b": [np.nan]}, index=["X"])
    out = score_teams(z, {"a": 30, "b": 10})
    assert math.isnan(out.loc["X", "score"])
    assert sorted(out.loc["X", "missing"]) == ["a", "b"]


def test_score_clipped():
    z = pd.DataFrame({"a": [1000.0]}, index=["X"])
    out = score_teams(z, {"a": 1.0})
    assert out.loc["X", "score"] == 100.0


def test_score_clipped_low():
    z = pd.DataFrame({"a": [-1000.0]}, index=["X"])
    out = score_teams(z, {"a": 1.0})
    assert out.loc["X", "score"] == 0.0


def test_contributions_sum_to_weighted_mean():
    z = pd.DataFrame({"a": [0.5], "b": [-0.25], "c": [1.0]}, index=["X"])
    out = score_teams(z, {"a": 2.0, "b": 1.0, "c": 1.0})
    contrib_sum = out.loc["X", ["a", "b", "c"]].sum()
    assert contrib_sum == pytest.approx((out.loc["X", "score"] - 50) / 25)


def test_scale_parameter():
    z = pd.DataFrame({"a": [1.0]}, index=["X"])
    out = score_teams(z, {"a": 1.0}, scale=10.0)
    assert out.loc["X", "score"] == pytest.approx(60.0)


def test_zero_weight_present_signals_give_nan_score():
    z = pd.DataFrame({"a": [1.0], "b": [np.nan]}, index=["X"])
    out = score_teams(z, {"a": 0.0, "b": 10.0})
    assert math.isnan(out.loc["X", "score"])
