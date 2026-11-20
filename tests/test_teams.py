import pytest

from dstmatchup.teams import ALL_TEAMS, normalize_team


def test_normalize_team():
    assert normalize_team("lar") == "LA"
    assert normalize_team("wsh") == "WAS"
    assert normalize_team("BUF") == "BUF"


def test_normalize_team_aliases():
    assert normalize_team("LAR") == "LA"
    assert normalize_team("JAC") == "JAX"


def test_normalize_team_rejects_unknown_with_valid_codes_listed():
    with pytest.raises(ValueError) as exc:
        normalize_team("XXX")
    assert "BUF" in str(exc.value)


def test_all_teams_has_32_codes():
    assert len(ALL_TEAMS) == 32
    assert "LA" in ALL_TEAMS
    assert "LAR" not in ALL_TEAMS
