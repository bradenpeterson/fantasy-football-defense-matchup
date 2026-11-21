from pathlib import Path

import pytest

from dstmatchup.config import load_config

REPO_CONFIG = str(Path(__file__).resolve().parent.parent / "config.toml")


def test_load_config_has_weights():
    cfg = load_config(REPO_CONFIG)
    assert cfg["weights"]["implied_total"] == 30
    assert cfg["scoring"]["shrink_k"] == 4


def test_load_config_rejects_negative_weight(tmp_path):
    text = Path(REPO_CONFIG).read_text()
    assert "qb_status = 20" in text
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("qb_status = 20", "qb_status = -1"))
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_weights_are_exactly_the_nine_remaining_signals():
    cfg = load_config(REPO_CONFIG)
    assert set(cfg["weights"]) == {
        "implied_total", "qb_status", "turnovers", "sack_rate", "dst_pts_allowed",
        "opp_ppg", "own_defense", "home_away", "rest",
    }
    assert "weather" not in Path(REPO_CONFIG).read_text()


def test_load_config_default_path_works_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["weights"]["implied_total"] == 30
