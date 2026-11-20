import pandas as pd
import pytest

from dstmatchup import cli, pipeline
from dstmatchup.config import load_config
from test_pipeline import REPO_CONFIG, SIGNALS, make_data


@pytest.fixture
def fake_data(monkeypatch):
    data = make_data()
    monkeypatch.setattr(cli, "load_schedule", lambda season: data["schedule"])
    monkeypatch.setattr(cli, "load_team_stats", lambda season: data["team_stats"])
    monkeypatch.setattr(cli, "load_pbp", lambda season: data["pbp"])
    monkeypatch.setattr(cli, "load_injuries", lambda season: data["injuries"])
    monkeypatch.setattr(cli, "load_depth_charts", lambda season: pd.DataFrame(
        columns=["team", "dt", "pos_abb", "pos_rank", "gsis_id"]))
    calls = []

    def fake_forecast(lat, lon, kickoff):
        calls.append((lat, lon, kickoff))
        return {"temp": 30.0, "wind": 5.0}

    monkeypatch.setattr(cli, "fetch_forecast", fake_forecast)
    return calls


@pytest.fixture
def spy_run(monkeypatch):
    captured = {}
    original = pipeline.run

    def spy(week, teams, cfg, data):
        captured.update(week=week, teams=teams, data=data)
        return original(week, teams, cfg, data)

    monkeypatch.setattr(pipeline, "run", spy)
    return captured


def test_invalid_team_returns_nonzero_and_lists_valid_codes(capsys):
    code = cli.main(["--teams", "ZZZ"])
    captured = capsys.readouterr()
    assert code != 0
    assert "ZZZ" in captured.err
    assert "BUF" in captured.err and "LA" in captured.err


def test_teams_flag_normalizes_and_filters(fake_data, spy_run, capsys):
    code = cli.main(["--teams", "lar,buf"])
    assert code == 0
    assert spy_run["teams"] == ["LA", "BUF"]


def test_output_table_has_header_and_rows(fake_data, capsys):
    code = cli.main(["--week", "5"])
    out = capsys.readouterr().out
    assert code == 0
    header = next(line for line in out.splitlines() if "Defense" in line)
    for column in ("Defense", "Opp", "H/A", "Opp Implied", "Opp TO/g", "Opp Sack%", "Opp QB", "Score"):
        assert column in header
    assert "BUF" in out and "DEN" in out


def test_breakdown_adds_each_signal_contribution(fake_data, capsys):
    code = cli.main(["--week", "5", "--breakdown"])
    out = capsys.readouterr().out
    assert code == 0
    header = next(line for line in out.splitlines() if line.startswith("Defense"))
    for name in SIGNALS:
        assert name in header


def test_plain_output_has_no_contribution_columns(fake_data, capsys):
    cli.main(["--week", "5"])
    header = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("Defense"))
    assert "turnovers" not in header and "implied_total" not in header


def test_played_and_bye_sections_printed(fake_data, capsys):
    cli.main(["--week", "5"])
    out = capsys.readouterr().out
    assert "Already played this week:" in out
    assert "GB @ CHI" in out
    assert "On bye:" in out
    assert "BAL" in out.split("On bye:")[1]


def test_forecast_uses_home_stadium_for_both_teams(fake_data, spy_run, monkeypatch, capsys):
    cli.main(["--week", "5"])
    calls = fake_data
    buf_lat, buf_lon = cli.STADIUM_COORDS["BUF"]
    assert (buf_lat, buf_lon, "2026-09-05 13:00") in calls
    forecast = spy_run["data"]["forecast"]
    assert "BUF" in forecast and "DEN" in forecast
    assert forecast["BUF"] == forecast["DEN"]
    assert "MIA" not in forecast


def test_forecast_failure_is_not_fatal(fake_data, monkeypatch, capsys):
    def boom(lat, lon, kickoff):
        raise RuntimeError("network down")

    monkeypatch.setattr(cli, "fetch_forecast", boom)
    code = cli.main(["--week", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "weather: roof only" in out


def test_default_available_teams_used_without_flag(fake_data, spy_run, monkeypatch, capsys):
    cfg = load_config(REPO_CONFIG)
    cfg["defaults"]["available_teams"] = ["buf", "den"]
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    cli.main(["--week", "5"])
    assert spy_run["teams"] == ["BUF", "DEN"]


def test_nan_score_prints_na():
    ranked = pd.DataFrame([{
        "defense": "ARI", "opponent": "ATL", "home": True, "opp_implied_total": float("nan"),
        "opp_turnovers_pg": float("nan"), "opp_sack_rate": float("nan"), "opp_qb_status": None,
        "score": float("nan"), "missing": ["turnovers"], "notes": "",
        **{f"c_{s}": 0.0 for s in SIGNALS},
    }])
    text = cli.format_result({"week": 5, "ranked": ranked, "played": pd.DataFrame(), "byes": []})
    row = next(line for line in text.splitlines() if line.startswith("ARI"))
    assert row.rstrip().endswith("n/a")


def test_python_m_entry_point_runs_help():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "dstmatchup", "--help"],
        cwd=repo, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--breakdown" in proc.stdout
