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


def test_cli_has_no_forecast_depth_chart_or_stadium_code():
    for name in ("load_depth_charts", "fetch_forecast", "STADIUM_COORDS", "_forecasts"):
        assert not hasattr(cli, name), name


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
        "score": float("nan"), "missing": ["turnovers"],
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


def test_teams_filter_shows_played_game_for_away_side(fake_data, capsys):
    code = cli.main(["--week", "5", "--teams", "GB"])
    out = capsys.readouterr().out
    assert code == 0
    played_section = out.split("Already played this week:")[1].split("On bye:")[0]
    assert "GB @ CHI" in played_section
    assert "none" not in played_section


def test_played_game_printed_once_when_both_sides_wanted(fake_data, capsys):
    cli.main(["--week", "5", "--teams", "CHI,GB"])
    out = capsys.readouterr().out
    assert out.count("GB @ CHI") == 1


def test_backtest_subcommand_dispatches_and_prints_summary(monkeypatch, capsys):
    from dstmatchup import backtest

    data = make_data()
    loaded = []

    def fake_season_data(season):
        loaded.append(season)
        return {"schedule": data["schedule"], "team_stats": data["team_stats"], "pbp": data["pbp"]}

    monkeypatch.setattr(backtest, "_season_data", fake_season_data)
    code = cli.main(["backtest", "--seasons", "2025", "2026"])
    out = capsys.readouterr().out
    assert code == 0
    assert loaded == [2025, 2026]
    assert "Sample size" in out
    assert "small sample" in out
    assert "ol_injuries" not in out and "weather" not in out
    assert "Suggested weights" in out
    for name in SIGNALS:
        assert name in out


def test_backtest_discloses_qb_hindsight_and_no_removed_signals(monkeypatch, capsys):
    from dstmatchup import backtest

    data = make_data()
    monkeypatch.setattr(backtest, "_season_data", lambda season: {
        "schedule": data["schedule"], "team_stats": data["team_stats"], "pbp": data["pbp"],
    })
    assert cli.main(["backtest", "--seasons", "2025"]) == 0
    out = capsys.readouterr().out
    assert "qb_status uses the schedule-named kickoff starter" in out
    assert "known at kickoff, not at earlier lineup deadlines" in out
    assert "mildly optimistic" in out
    qb_line = next(line for line in out.splitlines() if line.startswith("  qb_status"))
    assert "(kickoff starter)" in qb_line
    assert "(hindsight)" not in out
    assert "closing lines" in out
    assert "ol_injuries" not in out and "weather" not in out


def test_empty_teams_list_is_an_error_not_an_empty_table(capsys):
    code = cli.main(["--teams", ""])
    captured = capsys.readouterr()
    assert code == 2
    assert "--teams" in captured.err
    assert "Defense" not in captured.out


def test_breakdown_prints_na_for_missing_signal_contribution():
    ranked = pd.DataFrame([{
        "defense": "ARI", "opponent": "ATL", "home": True, "opp_implied_total": 22.0,
        "opp_turnovers_pg": 1.0, "opp_sack_rate": 0.05, "opp_qb_status": "starter",
        "score": 55.0, "missing": ["turnovers"],
        **{f"c_{s}": (0.5 if s == "implied_total" else 0.0) for s in pipeline.SIGNAL_NAMES},
    }])
    text = cli.format_result(
        {"week": 5, "ranked": ranked, "played": pd.DataFrame(), "byes": []}, breakdown=True,
    )
    # Breakdown order follows the signal order: implied_total, qb_status, turnovers, ...
    row = next(line for line in text.splitlines() if line.startswith("ARI"))
    after_score = row.split()[row.split().index("55.0") + 1:]
    assert after_score[0] == "+0.50"  # implied_total, present
    assert after_score[2] == "n/a"    # turnovers, missing
