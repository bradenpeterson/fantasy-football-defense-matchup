import re

import pandas as pd
import pytest

from dstmatchup import cli, colors
from test_pipeline import make_data

GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_escape_codes_and_thresholds():
    assert (colors.GREEN, colors.YELLOW, colors.RED, colors.RESET) == (GREEN, YELLOW, RED, RESET)
    assert colors.SCORE_GREEN_MIN == 60.0
    assert colors.SCORE_YELLOW_MIN == 45.0


def test_score_color_green_at_and_above_60():
    assert colors.score_color(60.0) == GREEN
    assert colors.score_color(77.3) == GREEN


def test_score_color_yellow_from_45_up_to_just_under_60():
    assert colors.score_color(45.0) == YELLOW
    assert colors.score_color(59.99) == YELLOW


def test_score_color_red_below_45():
    assert colors.score_color(44.99) == RED
    assert colors.score_color(0.0) == RED


def test_score_color_nan_and_none_are_uncoloured():
    assert colors.score_color(float("nan")) is None
    assert colors.score_color(None) is None


@pytest.mark.parametrize("status, expected", [
    ("backup", GREEN),
    ("out", GREEN),
    ("Out", GREEN),
    ("questionable", YELLOW),
    ("starter", None),
    ("unknown", None),
    (None, None),
    (float("nan"), None),
])
def test_qb_color(status, expected):
    assert colors.qb_color(status) == expected


def test_paint_passes_plain_text_through_when_uncoloured():
    assert colors.paint("x", None) == "x"
    assert colors.paint("x", GREEN) == f"{GREEN}x{RESET}"


def test_use_color_modes():
    assert colors.use_color("always", False, {}) is True
    assert colors.use_color("always", True, {"NO_COLOR": "1"}) is True
    assert colors.use_color("never", True, {}) is False
    assert colors.use_color("auto", True, {}) is True
    assert colors.use_color("auto", False, {}) is False
    assert colors.use_color("auto", True, {"NO_COLOR": "1"}) is False
    assert colors.use_color("auto", True, {"NO_COLOR": ""}) is False


def test_use_color_rejects_unknown_mode():
    with pytest.raises(ValueError):
        colors.use_color("blue", True, {})


def _ranked(rows):
    return pd.DataFrame([{
        "defense": d, "opponent": "ZZZ", "home": True, "opp_implied_total": 20.0,
        "opp_turnovers_pg": 1.0, "opp_sack_rate": 0.05, "opp_qb_status": qb,
        "score": score, "missing": [], "notes": "", "rank": i + 1, "why": "-",
        **{f"c_{s}": 0.0 for s in cli.pipeline.SIGNAL_NAMES},
    } for i, (d, score, qb) in enumerate(rows)])


def test_format_result_colours_only_score_and_opp_qb_cells():
    ranked = _ranked([("ARI", 70.0, "backup"), ("KC", 50.0, "questionable"), ("LOW", 20.0, "starter")])
    result = {"week": 5, "ranked": ranked, "played": pd.DataFrame(), "byes": []}
    colored = cli.format_result(result, color=True)
    plain = cli.format_result(result, color=False)

    rows = {line.split()[1]: line for line in colored.splitlines() if len(line.split()) > 1}
    assert f"{GREEN}70.0{RESET}" in rows["ARI"] and f"{GREEN}backup{RESET}" in rows["ARI"]
    assert f"{YELLOW}50.0{RESET}" in rows["KC"] and f"{YELLOW}questionable{RESET}" in rows["KC"]
    assert f"{RED}20.0{RESET}" in rows["LOW"]
    assert ANSI.findall(rows["LOW"]) == [RED, RESET]  # starter stays uncoloured
    assert ANSI.findall(rows["ARI"]) == [GREEN, RESET, GREEN, RESET]
    # Stripping the codes must give back the plain table, so alignment is unchanged.
    assert ANSI.sub("", colored) == plain
    assert "\x1b[" not in plain


def test_format_result_leaves_na_score_uncoloured():
    ranked = _ranked([("ARI", float("nan"), None)])
    text = cli.format_result(
        {"week": 5, "ranked": ranked, "played": pd.DataFrame(), "byes": []}, color=True,
    )
    row = next(line for line in text.splitlines() if line.split()[1:2] == ["ARI"])
    assert ANSI.findall(row) == []
    assert row.split()[-2:] == ["n/a", "-"]


@pytest.fixture
def cli_data(monkeypatch):
    data = make_data()
    monkeypatch.setattr(cli, "load_schedule", lambda season: data["schedule"])
    monkeypatch.setattr(cli, "load_team_stats", lambda season: data["team_stats"])
    monkeypatch.setattr(cli, "load_pbp", lambda season: data["pbp"])
    monkeypatch.setattr(cli, "load_injuries", lambda season: data["injuries"])


def test_cli_color_always_emits_codes(cli_data, capsys):
    assert cli.main(["--week", "5", "--color", "always"]) == 0
    assert "\x1b[" in capsys.readouterr().out


def test_cli_color_never_emits_no_codes_even_on_a_tty(cli_data, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_stdout_is_tty", lambda: True)
    assert cli.main(["--week", "5", "--color", "never"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_cli_default_auto_is_plain_when_not_a_tty(cli_data, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_stdout_is_tty", lambda: False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli.main(["--week", "5"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_cli_default_auto_colours_a_tty(cli_data, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_stdout_is_tty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli.main(["--week", "5"]) == 0
    assert "\x1b[" in capsys.readouterr().out


def test_cli_no_color_env_disables_auto_on_a_tty(cli_data, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_stdout_is_tty", lambda: True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli.main(["--week", "5"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_cli_rejects_unknown_color_mode(cli_data, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--color", "blue"])
    assert exc.value.code == 2
