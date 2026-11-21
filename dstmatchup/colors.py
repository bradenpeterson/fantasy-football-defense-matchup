"""Plain ANSI colour for the ranked table: the Score cell and the Opp QB cell only.

Pure helpers with no I/O. The CLI decides whether colour is on (`use_color`)
and passes the result to the formatter, which calls `score_color`, `qb_color`
and `paint`.
"""

import math
from collections.abc import Mapping

GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

SCORE_GREEN_MIN = 60.0   # score >= this is green
SCORE_YELLOW_MIN = 45.0  # score >= this (and below green) is yellow; below it is red
QB_GREEN = ("backup", "out")
QB_YELLOW = ("questionable",)
MODES = ("auto", "always", "never")


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def score_color(score) -> str | None:
    """Colour for a 0-100 display score; None (uncoloured) when the score is missing."""
    if _is_missing(score):
        return None
    if score >= SCORE_GREEN_MIN:
        return GREEN
    if score >= SCORE_YELLOW_MIN:
        return YELLOW
    return RED


def qb_color(status) -> str | None:
    """Colour for an opponent QB status: backup or out is green, questionable yellow."""
    if not isinstance(status, str):
        return None
    text = status.strip().lower()
    if text in QB_GREEN:
        return GREEN
    if text in QB_YELLOW:
        return YELLOW
    return None


def paint(text: str, color: str | None) -> str:
    """Wrap `text` in `color` and a reset; with no colour, return it unchanged."""
    return text if color is None else f"{color}{text}{RESET}"


def use_color(mode: str, is_tty: bool, env: Mapping[str, str]) -> bool:
    """Whether to colour output. `always` and `never` force it; `auto` needs a TTY and no NO_COLOR."""
    if mode not in MODES:
        raise ValueError(f"unknown colour mode {mode!r}; expected one of {', '.join(MODES)}")
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(is_tty) and "NO_COLOR" not in env
