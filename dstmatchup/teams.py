"""NFL team abbreviations, normalized to nflverse codes."""

ALIASES = {
    "LAR": "LA",
    "WSH": "WAS",
    "JAC": "JAX",
}

ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LV", "LAC", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS",
]


def normalize_team(code: str) -> str:
    upper = str(code).strip().upper()
    upper = ALIASES.get(upper, upper)
    if upper not in ALL_TEAMS:
        raise ValueError(
            f"unknown team {code!r}; valid codes: {', '.join(ALL_TEAMS)}"
        )
    return upper
