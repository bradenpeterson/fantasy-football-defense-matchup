"""Load and validate config.toml."""

import tomllib
from pathlib import Path

REQUIRED = {
    "weights": [
        "implied_total",
        "qb_status",
        "turnovers",
        "sack_rate",
        "dst_pts_allowed",
        "opp_ppg",
        "own_defense",
        "home_away",
        "rest",
    ],
    "scoring": ["shrink_k", "display_scale", "blend_recent", "last_n"],
    "yahoo": [
        "sack",
        "interception",
        "fumble_recovery",
        "safety",
        "td",
        "blocked_kick",
        "tiers",
    ],
    "defaults": ["available_teams"],
}


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.toml"


def load_config(path: str | None = None) -> dict:
    """Load config.toml. With no path, use the repository's config.toml."""
    config_path = DEFAULT_CONFIG_PATH if path is None else path
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"config missing section [{section}]")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"config missing [{section}].{key}")

    for name, value in cfg["weights"].items():
        if value < 0:
            raise ValueError(f"weight [{name}] must be non-negative, got {value}")

    return cfg
