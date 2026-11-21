"""Plain-text 'Why' for one ranked row: the score drivers that move the defense most.

Pure helpers with no I/O. Input is the per-signal contributions in z-weighted units;
they are converted to score points with the config display scale.
"""

LABELS = {
    "implied_total": "implied pts",
    "qb_status": "QB",
    "turnovers": "turnovers",
    "sack_rate": "sacks",
    "dst_pts_allowed": "D/ST pts allowed",
    "opp_ppg": "opp scoring",
    "own_defense": "own D",
    "home_away": "home/away",
    "rest": "rest",
}


WHY_MIN_POINTS = 1.0


def why_text(contribs: dict[str, float], scale: float) -> str:
    """The two largest drivers as '▲ label  ▼ label'; '-' when none is large enough."""
    points = {name: float(value) * scale for name, value in contribs.items()}
    # LABELS is in SIGNAL_NAMES order; the stable sort keeps that order for equal sizes.
    ordered = [name for name in LABELS if name in points]
    ranked = sorted(
        (name for name in ordered if abs(points[name]) >= WHY_MIN_POINTS),
        key=lambda name: -abs(points[name]),
    )
    parts = []
    for name in ranked[:2]:
        arrow = "▲" if points[name] > 0 else "▼"
        parts.append(f"{arrow} {LABELS[name]}")
    return "  ".join(parts) if parts else "-"
