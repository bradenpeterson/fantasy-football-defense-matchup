# D/ST Matchup Scorer: Design

Date: 2026-11-20 (NFL Week 11 of the 2026 season). Status: awaiting approval.

## Intent and why

A personal CLI that ranks every NFL defense (D/ST) for a week by how weak the opposing offense looks, so the user can quickly decide which Yahoo D/ST to start or pick up. The user is new to the season's decisions and wants to sanity-check the numbers, so every score shows how each signal contributed.

## Goals

- Rank all defenses for a week by a combined matchup score; show defense, opponent, home/away, opponent implied total, opponent turnovers/game, opponent sacks allowed (rate), opponent QB status, score.
- Default week = earliest week with any unplayed game (currently 11). `--week N` overrides.
- `--teams BUF,DEN,...` (or `config.toml` list) restricts the table to available/rostered defenses; with neither, rank all 32.
- Per-signal contribution breakdown for each team; missing signals listed per row.
- All weights and Yahoo D/ST scoring values live in one `config.toml`.
- Local parquet cache; repeated runs do not re-fetch.
- Tests for scoring logic on small fixtures.
- Later: backtest over 2025 and played 2026 weeks, correlating the score and each signal with actual D/ST fantasy points, then replace guessed weights.
- Easy to add a local web page later (it imports the scoring core).

## Non-goals

- Yahoo Fantasy API integration; The Odds API (nflverse schedule lines only; `get_lines()` is a single interface so other sources can plug in later); a multi-week streaming view; a web UI in v1.

## Anti-goals

- A score that silently treats missing data as average; a claim that it works without running it; secrets in git.

## Verified facts (run 2026-11-20 in this environment)

- `nflreadpy` installs on Python 3.12 and loads schedules, team stats, injuries, depth charts, play-by-play and ff_opportunity for 2026 (stats through week 11). It returns polars; converting to pandas needs `pyarrow`.
- Schedule rows carry `spread_line`, `total_line`, `*_qb_name`, `away_rest`, `home_rest`, `roof` for Week 11 and 12. `temp` and `wind` are empty for future games.
- Week 11: 13 games, 12 unplayed (IND@HOU Nov 19 played), 6 teams on bye.
- Injuries have position (T, G, C...) and report status. Depth charts are a single snapshot (`dt`), not weekly.
- Open-Meteo forecast API: no response from this environment (unverified whether sandbox-related).
- Unverified: freshness of schedule lines and the source of week 12 QB names (PHI showing Joe Milton III looks odd), so QB names are only a hint.

## Constraints

- Python, `requirements.txt`, README, `.env` + `.gitignore`, no secrets committed.
- Never crash on bye weeks, missing QB info or small samples.

## Approaches considered

- **A. Small package with pure scoring core (chosen).** loaders to parquet cache, signals, scoring (no I/O), cli, `config.toml`. Easy tests, web page and backtest reuse the core.
- B. Single script/notebook: fast, hard to test, rewrite for web/backtest. Rejected.
- C. SQLite warehouse: overkill for 32 teams. Rejected.
Chosen by the user: A.

## Decisions (user-confirmed unless marked "my call")

- Lines: nflverse only. Implied total for a team = `(total ± spread)/2` using nflverse's convention (positive spread = home favored); verify with a fixture test.
- Opponent QB status: derive from (1) the opponent's QB with most dropbacks this season, (2) injury report, (3) schedule-named starter. Named starter != dropback leader means "backup"; injury overrides; conflicts or missing inputs mean "unknown" and the signal is excluded. A backup QB gives a large bonus. Amended 2026-11-20 (user-approved): the named starter also counts as `starter` if they led dropbacks in each of the team's last 2 games (`qb_recent_games`), so a mid-season QB change is not labeled backup.
- Signal 6: standard Yahoo D/ST scoring (sack 1, INT 2, fumble recovery 2, safety 2, TD 6, blocked kick 2, points-allowed tiers), configurable, computed as a proxy from nflverse stats per opponent for games played.
- Normalization: z-score each signal across 32 teams (distance from league average in units of typical spread), shrink toward the mean by `n/(n+k)` for small samples, blend season and last-4-games (default 50/50). Sacks use sack rate per dropback.
- Score = sum(weight x normalized signal), displayed 0-100; missing signals dropped, remaining weights rescaled, missing signals named on the row.
- Starting weights (guesses until the backtest): implied total 30, QB status 20, turnovers 15, sack rate 15, D/ST points allowed 10, opponent PPG 5, own defense 5; home/away, weather, rest and OL-injury signals small (my call: 3 each, total rescaled). Own defense and home/away kept small because implied total already reflects them.
- Partly played week: played games shown in a separate "already played" section, ranked only on unplayed games; their stats still feed season data (my call).
- Weather: roof (dome/closed favors the offense) plus an optional forecast fetch for wind/cold that fails gracefully ("weather: roof only"); retry the forecast outside the sandbox at implementation.
- OL injuries: a starter (depth-chart rank 1-5 at T/G/C) counts as missing at 1.0 if Out, 0.75 Doubtful, 0.25 Questionable; the total missing weight favors the defense.
- Rest: opponent on 5 days or fewer gives the defense a small bonus; opponent on 10 or more gives a small penalty.
- Amended 2026-11-20 (user-approved): the weather and OL-injury signals were removed; home/away stays as a signal (weight 3).
- Added 2026-11-20 (user-approved): `#` rank (league-wide) and `Why` (top two score drivers, at least 1 point) columns in the ranked table.

## Left to the builder

Module and function names, parquet cache TTL (my default: 6 hours for the current season, permanent for past seasons), exact shrinkage constant `k`, table formatting library, the precise points-allowed tier table, fixture contents.

## Build order

1. Skeleton, requirements, .gitignore, .env.example, cache layer.
2. Loaders and signals, shown on real Week 11 data.
3. Scoring plus fixture tests.
4. CLI with `--week` and `--teams`.
5. Weather, rest, OL-injury signals.
6. Backtest, then README.

Each step is run and its real output shown; anything not verified is reported as such.
