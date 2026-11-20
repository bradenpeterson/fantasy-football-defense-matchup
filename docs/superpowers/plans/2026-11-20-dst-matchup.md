# D/ST Matchup Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI that ranks NFL D/STs for a week by opposing-offense weakness, with a per-signal contribution breakdown, then a backtest that tunes the weights.

**Architecture:** Package `dstmatchup/` at the repo root. `loaders` fetch nflverse data into a parquet cache; `signals` turn raw frames into one number per signal per team (higher = better for the defense); `scoring` is pure functions with no I/O (normalize, shrink, weight, combine); `cli` prints the table. All weights and Yahoo scoring live in `config.toml`.

**Tech Stack:** Python 3.12, nflreadpy (returns polars) + pyarrow/pandas, pytest, tabulate, tomllib (stdlib).

**Spec:** `docs/superpowers/specs/2026-11-20-dst-matchup-design.md`

**Process:** Strict TDD in every task: failing test, run it and see it fail, minimal code, run it and see it pass, commit. Tests use small hand-built pandas fixtures, never the network. Real-data runs are separate "Show" steps whose output is reported verbatim; say what could not be verified.

## Global Constraints

- Python with deps in `requirements.txt`; README with setup and usage; secrets only in `.env` (gitignored), `.env.example` committed.
- nflverse data only for lines (no Odds API). `get_lines()` is the single interface for lines.
- Implied total: nflverse `spread_line` is positive when the HOME team is favored; home implied = `(total + spread)/2`, away implied = `(total - spread)/2`.
- Rams are `LA` in nflverse; normalize every team abbreviation through `teams.normalize_team`.
- Default week = earliest week with any unplayed game (`result` null); `--week N` overrides. Played games are listed separately and excluded from the ranking.
- Missing signals are never treated as average: drop them, rescale remaining weights to sum to 1, name them on the row.
- Shrinkage `n/(n+k)`, default `k = 4`; season/last-4 blend default 50/50; sack signal is sack rate per dropback; display score = `clip(50 + 25 * weighted_mean_z, 0, 100)`. These live in `config.toml`.
- Starting weights: implied_total 30, qb_status 20, turnovers 15, sack_rate 15, dst_pts_allowed 10, opp_ppg 5, own_defense 5, home_away 3, weather 3, rest 3, ol_injuries 3.
- Yahoo D/ST scoring in config: sack 1, interception 2, fumble_recovery 2, safety 2, td 6, blocked_kick 2; points allowed 0 => 10, 1-6 => 7, 7-13 => 4, 14-20 => 1, 21-27 => 0, 28-34 => -1, 35+ => -4.
- Cache TTL: 6 hours for the current season, never expires for past seasons.
- QB status values: `starter`, `backup`, `questionable`, `out`, `unknown`.

## Review Focus

- Bye-week team or a team with zero games played (week 1): no crash, row shows missing signals. (Task 5, 7)
- A QB or team with zero dropbacks: no divide-by-zero, signal missing. (Task 5)
- QB inputs conflict (named starter vs injury vs dropback leader): status `unknown`, signal excluded. (Task 6)
- `--teams` contains an unknown or lowercase abbreviation, or `LAR` instead of `LA`: normalized or a clear error listing valid codes. (Task 8)
- All signals missing for a team, or a signal with zero variance across teams: score shown as `n/a` / z of 0, no crash. (Task 3)

---

### Task 1: Skeleton, config, team codes, cache

**Files:**
- Create: `requirements.txt`, `.gitignore` (`.env`, `.venv/`, `.cache/`, `__pycache__/`), `.env.example`, `config.toml`, `dstmatchup/__init__.py`, `dstmatchup/config.py`, `dstmatchup/teams.py`, `dstmatchup/cache.py`
- Test: `tests/test_config.py`, `tests/test_teams.py`, `tests/test_cache.py`

**Interfaces:**
- Produces: `config.load_config(path: str = "config.toml") -> dict` (parsed TOML, validates the keys in Global Constraints exist and weights are non-negative, else `ValueError`); `teams.normalize_team(code: str) -> str` (uppercases; maps `LAR`->`LA`, `WSH`->`WAS`, `JAC`->`JAX`; raises `ValueError` listing valid codes otherwise); `teams.ALL_TEAMS: list[str]` (32 codes); `cache.cached_frame(name: str, season: int, fetch: Callable[[], pd.DataFrame], current_season: int, ttl_hours: float = 6, cache_dir: str = ".cache") -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests**
  - `test_load_config_has_weights`: loaded config has `weights["implied_total"] == 30` and `scoring["shrink_k"] == 4`.
  - `test_load_config_rejects_negative_weight`: a temp TOML with `qb_status = -1` raises `ValueError`.
  - `test_normalize_team`: `"lar" -> "LA"`, `"wsh" -> "WAS"`, `"BUF" -> "BUF"`; `"XXX"` raises `ValueError` whose message contains `"BUF"`; `len(ALL_TEAMS) == 32`.
  - `test_cache_hit_skips_fetch` (tmp_path): second call with same name/season does not invoke `fetch` (counter stays 1).
  - `test_cache_expires_for_current_season`: with `ttl_hours=0` fetch is called twice; for `season != current_season` it is called once even with `ttl_hours=0`.
- [ ] **Step 2: Run** `python -m pytest tests -v`. Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** the functions above and `config.toml` with every value from Global Constraints (sections `[weights]`, `[scoring]`, `[yahoo]`, `[defaults]` with `available_teams = []`). Parquet via `DataFrame.to_parquet`; freshness via file mtime.
- [ ] **Step 4: Run** `python -m pytest tests -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: skeleton, config, team codes, cache"`

### Task 2: Loaders, schedule, lines, matchups

**Files:**
- Create: `dstmatchup/loaders.py`, `dstmatchup/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: `cache.cached_frame`, `teams.normalize_team`.
- Produces: `loaders.load_schedule/load_team_stats/load_pbp/load_injuries/load_depth_charts(season: int) -> pd.DataFrame` (nflreadpy, `.to_pandas()`, cached, team columns normalized); `schedule.current_week(schedule: pd.DataFrame) -> int`; `schedule.get_lines(schedule: pd.DataFrame, week: int) -> pd.DataFrame` with columns `team, opponent, home (bool), implied_total, opp_implied_total, total, spread_for_team`; `schedule.matchups(schedule, week) -> pd.DataFrame` one row per team playing, columns `team, opponent, home, gameday, played (bool), rest_days, opp_rest_days, roof`; `schedule.bye_teams(schedule, week) -> list[str]`.

- [ ] **Step 1: Write failing tests** (fixture: 3 games, week 11, one with `result` set)
  - `test_implied_totals`: home favored by 3.5 on total 45.5 gives home `24.5`, away `21.0`; the underdog row has `spread_for_team == -3.5`.
  - `test_current_week_is_earliest_with_unplayed`: weeks 10 (all played), 11 (one played, others null) gives `11`.
  - `test_matchups_marks_played_and_bye`: played game rows have `played == True`; a team in the 32 but not in the fixture appears in `bye_teams`.
  - `test_current_week_all_played_returns_next`: if every game is played, returns max week + 1 only if it exists, else the max week.
- [ ] **Step 2: Run** `python -m pytest tests/test_schedule.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the functions above; loaders carry no tests (thin wrappers).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Show (real data):** `python -c "from dstmatchup.loaders import load_schedule; from dstmatchup.schedule import *; s=load_schedule(2026); w=current_week(s); print(w, get_lines(s,w).head(), bye_teams(s,w))"`. Expected: `11`, 26 playing teams, 6 byes. Report the actual output.
- [ ] **Step 6: Commit** `git commit -am "feat: loaders, schedule, lines, matchups"` (add new files first).

### Task 3: Scoring core (pure)

**Files:**
- Create: `dstmatchup/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces: `zscore(values: pd.Series) -> pd.Series` (population std; NaN stays NaN; zero variance gives 0.0); `shrink(z: pd.Series, n: pd.Series, k: float) -> pd.Series` (`z * n/(n+k)`; `n == 0` gives NaN); `score_teams(z_by_signal: pd.DataFrame, weights: dict[str, float], scale: float = 25.0) -> pd.DataFrame` indexed by team with columns `score, missing` (list of names) and one contribution column per signal; contribution = `rescaled_weight * z`; missing signals have NaN z and contribute nothing.

- [ ] **Step 1: Write failing tests**
  - `test_zscore_basic`: `[1,2,3] -> [-1.2247, 0, 1.2247]` (approx); `[5,5,5] -> [0,0,0]`.
  - `test_shrink`: `z=2, n=4, k=4 -> 1.0`; `n=0 -> NaN`.
  - `test_missing_signal_rescales_weights`: weights `{a: 30, b: 10}`, team X has `a=1.0, b=NaN`: contributions `a == 1.0` (weight rescaled to 1), `missing == ["b"]`, `score == 75.0`.
  - `test_all_missing_gives_nan_score`: team with every z NaN has `score` NaN, `missing` listing all signals.
  - `test_score_clipped`: huge z gives score `100.0`.
  - `test_contributions_sum_to_weighted_mean`: contributions sum equals `(score - 50) / 25` (unclipped case).
- [ ] **Step 2: Run** `python -m pytest tests/test_scoring.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three functions, no I/O or config reading.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: scoring core`.

### Task 4: Yahoo D/ST points proxy

**Files:**
- Create: `dstmatchup/dst_points.py`
- Test: `tests/test_dst_points.py`

**Interfaces:**
- Consumes: `config` yahoo section.
- Produces: `points_allowed_score(points: int, tiers: list) -> int`; `dst_points_by_game(team_stats: pd.DataFrame, schedule: pd.DataFrame, yahoo: dict) -> pd.DataFrame` columns `team, week, opponent, dst_points` (defense of `team` in that game, from team stats `def_sacks`, `def_interceptions`, fumble recoveries if present, `def_safeties`, `def_tds`, blocks, plus the points the opposing offense scored from the schedule); `dst_points_allowed_to(df: pd.DataFrame) -> pd.DataFrame` columns `offense, games, dst_pts_allowed_per_game` (what each offense has given up to opposing D/STs).

- [ ] **Step 1: Write failing tests**
  - `test_points_allowed_tiers`: `0 -> 10`, `6 -> 7`, `7 -> 4`, `20 -> 1`, `27 -> 0`, `34 -> -1`, `35 -> -4`.
  - `test_dst_points_by_game`: fixture defense with 3 sacks, 1 INT, 0 TD allowing 13 points = `3 + 2 + 4 = 9`.
  - `test_allowed_to_averages_per_offense`: two games against offense X with 9 and 3 give `6.0` over `2` games; a team with no games is absent, not zero.
- [ ] **Step 2: Run** tests; Expected FAIL. **Step 3: Implement.** **Step 4: Run;** Expected PASS.
- [ ] **Step 5: Show (real data):** top and bottom 5 offenses by `dst_pts_allowed_per_game` for 2026. Report output; note it is a proxy (team stats lack special-teams blocks and fumble recoveries may be missing).
- [ ] **Step 6: Commit** `feat: Yahoo D/ST points proxy`.

### Task 5: Opponent and own-defense signals

**Files:**
- Create: `dstmatchup/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: Task 3 `zscore`/`shrink`, Task 4 `dst_points_allowed_to`.
- Produces: `signals.team_rates(team_stats: pd.DataFrame, upto_week: int, last_n: int = 4) -> pd.DataFrame` indexed by team with `games`, and season and last-n values for `turnovers_pg` (interceptions thrown + fumbles lost), `sack_rate_allowed` (`sacks_suffered / (attempts + sacks_suffered)`), `ppg`, `def_sacks_pg`, `def_takeaways_pg`; uses only weeks `< upto_week` (so Week 11's played Thursday game does not feed Week 11 scoring). `signals.blend(season: pd.Series, recent: pd.Series, w_recent: float) -> pd.Series`; `signals.build_signal_frame(matchups: pd.DataFrame, rates: pd.DataFrame, allowed: pd.DataFrame, lines: pd.DataFrame, cfg: dict) -> pd.DataFrame` indexed by defense team, one raw column per signal in `{implied_total, turnovers, sack_rate, dst_pts_allowed, opp_ppg, own_defense, home_away}`, each oriented higher = better for the defense (implied total, ppg negated), plus `opp_games`.

- [ ] **Step 1: Write failing tests**
  - `test_rates_only_use_prior_weeks`: week-11 stats in the fixture are ignored when `upto_week=11`.
  - `test_sack_rate_zero_dropbacks_is_nan`: a team with 0 attempts and 0 sacks gives NaN, not an error.
  - `test_blend_half_half`: `blend(season=10, recent=20, 0.5) == 15`; recent NaN falls back to season.
  - `test_signal_orientation`: opponent with lower implied total gets a higher `implied_total` raw value; opponent with more turnovers gets a higher `turnovers` value.
  - `test_bye_team_absent_and_zero_games_missing`: a team with `games == 0` has NaN signals, not zeros.
- [ ] **Step 2: Run;** Expected FAIL. **Step 3: Implement.** **Step 4: Run;** Expected PASS.
- [ ] **Step 5: Show (real data)** the Week 11 signal frame head. Report it.
- [ ] **Step 6: Commit** `feat: opponent and own-defense signals`.

### Task 6: Opponent QB status

**Files:**
- Create: `dstmatchup/qb.py`
- Test: `tests/test_qb.py`

**Interfaces:**
- Produces: `qb_status(named_starter: str | None, dropback_leader: str | None, injury_status: str | None) -> str` returning one of the five QB status values; `qb_status_table(schedule, pbp, injuries, week) -> pd.DataFrame` columns `team, qb_name, status`; `qb_signal(status: str) -> float | None` (`backup` 1.0, `out` 1.0, `questionable` 0.5, `starter` 0.0, `unknown` None).

- [ ] **Step 1: Write failing tests**
  - `starter == leader`, no injury -> `starter`; `starter != leader` -> `backup`; starter listed `Out` -> `out`; `Questionable` -> `questionable`.
  - `test_conflict_is_unknown`: named starter missing and no dropback leader -> `unknown`; named starter equals leader but leader is `Out` per injuries -> `out` (injury overrides).
  - `test_signal_unknown_is_none`: `qb_signal("unknown") is None`.
  - `test_qb_status_table_matches_names_by_normalized_string` (lowercase, suffix `III`/`Jr.` stripped) so "Joe Milton III" equals "Joe Milton".
- [ ] **Step 2: Run;** Expected FAIL. **Step 3: Implement.** **Step 4: Run;** Expected PASS.
- [ ] **Step 5: Show (real data)** the status table for Week 11; list every team marked `unknown` or `backup` and say the schedule QB names were not independently verified.
- [ ] **Step 6: Commit** `feat: opponent QB status`.

### Task 7: Rest, OL injuries, weather (roof)

**Files:**
- Modify: `dstmatchup/signals.py` (add three functions and add columns to `build_signal_frame`), Create: `dstmatchup/weather.py`
- Test: `tests/test_extra_signals.py`

**Interfaces:**
- Produces: `signals.rest_signal(opp_rest_days: float) -> float` (`<= 5` gives `+1.0`, `>= 10` gives `-1.0`, else `0.0`, NaN gives NaN); `signals.ol_missing(injuries: pd.DataFrame, depth: pd.DataFrame, team: str, week: int) -> float` (sum over starters at depth rank 1-5 in `T`, `G`, `C`: Out 1.0, Doubtful 0.75, Questionable 0.25); `weather.roof_signal(roof: str | None) -> float | None` (`dome`/`closed` gives `0.0`; `outdoors`/`open` gives `0.5`; None gives None); `weather.fetch_forecast(lat: float, lon: float, kickoff: str) -> dict | None` returns None on any network error.

- [ ] **Step 1: Write failing tests**
  - `rest_signal(4) == 1.0`, `rest_signal(14) == -1.0`, `rest_signal(7) == 0.0`, NaN gives NaN.
  - `test_ol_missing_weights`: one starting T Out + one G Questionable = `1.25`; a non-starter (rank 6) Out adds `0`; empty injuries gives `0.0`.
  - `test_roof_signal`: dome gives `0.0`, outdoors gives `0.5`, None gives None (dome favors the offense, so no bonus for the defense).
  - `test_forecast_failure_returns_none`: patched `requests.get` raising `ConnectionError` yields None, no exception.
- [ ] **Step 2: Run;** Expected FAIL. **Step 3: Implement** (Open-Meteo hourly `temperature_2m,wind_speed_10m`; the stadium lat/lon table is a small dict in `weather.py`). Wind > 15 mph or temp < 25F adds `+0.5` to the weather signal when the forecast is available; otherwise the row notes "weather: roof only".
- [ ] **Step 4: Run;** Expected PASS.
- [ ] **Step 5: Show (real data)** whether the forecast fetch works; retry outside the sandbox if it returns None. Report honestly if it still fails.
- [ ] **Step 6: Commit** `feat: rest, OL injury, weather signals`.

### Task 8: Pipeline and CLI

**Files:**
- Create: `dstmatchup/pipeline.py`, `dstmatchup/cli.py`, `dstmatchup/__main__.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `pipeline.run(week: int | None, teams: list[str] | None, cfg: dict, data: dict[str, pd.DataFrame]) -> dict` with keys `week, ranked (DataFrame: defense, opponent, home, opp_implied_total, opp_turnovers_pg, opp_sack_rate, opp_qb_status, score, missing, contributions), played, byes`; `cli.main(argv: list[str] | None = None) -> int` with `--week`, `--teams`, `--season`, `--breakdown` flags. `python -m dstmatchup` runs it.

- [ ] **Step 1: Write failing tests** on small fixture dicts (no network)
  - `test_pipeline_ranks_weak_opponent_first`: opponent with the lowest implied total, most turnovers and a backup QB ranks first.
  - `test_pipeline_excludes_played_and_lists_byes`.
  - `test_teams_filter_normalizes`: `--teams lar,buf` filters to `LA`, `BUF`; `--teams ZZZ` exits non-zero with message listing valid codes.
  - `test_missing_signals_named_in_output`: a team with no QB info shows `qb_status` in its `missing`.
  - `test_cli_prints_table_and_breakdown` (capsys): header contains `Defense`, `Opp`, `Score`; with `--breakdown` each signal's contribution appears.
- [ ] **Step 2: Run;** Expected FAIL. **Step 3: Implement** (`tabulate` for output; default week from `current_week`; `--teams` falls back to `config.defaults.available_teams`; empty means all 32).
- [ ] **Step 4: Run** `python -m pytest tests -v`; Expected all PASS.
- [ ] **Step 5: Show (real data):** `python -m dstmatchup --breakdown`. Paste real output for Week 11 and list anything not verified.
- [ ] **Step 6: Commit** `feat: pipeline and CLI`.

### Task 9: Backtest

**Files:**
- Create: `dstmatchup/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `backtest.run(seasons: list[int], cfg: dict) -> pd.DataFrame` one row per defense-week with `season, week, team, score, actual_dst_points`, scoring each week using only data from earlier weeks; `backtest.signal_correlations(df: pd.DataFrame) -> pd.Series` (Spearman per signal vs `actual_dst_points`); `backtest.suggest_weights(corr: pd.Series) -> dict[str, float]` (clip negatives to 0, normalize to total 100); CLI `python -m dstmatchup backtest --seasons 2025 2026`.

- [ ] **Step 1: Write failing tests**
  - `test_no_lookahead`: scoring week W never reads rows with week >= W (fixture where week-W stats are poisoned with absurd values must not change the score).
  - `test_signal_correlations_known_values`: a perfectly monotone fixture gives `1.0`, reversed gives `-1.0`.
  - `test_suggest_weights_sum_and_nonnegative`.
- [ ] **Step 2: Run;** Expected FAIL. **Step 3: Implement.** **Step 4: Run;** Expected PASS.
- [ ] **Step 5: Show (real data):** run on 2025 + 2026 and report the per-signal correlations and the suggested weights vs the guessed ones. State sample size and whether 2025 data loaded. Do not overwrite `config.toml` weights without the user's say-so.
- [ ] **Step 6: Commit** `feat: backtest`.

### Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1:** Write setup (`python3 -m venv .venv`, `pip install -r requirements.txt`), usage (`--week`, `--teams`, `--breakdown`, `backtest`), config explanation (weights, z-score in plain words), data caveats (QB names unverified, proxy D/ST points, weather forecast status), and how to add a web page later (import `dstmatchup.pipeline.run`).
- [ ] **Step 2:** Follow the README from a clean venv; run the tests and the CLI. Expected: both work as written.
- [ ] **Step 3: Commit** `docs: README`.

---

## Self-Review

- **Spec coverage:** current-week default and `--week` (Tasks 2, 8); `--teams`/config list (8); breakdown and missing signals (3, 8); config-held weights and Yahoo values (1); cache (1); implied totals, QB status, signal 6, normalization/shrink/blend, home/away, rest, OL injuries, weather (2-7); played-games section and byes (2, 8); backtest and weight replacement (9); README (10); `.env`/`.gitignore` (1). Odds API and web UI are non-goals.
- **Types:** `zscore`, `shrink`, `score_teams`, `build_signal_frame` column names are identical across Tasks 3, 5, 8, 9; QB status values match the Global Constraints list.
- **Known soft spots flagged for the executor:** Task 4 fumble recoveries may not exist in team stats (report which columns are used); `build_signal_frame` also needs `qb_status`, `rest`, `ol_injuries` and `weather` columns once Tasks 6 and 7 land, so Task 7 and Task 8 extend it and Task 8's test asserts all eleven signal names appear in the weights.
