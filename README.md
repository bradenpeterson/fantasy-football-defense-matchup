# D/ST Matchup Ranker

A command-line tool that ranks every NFL defense (D/ST) for one week by how
easy its opponent looks. Use it to decide which defense to start or pick up in
Yahoo. The top of the list is the best matchup this week; the bottom is the
worst.

It reads public NFL data (schedules, betting lines, team stats, injuries,
play-by-play) through the `nflreadpy` package. You do not need
any accounts or API keys.

## What the output looks like

Below is a trimmed excerpt from `--breakdown` run on 2026-11-20 with the
recent-starter QB rule (`qb_recent_games = 2`), which is Week 11 of the 2026
season. Your numbers will change as the season moves on.

```
Defense    Opp    H/A    Opp Implied    Opp TO/g    Opp Sack%    Opp QB    Score    implied_total    qb_status    turnovers    sack_rate    dst_pts_allowed    opp_ppg    own_defense    home_away    rest
---------  -----  -----  -------------  ----------  -----------  --------  -------  ---------------  -----------  -----------  -----------  -----------------  ---------  -------------  -----------  ------
KC         ARI    H      12.0           1.0         8.5%         backup    77.3     +0.57            +0.39        -0.04        +0.06        +0.06              +0.04      -0.02          +0.03        +0.01
BUF        MIA    H      14.0           1.0         10.2%        backup    77.3     +0.46            +0.39        -0.04        +0.13        +0.07              +0.03      +0.02          +0.03        +0.01
PIT        PHI    A      17.0           1.0         9.7%         backup    70.4     +0.30            +0.39        -0.04        +0.11        +0.05              +0.05      +0.04          -0.03        -0.04
DEN        LV     H      15.0           1.2         5.5%         backup    69.0     +0.41            +0.39        +0.01        -0.07        -0.03              -0.00      +0.02          +0.03        +0.01
...
TEN        DAL    A      31.5           0.8         4.6%         starter   27.9     -0.48            -0.09        -0.09        -0.11        -0.07              -0.05      +0.02          -0.03        +0.01

Already played this week:
  IND @ HOU

On bye:
  ATL, CLE, GB, LA, NE, SEA
```

How to read the columns:

- **Defense / Opp / H/A**: the defense you are ranking, the team it plays, and whether that defense is at home.
- **Opp Implied**: the points the opponent is expected to score, from the betting total and spread. Lower is better for the defense.
- **Opp TO/g**: the opponent's turnovers per game (interceptions and lost fumbles).
- **Opp Sack%**: the share of the opponent's pass plays that end in a sack.
- **Opp QB**: the opponent's quarterback status (see "Data caveats").
- **Score**: 0 to 100, where 50 is an average matchup and higher is better.
- **The columns after Score** (shown with `--breakdown`) show how much each signal pushed the score up (`+`) or down (`-`). Use them to see *why* a defense is high or low. They are in z-score units (each is the signal's weighted z); multiply by 25 to get score points. `n/a` means the signal is missing for that defense, so it contributed nothing.
- **missing: ...** under a defense means some signals had no data for that team. The score was calculated without them (see "How the score works").

## Setup

You need Python 3.11 or newer. The project was built and tested on Python 3.12.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On the first run the tool downloads the data it needs (this can take a few
minutes) and saves it in a `.cache` folder. Later runs reuse that cache.

No API keys are needed. The project uses nflverse data only. The Odds API and
the Yahoo Fantasy API are not used in this version. `.env.example` is there
for keys that future versions might need. A `.env` file is gitignored, so
secrets you put there will not be committed.

## Usage

Run everything from the project folder, using the virtual environment's Python.

```
.venv/bin/python -m dstmatchup                      # ranking for the next unplayed week
.venv/bin/python -m dstmatchup --week 12            # a specific week
.venv/bin/python -m dstmatchup --teams BUF,DEN,kc   # only these defenses (any case; LAR is accepted for LA)
.venv/bin/python -m dstmatchup --season 2025        # a different season (default 2026)
.venv/bin/python -m dstmatchup --breakdown          # add the per-signal contribution columns
```

- **Default week**: the earliest week that still has an unplayed game. In Week
  11 of 2026, that is week 11.
- **`--teams`**: shows only the defenses you list, separated by commas. Games
  already played this week still appear in the "Already played" section if a
  listed team was in them. You can also set a default list in the
  `available_teams` line of `config.toml`; `--teams` overrides it.
- **Cache note**: the cache folder `.cache` is created in the folder you run
  the command from. Run from the project folder to reuse the existing cache.

### Backtest

```
.venv/bin/python -m dstmatchup backtest --seasons 2025 2026
```

The backtest checks how well each signal predicted real D/ST fantasy points
in past weeks. For each week it scores the defenses using only information
from earlier weeks, then compares that score with the points the defense
actually produced (using the same Yahoo-style scoring as the ranking).

It prints:

- A sample size (on 2026-11-20: 654 defense-weeks across 2025 and 2026).
- A correlation for each signal and for the overall score. A correlation near
  +1 means "higher signal, more points"; near 0 means no clear link. On
  2026-11-20 (with `qb_recent_games = 2`) the overall score had a correlation of
  +0.318, the implied total +0.324, the sack rate +0.210, and the QB status signal
  +0.008 (it was +0.091 before the recent-starter rule).
- A table of suggested weights next to your current weights. The suggestions
  come from the positive correlations only.

Read the results with caution:

- The sample is small and the correlations are noisy.
- The QB signal uses the kickoff starter: the QB the schedule names for the game.
  That is known at kickoff but not at earlier lineup deadlines. In the 2025 data it
  matches the team's actual top passer in about 96% of team-games; the mismatches
  are mostly starters hurt or pulled during the game. The QB signal is therefore
  mildly optimistic here.
- The implied total uses closing lines: the nflverse total and spread, which are
  set just before kickoff. A mid-week decision would not have them, so the implied
  total's correlation (the strongest one here) is flattered.
- The suggested weights add up to 100 while `config.toml` adds up to 106.
- The suggested weights are not proof that a signal predicts points. Treat them
  as ideas to test over more weeks.

The backtest **never writes `config.toml`**. You change weights yourself, if
you decide to.

### Running the tests

```
.venv/bin/python -m pytest
```

The tests use small made-up data sets and do not download anything.

## How the score works

Each defense gets a score built from 9 signals. Each signal is a number where
higher means a better matchup for the defense.

| Signal | Starting weight | Plain meaning |
| --- | --- | --- |
| `implied_total` | 30 | The opponent's expected points (lower is better) |
| `qb_status` | 20 | The opponent's QB is a backup or out, which helps the defense |
| `turnovers` | 15 | The opponent turns the ball over often |
| `sack_rate` | 15 | The opponent's QB gets sacked often (sacks per pass play) |
| `dst_pts_allowed` | 10 | Defenses have scored well against this opponent |
| `opp_ppg` | 5 | The opponent scores few points per game |
| `own_defense` | 5 | This defense has scored well itself |
| `home_away` | 3 | The defense is at home |
| `rest` | 3 | The opponent has short rest (bonus) or long rest (penalty) |

The steps, in plain words:

1. **Use recent and season data.** Each stat is a blend: half from the whole
   season so far and half from the last 4 games. The blend is set by
   `blend_recent` and `last_n` in `config.toml`.
2. **Compare each signal to the league.** Every signal is turned into a
   z-score. A z-score says how far a team is from the league average, measured
   in typical spreads: 0 is average, +1 is one typical spread better. The league
   is every team playing that week, including games already final, so a team's
   z does not change when other games in the week finish.
3. **Shrink small samples.** A z-score based on two games is less trustworthy
   than one based on ten. So five signals are pulled toward 0 by a factor of
   games / (games + 4): turnovers, sack rate, D/ST points allowed and opponent
   points per game (by the opponent's games), and own defense (by the defense's
   games). With 4 games the stat counts half as much. The other signals are not
   shrunk.
4. **Add up the weighted signals.** The score is the average of the signals,
   weighted by the table above.
5. **Handle missing signals.** If a signal has no data for a team (for example,
   no betting line or an unknown QB), that signal is dropped for that team. The
   other weights are scaled up so they still add to a full total. The dropped
   signal names are printed under the defense as `missing: ...`. A missing
   value is never filled with 0 or treated as average. One case to know: a
   signal with the same value for every team that week has no spread, so every
   team gets z = 0 for it. That is "no difference", not "missing", so it is not
   listed.
6. **Show a 0 to 100 score.** The weighted average is turned into a display
   score: 50 is average, and the number goes up or down by 25 points per typical
   spread. Scores are capped between 0 and 100.

The weights are starting guesses, not tested values. Change them in the
`[weights]` section of `config.toml`. The other settings there control the
blend, shrinkage, display scale, and the Yahoo scoring values used for the
D/ST points proxy and the points-allowed tiers. The weights do not need to add
to 100; the tool rescales them.

## Data caveats

Please read these before trusting a single number:

- **QB status can still be wrong for a QB who took over mid-season.** The tool
  calls the QB named in the schedule a `starter` if that QB has the most
  dropbacks this season, or if that QB led dropbacks in each of the team's last
  2 games (`qb_recent_games` in `config.toml`). Otherwise the named QB shows as
  `backup`, which gives the defense a bonus. A new starter who has led only one
  recent game, or who is not the season leader and did not lead the last two
  games, still shows as `backup`, so the bonus can be too high. The "Opp QB"
  column shows this. Check it against the news before you act on it.
- **D/ST points allowed is a proxy.** The tool does not have official Yahoo D/ST
  scores. It calculates them from team stats (sacks, turnovers, touchdowns,
  blocked kicks, points allowed) using the Yahoo values in `config.toml`. Small
  differences from Yahoo's own numbers are expected.
- **The 0 to 100 weights are starting guesses.** The backtest is a small
  sample, so its suggested weights are only ideas to test (see Backtest).
- **Betting lines** come from nflverse. How up to date they are has not been
  verified. Check the lines at your sportsbook if the numbers look stale.

## Adding a local web page later

The scoring code does not touch the network. To build a page on top of it,
load the data the same way the command-line tool does, then call the pipeline:

```python
from dstmatchup import pipeline
from dstmatchup.config import load_config

cfg = load_config()
data = {
    "schedule": ...,    # pandas frames from dstmatchup.loaders
    "team_stats": ...,
    "pbp": ...,
    "injuries": ...,
}
result = pipeline.run(week, None, cfg, data)   # teams=None ranks all defenses
result["ranked"]   # the ranking table
result["played"]   # games already played this week
result["byes"]     # teams on bye
```

`dstmatchup/cli.py` shows how the data is loaded (`_load_data`). Copy that step for a web app.

## Project layout

- `dstmatchup/__main__.py`: lets `python -m dstmatchup` run the command.
- `dstmatchup/cli.py`: reads the options, loads data, and prints the table.
- `dstmatchup/pipeline.py`: builds one week's ranking from data frames, with no network access.
- `dstmatchup/signals.py`: calculates the raw signal numbers for each team from stats.
- `dstmatchup/scoring.py`: z-scores, shrinkage, and the 0 to 100 score.
- `dstmatchup/qb.py`: works out each opponent's QB status.
- `dstmatchup/dst_points.py`: calculates the Yahoo-style D/ST points proxy from games and stats.
- `dstmatchup/schedule.py`: current week, betting lines, matchups, and byes.
- `dstmatchup/loaders.py`: downloads the nflverse data, through the cache.
- `dstmatchup/cache.py`: stores downloads as parquet files (current season refreshes every 6 hours).
- `dstmatchup/config.py`: loads and checks `config.toml`.
- `dstmatchup/teams.py`: team codes and aliases such as LAR to LA.
- `dstmatchup/backtest.py`: the walk-forward backtest and suggested weights.
- `config.toml`: weights, scoring settings, and Yahoo values.
- `tests/`: unit tests for each module, run with pytest.
