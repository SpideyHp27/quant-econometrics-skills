---
name: ts-decomposition
description: Decompose a time series into trend / seasonal / noise (STL with a shuffled-null calibration so pseudo-cycles don't masquerade as seasonality) and test the classic calendar effects — day-of-week and turn-of-month — with HAC p-values plus an honest fixed-window TOM backtest vs buy & hold. Use when asked "is there a calendar edge", "day-of-week effect", "turn-of-month", "seasonality", or "trend vs noise decomposition". Reports nulls proudly; windows come from the literature, never fitted.
---

# Time-series decomposition & calendar effects

Three questions, one tool, all guarded against the two classic ways this analysis lies:

1. **Decomposition (STL, Cleveland et al. 1990)** — trend vs seasonal vs residual variance shares.
   *Guard #1:* STL extracts *something* at any period you give it. So the tool also runs STL on a
   **shuffled-returns null** (a random walk with identical return distribution) and reports the
   **excess** seasonal share over that null. On real index futures the excess is ~0 — printed, not hidden.
2. **Calendar cells** — day-of-week (French 1980) and turn-of-month (Ariel 1987; Lakonishok & Smidt
   1988) mean returns, every cell printed with a Newey-West HAC p-value. No cherry-picking.
3. **Proof harness** — the classic TOM strategy (long only T−1..T+3 around month-end, flat otherwise,
   ~19% time in market) vs buy & hold, with a pre-registered IS/OOS split.
   *Guard #2:* the window is **fixed from the 1987/1988 literature, never fitted to your data** —
   fitting the window would be exactly the calendar-mining this skill exists to referee.

## How to run
```bash
uv run scripts/seasonality.py --csv prices.csv [--split 0.7] [--stl-period 21] [--json]
```
Any CSV/parquet with a date-ish + close column. Import for pipelines: `tom_mask(idx)`, `stl_shares`, `hac_p`.

## What it found on real CME data (see REPORT)
- **TOM is dead on modern index futures** — ES/NQ 2020-26: TOM days earn *less* than non-TOM days;
  harness OOS negative. The 1980s anomaly is arbed. The tool said so on both instruments (honest null).
- **Monday is alive** — ES +15.4bp (p=.0096), NQ +22.1bp (p=.0011), independently significant on both.
  A *finding* for further validation, not a validated edge (and on CME session days, "Monday" includes
  the weekend hold — it may be a weekend-premium in disguise).

## How it plugs into the research
- **Calendar gates for existing sleeves** — e.g. test whether OTE/MeanRev expectancy concentrates in
  the Monday/Wednesday cells before adding any day-filter (then validate OOS like everything else).
- **Referee for "seasonal" claims** — any month-of-year / day-of-week story from Discord or a video
  gets run through the same cells + HAC + null-calibrated STL before a minute is spent on it.
- Composes with `stationarity-tests` (structure) and `garch-volatility` (vol regime by calendar).

## Hard caveat
Calendar effects are the most mined patterns in finance; with 5 DOW cells × N instruments, some p<0.05
appears by luck alone. This tool controls what it can (HAC, fixed windows, all cells printed, null-
calibrated STL) — cross-instrument coherence and OOS survival are still on you before any deployment.

## References
Cleveland et al. (1990) STL, J. Official Statistics · French (1980) JFE · Ariel (1987) JFE ·
Lakonishok & Smidt (1988) RFS. Implementation: statsmodels STL + numpy; PEP 723.
