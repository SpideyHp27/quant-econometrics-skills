# Skill Report — `ts-decomposition`

**Bundle:** Quant Econometrics Skills (skill 4 built) · **Built:** 2026-07-23
**Theory:** STL (Cleveland 1990); DOW effect (French 1980); TOM effect (Ariel 1987, Lakonishok & Smidt 1988).

## What it does
STL trend/seasonal/noise decomposition **calibrated against a shuffled-returns null** (STL finds a pseudo-cycle in any random walk — the null makes that visible instead of publishable), + all day-of-week and turn-of-month cells with Newey-West HAC p-values, + the classic fixed-window TOM strategy backtested vs buy & hold with a pre-registered IS/OOS split. Windows from the literature, never fitted.

## What it does for the research
The referee for every calendar claim — and a scanner for where existing sleeves' expectancy actually lives in the week. Two guards are the point: null-calibrated STL (kills fake seasonality) and fixed windows (kills calendar-mining).

## Validation (real CME futures, back-adjusted, 2020-01 → 2026-06, 1,661 sessions each)

**Decomposition — the null calibration earning its keep (NQ):**
```
STL shares: trend 1.2% · "seasonal" 31.8% · resid 67.0%
shuffled-null "seasonal": 31.5%  →  EXCESS = 0.3%  ⇒  no genuine monthly cycle
```
Without the null, "31.8% seasonal" would have been reported as structure. It's artifact.

**Turn-of-month — the famous anomaly is DEAD here (honest null, both instruments):**
| | TOM days (18.8% of days) | non-TOM days | TOM harness OOS |
|---|---|---|---|
| ES | +3.9 bp/day (p=.45) | +4.8 bp/day | **−1.8%/yr, Sharpe −0.25** |
| NQ | +4.8 bp/day (p=.46) | +6.4 bp/day | **−1.2%/yr, Sharpe −0.08** |

TOM days earn *less* than ordinary days, 2020-26. The pre-registered verdict logic printed "weak/no calendar edge (honest null)" on both — the 1980s effect is arbed away on index futures.

**Day-of-week — a live finding (NOT yet a validated edge):**
| Day | ES (bp/day, HAC p) | NQ (bp/day, HAC p) |
|---|---|---|
| **Mon** | **+15.4 (p=.0096)** | **+22.1 (p=.0011)** |
| Tue | +1.9 (.71) | +3.6 (.55) |
| **Wed** | **+10.6 (p=.032)** | **+15.5 (p=.017)** |
| Thu | −7.1 (.34) | −8.5 (.32) |
| Fri | +2.5 (.65) | −2.2 (.73) |

Monday is independently significant on both instruments (NQ survives a ×10 multiple-testing correction). Two honesty flags before anyone trades it: (1) on CME session-days "Monday" = Sunday 18:00 → Monday 17:00, so this may be a **weekend-hold premium** rather than a Monday effect; (2) it needs its own OOS regime check. Logged as a research lead, not an edge.

## Caveat
Calendar cells are the most mined statistics in finance. This tool prints every cell, fixes every window, HAC-corrects every p, and null-calibrates the decomposition — but coherence and out-of-sample survival remain the user's burden before deployment.

## Files
- `../../scripts/seasonality.py` — tool (`--json`; importable `tom_mask`/`stl_shares`/`hac_p`).
- `SKILL.md` — skill definition.
