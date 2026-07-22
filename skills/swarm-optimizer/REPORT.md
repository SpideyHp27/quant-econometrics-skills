# Skill Report — `swarm-optimizer`

**Bundle:** Quant Econometrics Skills (skill 3 built) · **Built:** 2026-07-22
**Theory:** Kennedy & Eberhart 1995 (PSO) + Bailey & López de Prado 2014 (Deflated Sharpe Ratio).
**Origin:** owner shared a "Swarm Intelligence in Financial Market Analysis" article (PSO/ACO); assessment was that PSO is genuinely useful here for exactly ONE thing — portfolio weights — and only with the anti-overfit gauntlet welded on. This is that tool.

## What it does
A particle swarm searches the weight simplex (long-only, per-strategy cap, weights sum to 1) to allocate across a bundle of strategies. The cage: optimizer sees only the train window; the test window delivers the verdict; every candidate evaluated is charged against the result via DSR; equal-weight is the benchmark it must beat out of sample, else the tool says "use equal weight."

## What it does for the research
The live book runs 7–10 strategies sized by convention (risk-percent per strategy). This produces a validated re-weighting *proposal* with the same honesty machinery as the rest of the project (DSR was already the project standard for sweep results). It also completes the "swarm intelligence" question with a working artifact instead of an opinion.

## Validation (real project data)
Input: the 7 core bundle strategies' trade CSVs (`portfolio_csvs_for_aristhrottle_2026_05_06`, 2018-01 → 2026-04, 2,172 trading days). Split: train → 2023-10, test 2023-10 → 2026-04 (untouched). 4,840 trials evaluated; luck bar (expected max trial Sharpe) 0.54.

```
                train Sharpe   train MaxDD   TEST Sharpe   TEST MaxDD
equal_weight            1.31        11.18%          2.03       11.25%
pso                     2.25         0.55%          2.95        0.60%

weights: TT_NDX 0.350(cap) · MeanRev_NDX 0.214 · PellaRB_UJ 0.178 · ChBVIP_UJ 0.139
         · ChBVIP_XAU 0.114 · Strat11_XAU 0.005 · Strat11_UJ 0.000
DSR (vs best-of-4840 luck): 1.000  → VALIDATED out-of-sample at the weights level
```

**Findings, honestly read:**
- PSO beats equal-weight out of sample (Sharpe 2.95 vs 2.03) with far lower drawdown at these baseline sizings, and the DSR says that's not trial luck. The *machinery* works.
- **The proposal itself must NOT be deployed as-is.** The optimizer put the maximum allowed weight on TT_NDX — whose *backtest* daily P&L is superb but whose *live* record includes a single −14.75% swing-tail day and a detachment. A daily-P&L Sharpe cannot see intraday tails; this is the tool demonstrating its own stated caveat on real data.
- It correctly zeroed the two weakest sleeves (Strat11 eval variants ≈ 0), which matches the project's standalone verdicts.
- Weights-level OOS ≠ strategy-level OOS: the input backtests were developed on this same history.

## Caveat
This is a proposal generator for the human + Monte-Carlo gauntlet (`portfolio_correlated_mc`), never a deploy order. Heuristic optimizer: the plateau matters, not the third decimal.

## Files
- `../../scripts/swarm_optimizer.py` — tool (`--dir|--wide`, `--json`; pure numpy/scipy, PEP 723).
- `SKILL.md` — skill definition.
