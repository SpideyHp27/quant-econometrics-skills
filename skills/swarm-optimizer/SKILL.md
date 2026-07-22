---
name: swarm-optimizer
description: Optimize portfolio weights across multiple strategies with Particle Swarm Optimization (PSO) wrapped in an anti-overfit gauntlet — train/test split the optimizer cannot see past, Deflated Sharpe Ratio charged for every candidate evaluated, and an equal-weight benchmark it must beat OUT OF SAMPLE or the tool tells you to use equal weight. Use for allocating across a strategy bundle, re-weighting a live book, or any bounded search where a stronger optimizer would otherwise just find overfits faster. Swarm intelligence, honestly caged.
---

# Swarm optimizer — PSO portfolio weights with the gauntlet welded on

Particle Swarm Optimization (Kennedy & Eberhart 1995): a swarm of candidate weight vectors flies through
allocation space, each pulled toward its personal best and the swarm's global best. It finds optima
*fast* — which in finance is the hazard, not the feature: **a stronger optimizer finds overfits faster.**
This project proved that the hard way (80-strategy sweep → DSR → zero real survivors). So this skill
never reports an in-sample answer:

## The four cage bars
1. **The swarm sees ONLY the train window** (first 70% by default). The test window stays untouched
   until optimization is finished.
2. **Every evaluation is counted** (particles × iterations, typically ~5,000 trials) and the test Sharpe
   is **deflated for all of them** — Deflated Sharpe Ratio (Bailey & López de Prado 2014): the printed
   "luck bar" is the Sharpe the best of N random tries would show; DSR = probability the result clears it.
3. **Equal-weight is the benchmark.** If PSO doesn't beat equal-weight on the TEST window (Sharpe up,
   MaxDD not >25% worse), the verdict says **"USE EQUAL WEIGHT"** — the honest null is a feature.
4. **Concentration cap** (`--wmax`, default 0.35) — no single strategy can dominate, enforced by proper
   iterative simplex projection.

## How to run
```bash
uv run scripts/swarm_optimizer.py --dir <folder-of-trade-csvs> --glob "0[1-7]_*.csv"
uv run scripts/swarm_optimizer.py --wide daily_returns.csv --json
# options: --equity 100000 --test-frac 0.30 --wmax 0.35 --particles 40 --iters 120 --lambda-dd 0.02
```
Input: one trade CSV per strategy (needs `date` + `profit` columns — the project's standard export), or
a wide daily-returns matrix. Output: weights, train/test Sharpe & MaxDD for PSO vs equal-weight, trial
count, luck bar, DSR, and a plain-language verdict.

## How it plugs into the research
- **The live bundle's weights** are the real target: 7-10 strategies on the book, currently sized by
  per-strategy risk-percent convention. This gives a validated, DSR-checked re-weighting proposal.
- Composes with `garch-volatility` (vol-target the *total* book after weighting) and with the project's
  portfolio MC (run `portfolio_correlated_mc` on the proposed weights before any deploy).

## Hard caveats
- **Weights-level honesty ≠ strategy-level honesty.** The train/test split is out-of-sample for the
  *weights*, but the input backtests were themselves developed on this history. Garbage in, optimally
  weighted garbage out.
- **A daily-P&L Sharpe optimizer cannot see intraday tails.** In validation it assigned the max-cap
  weight to a strategy whose *backtest* is beautiful but whose live record includes a single −14.75%
  swing-tail day (TT_NDX). Optimizer output is a PROPOSAL for the human + MC gauntlet, never a deploy order.
- PSO is a heuristic: reruns with different seeds give slightly different weights (the plateau matters,
  not the decimals).

## References
Kennedy & Eberhart (1995), *Particle Swarm Optimization*, IEEE ICNN. · Bailey & López de Prado (2014),
*The Deflated Sharpe Ratio*, J. Portfolio Management. · Implementation: pure numpy/scipy, PEP 723.
