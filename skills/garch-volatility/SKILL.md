---
name: garch-volatility
description: Forecast market volatility (magnitude, one-day-ahead) with a walk-forward GARCH(1,1) model, classify the vol regime (calm/normal/storm), and derive vol-targeted position-size multipliers. Use for position sizing, "how much should I put on", regime gating of strategies, portfolio vol-targeting, and risk analysis. Works on any CSV/parquet with a date + close column. Answers "how much" — never "which way".
---

# GARCH volatility forecasting & vol-targeted sizing

Engle's ARCH (1982) / Bollerslev's GARCH (1986) — the framework that won the **2003 economics Nobel** and
that real risk desks run daily. It models **volatility clustering** (calm begets calm, storms beget storms)
and produces a **one-day-ahead forecast of the *size* of the next move**. It does **NOT** predict direction —
pair it with a directional edge, never use it alone.

## When to use this skill
- **Position sizing** — scale a strategy inversely to forecast vol: bigger when calm, smaller when stormy. The
  single most reliable use, and the archetype-correct fix for no-stop mean-reversion (it blows up *precisely*
  when vol spikes — GARCH sizes it down right then, without touching the entry logic).
- **Regime gating** — turn a strategy down/off in the "storm" regime.
- **Portfolio vol-targeting** — hold total book vol near a target (e.g. 10–15% annualized).
- **Risk analysis** — a forward-looking exposure estimate that beats a fixed-lookback stdev.

## Method (zero look-ahead)
GARCH(1,1), Student-t (fat tails), on % log-returns. Expanding window, walk-forward: params re-estimated
periodically on all data *before* day t; between refits the variance recursion rolls forward on **realized**
returns known at forecast time. Every forecast for day t uses only data that existed before day t.

## How to run
```bash
uv run scripts/garch_forecast.py --csv prices.csv [--target-vol 0.15] [--backtest]
uv run scripts/garch_forecast.py --csv prices.csv --json
```
Input: any CSV/parquet with a date-ish column + a close column. `--backtest` compares buy&hold vs vol-targeted
(no-lookahead) equity. Import for pipelines: `walk_forward_garch`, `position_size`, `classify_regime`.

## Outputs
`sigma_daily` / `sigma_ann` (1-day-ahead vol), `regime` (calm/normal/storm by trailing-year percentile),
`size_mult` = `target_vol / forecast_vol` capped to **[0.25×, 2.0×]`.

## How it plugs into the research
- **Size the live strategies** — MeanRev_NDX runs ~3.7× leverage with no stop; GARCH sizing scales exposure by
  forecast vol and never touches the entry (the thing that kills mean-reversion edges). Proven on real NQ:
  vol-targeting cut MaxDD **38.5% → 24.2%** with higher Sharpe.
- **Regime layer** for any top-down entry engine; **portfolio overlay** for the multi-strategy book.

## Hard caveat
Forecasts the **magnitude** of the next move, never the direction. A risk/sizing tool, not a signal — on its own
it makes no money; it makes an existing directional edge survivable and smoother.

## References
- R. F. Engle (1982), "Autoregressive Conditional Heteroscedasticity…", *Econometrica*.
- T. Bollerslev (1986), "Generalized ARCH", *Journal of Econometrics*.
- Implementation on `arch` (Kevin Sheppard). Structural template: milesdeutscher/garchmethod.

## Dependency
`arch`, `numpy`, `pandas` (declared inline via PEP 723 — `uv run` resolves them).
