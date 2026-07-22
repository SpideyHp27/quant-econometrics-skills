---
name: stationarity-tests
description: Test whether a price/returns series is stationary or has a unit root (ADF + KPSS + Phillips-Perron fused into one verdict), find its integration order d (the "I" in ARIMA), and test two series for Engle-Granger cointegration (tradeable pairs / spreads). The pre-flight for every mean-reversion, ARIMA, or cross-asset regression model. Guards against spurious regression. Use whenever asked "is this mean-reverting or a random walk", "is this stationary", "are these two cointegrated / a pair". Descriptive tests, not a signal.
---

# Stationarity, unit roots & cointegration

The gate you must pass before trusting any time-series model. A series with a **unit root** (it is *I(1)* — a
random walk with drift) has no fixed level to revert to and an ever-growing variance. Two consequences bite
directly:

1. **A random walk is not mean-reverting.** Model an I(1) price as if it reverts to a level and your "edge" is
   the trend in disguise, not a real pull-back force.
2. **Regressing one I(1) series on another manufactures fake significance** — *spurious regression* (Granger &
   Newbold 1974). High R², tiny p-value, between two unrelated random walks. Any cross-asset predictor must
   clear a unit-root check first or its p-value is a lie.

This is the theory *under* the `garch-volatility` skill (GARCH models the variance of a *stationary* return
series; this skill establishes that returns are stationary and price is not).

## When to use this skill
- **Before any mean-reversion strategy** — is the thing you want to fade stationary, or a trending random walk?
- **Before any ARIMA model** — the test hands you the differencing order `d` directly.
- **Before any cross-asset / factor regression** — unit-root both sides so you don't publish a spurious edge.
- **Pairs / stat-arb** — two I(1) instruments that are *cointegrated* have a stationary, tradeable spread.

## Method (three tests, one fused verdict)
- **ADF** (Dickey-Fuller 1979) — null "has a unit root"; small p ⇒ **stationary**.
- **KPSS** (1992) — null "is stationary"; small p ⇒ **non-stationary**. *Opposite null on purpose:* when ADF and
  KPSS agree the read is far stronger than either alone.
- **Phillips-Perron** (1988, via `arch`) — corroborates ADF, robust to the heteroskedasticity + serial
  correlation financial series always carry.
- **Integration order `d`** — difference until ADF turns stationary (the ARIMA `d`).
- **Engle-Granger** (1987) — regress y on x, unit-root the spread; stationary spread ⇒ cointegrated ⇒ tradeable
  pair, with the OLS hedge ratio returned.

## How to run
```bash
uv run scripts/stationarity.py --csv data/ustec/USTEC_D1.csv
uv run scripts/stationarity.py --csv A.csv --coint B.parquet    # pairs test
uv run scripts/stationarity.py --csv prices.csv --json
```
Handles the project's `date,time,…` MT5 exports and the Databento parquets. Import for pipelines:
`combined_verdict(series)`, `integration_order(prices)`, `cointegration(y, x)`.

## How it plugs into the research
- **MeanRev_NDX reality check** — USTEC price is **I(1)**; the strategy reverts to a moving SMA, not a stationary
  level. Formalises the "it's leveraged beta, size it down" verdict with a hard test.
- **Cross-asset predictors** — unit-root ChBVIP↔DXY / MeanRev↔VIX before trusting the regression.
- **Feeds `arima-forecast`** — the `d` this returns is the differencing order that skill consumes.
- **New pairs sleeve** — a cointegration screen surfaces market-neutral spreads (the counter-cyclical direction
  the mono-directional bundle lacks).

## Hard caveat
Stationarity is necessary, not sufficient — a stationary spread can be too small to trade after costs, and
cointegration can break (regime change, index reconstitution). These tests say *whether a mean exists to revert
to*; they do not size the trade. Descriptive, not a signal.

## References
- Dickey & Fuller (1979), JASA. KPSS (1992), J. Econometrics. Phillips & Perron (1988), Biometrika.
- Engle & Granger (1987), Econometrica (cointegration). Granger & Newbold (1974), J. Econometrics (spurious regression).

## Dependency
`statsmodels` (ADF/KPSS/coint) + `arch` (Phillips-Perron), `numpy`, `pandas`, `scipy` — declared inline via PEP 723.
