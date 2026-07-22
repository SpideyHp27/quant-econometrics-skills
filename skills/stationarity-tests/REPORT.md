# Skill Report — `stationarity-tests`

**Bundle:** Quant Econometrics Skills (skill 2 of 6) · **Built:** 2026-07-22
**Theory:** Econometrics KB hubs *stationarity* + *unit-root* → `D:\MEP` course. **Sibling:** `garch-volatility` (this is the theory under it).

## What it does
Runs three unit-root / stationarity tests and fuses them into one verdict:

| Test | Null hypothesis | Small p-value means |
|---|---|---|
| **ADF** (Dickey-Fuller 1979) | has a unit root | stationary |
| **KPSS** (1992) | is stationary | **non**-stationary |
| **Phillips-Perron** (1988, `arch`) | has a unit root | stationary |

ADF and KPSS carry **opposite nulls**, so their *agreement* is a much stronger read than either alone. It also reports the **integration order `d`** (the "I" in ARIMA) and an **Engle-Granger cointegration** test for two series (stationary spread ⇒ tradeable pair, with the OLS hedge ratio).

## What it does for the research
The gate between a real edge and a statistical mirage:
- **Kills spurious regression** — two random walks regress with high R² and tiny p. Every cross-asset predictor must unit-root both sides first.
- **Reality-checks mean-reversion** — a strategy can only revert to a mean if a mean exists. I(1) price ⇒ the P&L is trend/beta, not reversion.
- **Feeds ARIMA** — hands `d` straight to `arima-forecast`.
- **Opens a pairs/stat-arb archetype** — a cointegration screen surfaces market-neutral spreads, the counter-cyclical direction the current bundle lacks.

## Validation (real project data)
**USTEC daily** (live MeanRev_NDX instrument, 2150 bars, 2018-01-02 → 2026-04-29), cointegrated vs **CME NQ daily** (Databento):

```
[LEVELS]  raw price
  ADF              p=0.9873  -> NON-stationary
  KPSS             p=0.0100  -> NON-stationary
  Phillips-Perron  p=0.9854  -> NON-stationary
  VERDICT: NON-STATIONARY (unit root, I(1)) -- all three agree

[LOG-RETURNS]
  ADF              p=0.0000  -> stationary
  KPSS             p=0.1000  -> stationary
  Phillips-Perron  p=0.0000  -> stationary
  VERDICT: STATIONARY (I(0)) -- all three agree

[INTEGRATION ORDER]  d = 1   (price is I(1); ARIMA d-term = 1)

[COINTEGRATION]  USTEC ~ NQ  | n=1634
  coint_t=-6.824  p=0.0000  hedge_ratio=0.9905  -> COINTEGRATED (spread tradeable)
```

**Finding:** USTEC price carries a unit root — it is **I(1)**, a random walk with drift. This is the MeanRev_NDX forensic conclusion made formal: the live strategy is **not** reverting to a stationary level, it is leveraged bull-beta. Cointegration vs NQ is the expected sanity check — same underlying, hedge ratio ≈ 1.00, spread stationary.

## Functional payoff — pairs / stat-arb backtest (`pairs_backtest.py`)
The diagnostic earns its keep by driving a real strategy. `pairs_backtest.py` builds a market-neutral spread trade (trailing-window hedge ratio → walk-forward z-score → long/short spread), models costs, and reports honest P&L — zero look-ahead. Screened + backtested across 6 pairs (futures = Databento daily 2018→26; ETFs = yfinance 2012→26):

| Pair | Coint p | Backtest Sharpe | PF | Verdict |
|---|---:|---:|---:|---|
| YM ~ ES (Dow/S&P) | **0.0065** ✓ | +0.26 | 1.16 | cointegrated but marginal (no robust plateau) |
| GLD ~ GDX (gold/miners) | 0.062 ~ | +0.37 | 1.24 | borderline, best of the set — still sub-gate |
| ES ~ NQ | 0.21 | −0.28 | 0.84 | not cointegrated → loses |
| GC ~ SI (gold/silver) | 0.25 | −0.82 | 0.53 | not cointegrated → loses badly (61% DD) |
| EWA ~ EWC (Chan's pair) | 0.47 | −0.37 | 0.81 | decoupled post-2015 → loses |
| KO ~ PEP | 0.91 | −0.02 | 0.98 | not cointegrated → flat |

**Finding (honest):** the cointegration screen is **predictive of tradeability** — the only two pairs with PF>1 are the only two that (borderline-)cointegrated; every clearly-non-cointegrated pair loses money. That is the screen doing its job as a filter. **But no pair clears the deploy gate (Sharpe ≥ 1.0):** static-cointegration daily pairs on liquid instruments are marginal-to-null. The famous textbook pairs (EWA/EWC) held only over their original windows and have since drifted out of cointegration — a real lesson in relationship non-stationarity, not a tooling failure. The tool's value is that it **proves this honestly and refuses to over-trade** — a reusable stat-arb generator + built-in honesty gate for richer universes (sector baskets, crypto, intraday) where cointegrated pairs are denser.

## Caveat
Stationarity is necessary, not sufficient. A stationary spread can be too small to trade after costs, and cointegration itself drifts (the relationships are non-stationary — see EWA/EWC above). The tool says *whether a mean exists and whether trading it paid*; it does not promise the relationship persists out of sample.

## Files
- `../../scripts/stationarity.py` — tool (`--coint`, `--json`; importable `combined_verdict` / `integration_order` / `cointegration`).
- `SKILL.md` — skill definition.

## References
Dickey & Fuller (1979) JASA · KPSS (1992) J. Econometrics · Phillips & Perron (1988) Biometrika · Engle & Granger (1987) Econometrica · Granger & Newbold (1974) J. Econometrics.
