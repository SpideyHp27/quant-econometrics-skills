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

## Caveat
Stationarity is necessary, not sufficient. A stationary spread can be too small to trade after costs, and cointegration can break on regime change. The tool says *whether a mean exists*; it does not size the trade.

## Files
- `../../scripts/stationarity.py` — tool (`--coint`, `--json`; importable `combined_verdict` / `integration_order` / `cointegration`).
- `SKILL.md` — skill definition.

## References
Dickey & Fuller (1979) JASA · KPSS (1992) J. Econometrics · Phillips & Perron (1988) Biometrika · Engle & Granger (1987) Econometrica · Granger & Newbold (1974) J. Econometrics.
