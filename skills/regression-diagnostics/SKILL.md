---
name: regression-diagnostics
description: OLS with the full Gauss-Markov referee battery — Breusch-Pagan/White heteroskedasticity, Durbin-Watson/Breusch-Godfrey serial correlation, VIF multicollinearity, Jarque-Bera, Ramsey RESET, Chow break — and the verdict column that matters, naive p vs HAC p side by side. Use for "is this regression edge real", "does X predict Y", factor and cross-asset predictor claims. If significance only exists in the naive column, the edge is an artifact.
---

# Regression referee — naive p vs HAC p

Financial data violates OLS assumptions by default, which makes naive p-values LIE small. This
prosecutes any "X predicts Y" claim: fits the OLS, runs the full assumption battery, and prints
every coefficient with **naive p and Newey-West HAC p side by side** — significance that vanishes
under HAC is an artifact, and the verdict says exactly that.

```bash
uv run scripts/regression_diag.py --y-prices A.parquet --x-prices B.parquet --lag 1   # predictor test
uv run scripts/regression_diag.py --csv data.csv --y target --x col1,col2 [--json]
```

## Validated on real CME data (see REPORT)
NQ_ret ~ ES_ret(lag-1): **β=−0.172, HAC p=0.0028 — survives the referee** (yesterday's ES move
negatively predicts today's NQ: cross-serial reversion, the phenomenon the MeanRev family trades).
The battery simultaneously flagged heteroskedasticity + serial correlation — exactly why the HAC
column was needed to render an honest verdict.

## Caveat
Surviving HAC ≠ tradeable (R² was 2.2%) — it means the claim has earned a gauntlet, not a deploy.
Refs: White 1980 · Breusch & Pagan 1979 · Newey & West 1987 · Chow 1960. statsmodels; PEP 723.
