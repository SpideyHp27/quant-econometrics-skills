---
name: arima-forecast
description: Box-Jenkins ARIMA forecasting judged honestly against the random walk — order picked by ADF+AIC on train only, walk-forward 1-step forecasts with zero look-ahead, RMSE skill vs naive, directional accuracy vs 50%, Ljung-Box residual check, and a sign-trading proof harness with costs. Use for "forecast this series", "ARIMA", "does anything predict tomorrow's return". Expects and reports the honest null on liquid markets.
---

# ARIMA forecasting — judged against the random walk

ARIMA always FITS; the honest question is whether it FORECASTS better than "tomorrow = today"
(Fama's null). This tool: picks `d` by ADF and `(p,q)≤(3,3)` by AIC **on the train window only**;
walks forward 1-step across the untouched test window; judges by RMSE-vs-naive, directional
accuracy vs 50%, and Ljung-Box on residuals; then runs a costed sign-trading harness vs buy & hold.

```bash
uv run scripts/arima_forecast.py --csv prices.csv [--test-frac .3] [--refit 20] [--json]
```

## Validated on real CME NQ (see REPORT)
Verdict on NQ daily: **does NOT beat the random walk** (skill +0.44%, dir-acc 52.8%) — the expected
honest null, stated plainly. The sign harness echoed the market's weak negative lag-1 structure
(consistent with `randomness-tests` VR<1) — a lead for the gauntlet, not a claim.

## Plugs into the research
Consumes `d` from `stationarity-tests`; its residual Ljung-Box feeds `randomness-tests`; any
"forecasting model" claim (video, Discord, paper) gets this exact referee before a minute is spent on it.

## Caveat
Order selection is in-sample by necessity (AIC on train); the ONLY numbers that matter are the
test-window ones. Refs: Box & Jenkins (1970); statsmodels ARIMA; PEP 723.
