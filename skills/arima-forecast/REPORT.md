# Skill Report — `arima-forecast`
**Built:** 2026-07-23 · **Theory:** Box & Jenkins 1970.

## Validation — real CME NQ daily (back-adjusted; test = 2024-07 → 2026-06, 498 bars, walk-forward)
```
order (train-only AIC): ARIMA(2,1,2)
RMSE  0.013386 vs naive 0.013445  -> skill +0.44%   (needs >1% to matter)
directional accuracy 52.8%        (benchmark 50%)
residual Ljung-Box p=0.265        (captured the linear structure it claimed)
harness: sign-trading Sharpe 1.14 vs B&H 0.65  (single window — a LEAD, not a result)
VERDICT: does NOT beat the random walk (the expected honest null on liquid markets)
```
The verdict logic held the line exactly as designed: tiny RMSE skill + sub-53% direction = null,
even though the harness looked pretty. That harness echo coheres with NQ's VR<1 reversion flavor
(`randomness-tests`) and the lag-1 cross-predictor (`regression-diagnostics`) — one phenomenon seen
by three instruments, already deployed in this research program as the MeanRev family.

## Accuracy audit (synthetic ground truth) — 2/2
Planted AR(1) φ=0.6: order picker chose AR terms; walk-forward beat naive by **+30.1% RMSE skill**.
The tool finds real structure when it exists and refuses to invent it when it doesn't.

## Caveat
~30–40s for a 200–500-step walk-forward (statsmodels refits). Use `--refit 40+` on long tests.
