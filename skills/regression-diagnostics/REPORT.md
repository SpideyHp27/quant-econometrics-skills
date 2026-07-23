# Skill Report — `regression-diagnostics`
**Built:** 2026-07-23 · **Theory:** Gauss-Markov battery (White 1980, BP 1979, BG, JB, RESET, Chow 1960) + Newey-West 1987 HAC.

## Validation — real cross-predictor case (NQ_ret ~ ES_ret lag-1, n=1,660 sessions, back-adjusted)
```
ES_ret_lag1   beta −0.172   p_naive 0.0000   p_HAC 0.0028   R² 2.2%
violations flagged: heteroskedasticity (BP/White p<.001) · serial correlation (BG p=.044)
VERDICT: edge survives HAC
```
A real finding: yesterday's ES move negatively predicts today's NQ (cross-serial reversion) — and
it survives the HAC correction that kills most such claims. Coheres with `randomness-tests`
(NQ VR<1) and the deployed MeanRev family. R² 2.2% = a lead for the gauntlet, not a strategy.

## Accuracy audit (synthetic ground truth) — 4/4
Planted heteroskedasticity caught (BP p=.017) · planted AR(1) errors caught (BG p≈0, DW 0.85) ·
planted collinearity caught (VIF 354) · **clean regression stays clean (zero false flags)**.
The audit also caught a real input-handling bug (numpy VIF path) before release.
