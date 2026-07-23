# Skill Report — `randomness-tests`
**Built:** 2026-07-23 · **Theory:** Ljung-Box 1978; Lo & MacKinlay 1988 (robust variance ratio); Wald-Wolfowitz runs.

## Validation — real CME NQ (back-adjusted)
```
NQ daily (1,661):   LB p≈0 all lags · VR2 0.874 (reversion flavor) · vol clustering p≈0
                    ARCHETYPE: MEAN-REVERSION-flavored -> fade/reversion archetypes
NQ 15m (150,644):   VR ≈ 1.0 (mean ~dead) · vol clustering p≈0
                    ARCHETYPE: weak mixed structure
```
The scanner independently recovered the program's hard-won conclusions in one pass: daily NQ
mean-reverts (the deployed MeanRev edge), intraday linear predictability is ~dead (why OTE needs
displacement structure, not autocorrelation), and volatility clusters everywhere (why GARCH works).

## Accuracy audit (synthetic ground truth) — 4/4
IID → correctly "nothing in the mean" · planted AR(1) → structure + momentum flavor (VR2 1.64) ·
GARCH-simulated returns → vol clustering flagged while the mean correctly stays unstructured.
