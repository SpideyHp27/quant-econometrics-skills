---
name: randomness-tests
description: Is there ANY exploitable structure, or is this white noise? Ljung-Box portmanteau at multiple lags, Lo-MacKinlay variance ratios (heteroskedasticity-robust) at q=2/5/10, runs test, and a vol-clustering check on absolute returns — fused into a strategy-archetype verdict per series, momentum-flavored (VR>1), mean-reversion-flavored (VR<1), or nothing-in-the-mean. Use before fitting ANY model, for "is this random", "momentum or mean reversion", and multi-file universe scans.
---

# Randomness battery — the gate before every model

If returns are indistinguishable from white noise, every backtest on them is curve-fitting. This
runs the classic structure tests and answers the PRACTICAL question: which archetype (if any) does
this series support?

```bash
uv run scripts/randomness.py --csv A.parquet [B.csv ...] [--json]    # multi-file universe scan
```
VR>1 ⇒ momentum-flavored · VR<1 ⇒ mean-reversion-flavored · no rejections ⇒ don't fit directional
models. Separately reports **vol clustering** on |r| — the structure GARCH feeds on, present even
when the mean is dead.

## Validated on real CME NQ (see REPORT)
Daily: **mean-reversion-flavored** (VR2 0.874) + massive vol clustering — matching the deployed
MeanRev edge and the GARCH skill's food supply. 15m: mean ≈ dead (VR≈1.0) with extreme vol
clustering — exactly why high-frequency entries need *structure* (OTE displacement) rather than
autocorrelation.

## Caveat
Linear tests only — nonlinear/conditional structure (like OTE's) is invisible here. A "nothing"
verdict kills naive models, not all models. Refs: Ljung & Box 1978 · Lo & MacKinlay 1988.
