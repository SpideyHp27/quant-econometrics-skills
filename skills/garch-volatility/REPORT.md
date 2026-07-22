# Skill Report — `garch-volatility`

**Bundle:** Quant Econometrics Skills (skill 1 of 6) · **Built:** 2026-07-22
**Theory:** Engle ARCH (1982) + Bollerslev GARCH (1986), Nobel 2003. **Template:** milesdeutscher/garchmethod.

## What it does
Walk-forward GARCH(1,1) (Student-t, expanding window, zero look-ahead) → a 1-day-ahead **volatility** forecast → a **regime** label (calm/normal/storm by trailing-year percentile) → a **position-size multiplier** (`target_vol / forecast_vol`, capped [0.25×, 2.0×]). Forecasts the *size* of the next move, never the direction.

## What it does for the research
The risk/sizing layer for the whole book. Its headline job here is the no-stop mean-reversion problem: **MeanRev_NDX** runs ~3.7× leverage with no stop and blows up exactly when volatility spikes. GARCH sizing scales exposure *down* right then, and — critically — never touches the entry logic (tight stops kill the mean-reversion edge; vol-targeting doesn't). It also serves as a portfolio vol-target overlay and a regime gate for entry engines.

## Validation (real project data)
On real CME NQ daily, vol-targeting vs buy&hold (no-lookahead sizing):

```
MaxDD   38.5%  ->  24.2%     (down ~14 points)
Sharpe   0.52  ->   0.60     (up)
```

The drawdown reduction with a Sharpe *improvement* is the signature of correct vol-targeting: it removes risk in the storm regime without giving up the calm-regime returns.

## Caveat
Never a direction call. On its own it makes no money — it makes an existing directional edge survivable. Reports the drawdown honestly, every time.

## Files
- `../../scripts/garch_forecast.py` — tool (`--backtest`, `--json`; importable `walk_forward_garch` / `position_size` / `classify_regime`).
- `SKILL.md` — skill definition.
