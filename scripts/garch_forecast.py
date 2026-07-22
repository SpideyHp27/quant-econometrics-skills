# /// script
# requires-python = ">=3.10"
# dependencies = ["arch>=6.0", "pandas>=2.0", "numpy>=1.24"]
# ///
"""garch_forecast.py — walk-forward GARCH(1,1) volatility forecast + vol-target position sizing.

Engle ARCH (1982) / Bollerslev GARCH (1986) — the framework that won the 2003 economics Nobel. Models
volatility CLUSTERING (calm begets calm, storms beget storms) and produces a 1-day-ahead forecast of the
SIZE of the next move. EXPANDING window, ZERO look-ahead: params re-estimated periodically on all data
before day t; between refits the variance recursion (sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1})
rolls forward on realized returns known at forecast time.

What this does NOT do: predict direction. It forecasts MAGNITUDE only. Pair it with a directional edge.

Inspired by github.com/milesdeutscher/garchmethod; clean implementation on the `arch` library.

Usage:
  uv run scripts/garch_forecast.py --csv prices.csv [--target-vol 0.15] [--backtest]
  uv run scripts/garch_forecast.py --csv prices.csv --json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from arch import arch_model


def load_prices(path, price_col):
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    dcol = next((c for c in ["date", "datetime", "time", "et", "timestamp"] if c in df.columns), None)
    tcol = "time" if ("time" in df.columns and dcol != "time") else None
    if dcol is not None:
        stamp = df[dcol].astype(str) + ((" " + df[tcol].astype(str)) if tcol else "")
        df.index = pd.to_datetime(stamp, errors="coerce", format="mixed")
    pc = price_col.lower()
    if pc not in df.columns:
        pc = "close" if "close" in df.columns else df.columns[-1]
    return df[pc].astype(float).dropna().sort_index()


def walk_forward_garch(prices, min_train=500, refit_every=20, ann=252):
    """DataFrame(ret, sigma_daily[1-day-ahead], sigma_ann). Strictly zero-lookahead."""
    ret = 100.0 * np.log(prices / prices.shift(1)).dropna()
    r = ret.values; n = len(r); idx = ret.index
    fc = np.full(n, np.nan)
    mu = omega = alpha = beta = None; last_fit = -10**9; s2_prev = np.var(r[:min_train])
    for t in range(min_train, n):
        if t - last_fit >= refit_every:
            res = arch_model(ret.iloc[:t], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
            pr = res.params
            mu, omega, alpha, beta = pr["mu"], pr["omega"], pr["alpha[1]"], pr["beta[1]"]
            last_fit = t
            s2_prev = float(np.asarray(res.conditional_volatility)[-1]) ** 2
        eps_prev = r[t - 1] - mu
        s2_t = omega + alpha * eps_prev ** 2 + beta * s2_prev
        fc[t] = np.sqrt(s2_t) / 100.0
        s2_prev = s2_t
    out = pd.DataFrame({"ret": ret.values / 100.0, "sigma_daily": fc}, index=idx).dropna()
    out["sigma_ann"] = out["sigma_daily"] * np.sqrt(ann)
    return out


def classify_regime(sigma, lookback=252):
    pct = sigma.rolling(lookback, min_periods=60).apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False)
    reg = pd.Series(np.where(pct < 0.33, "calm", np.where(pct > 0.66, "storm", "normal")), index=sigma.index)
    return reg, pct


def position_size(sigma_ann, target_vol, cap=(0.25, 2.0)):
    return (target_vol / sigma_ann).clip(cap[0], cap[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--price-col", default="close")
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--min-train", type=int, default=500); ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--ann", type=int, default=252); ap.add_argument("--out", default=None)
    ap.add_argument("--backtest", action="store_true"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    px = load_prices(a.csv, a.price_col)
    df = walk_forward_garch(px, a.min_train, a.refit_every, a.ann)
    df["regime"], df["vol_pct"] = classify_regime(df["sigma_ann"])
    df["size_mult"] = position_size(df["sigma_ann"], a.target_vol)
    last = df.iloc[-1]

    if a.json:
        print(json.dumps({
            "as_of": str(df.index[-1])[:10],
            "forecast_vol_daily_pct": round(last.sigma_daily * 100, 3),
            "forecast_vol_annualized_pct": round(last.sigma_ann * 100, 1),
            "vol_percentile_1y": round(last.vol_pct * 100, 1),
            "regime": last.regime,
            "position_size_multiplier": round(last.size_mult, 2),
            "target_vol_pct": a.target_vol * 100,
            "note": "GARCH forecasts magnitude (volatility), not direction."}, indent=2, default=float))
        return

    print("GARCH(1,1) walk-forward | %s..%s | %d obs" % (str(df.index[0])[:10], str(df.index[-1])[:10], len(df)))
    print("LATEST 1-day-ahead vol: daily %.3f%%  annualized %.1f%%  | regime=%s (pct %.0f%%) | size x%.2f (target %.0f%%)"
          % (last.sigma_daily * 100, last.sigma_ann * 100, last.regime, last.vol_pct * 100, last.size_mult, a.target_vol * 100))
    print("regime days:", df["regime"].value_counts().to_dict())

    if a.backtest:
        pos = df["size_mult"].shift(1).fillna(1.0)
        bh = df["ret"]; vt = df["ret"] * pos
        def stat(x):
            return dict(cagr=round((np.prod(1 + x) ** (a.ann / len(x)) - 1) * 100, 1),
                        sharpe=round(x.mean() / x.std() * np.sqrt(a.ann), 2),
                        maxdd=round((1 - (1 + x).cumprod() / (1 + x).cumprod().cummax()).max() * 100, 1))
        print("\nBACKTEST (buy&hold vs vol-targeted, no-lookahead sizing):")
        print("  buy&hold    :", stat(bh))
        print("  vol-targeted:", stat(vt), " <- Sharpe up / DD down if GARCH sizing helps")

    if a.out:
        df.to_csv(a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
